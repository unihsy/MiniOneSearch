"""Unified Generative Search SFT & Joint Self-Distillation (`Joint SDFT`) Trainer.

Supports:
  - Multi-GPU `torchrun` DDP BF16 full-parameter training across Qwen3-Base (0.6B / 1.7B / 4B / 8B)
  - 3-Stage Progressive Alignment (`stage1` -> `stage2` -> `stage3`) or Single-Stage ablations
  - Completion-Sliced Joint Self-Distillation (`Joint SDFT`):
      L = L_CE(student) + w_tea * L_CE(teacher) + w_kl * T^2 * KL(stopgrad(P_tea^T) || P_stu^T)
    computed strictly on completion logit positions (`~6` SID tokens), giving zero prompt-KL waste
    and zero online inference overhead!
  - Sharded Multi-GPU Constrained Beam Search Validation & Test Evaluation
"""
from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm

from minionesearch.decoding.kv_cache_engine import UserPrefixKVCacheEngine
from minionesearch.decoding.trie import GPUPrefixTrie, PrefixTrieLogitsProcessor
from minionesearch.evaluation import evaluate_predictions, summarize_latencies_ms
from minionesearch.sft.dataset import (
    PromptBuilder,
    SFTMultiStageDataset,
    build_stage1_alignment_records,
    build_stage2_cooccurrence_records,
    build_stage3_personalized_records,
    collate_sft_or_sdft_batch,
)
from minionesearch.sft.vocab_init import initialize_sid_embeddings_semantic
from minionesearch.utils import iter_jsonl, set_global_seed


@dataclass
class JointSDFTConfig:
    base_model: str = "Qwen/Qwen3-1.7B-Base"
    use_progressive_stages: bool = True
    use_joint_sdft: bool = True
    personalized: bool = True
    user_id_mode: str = "mu_seq"         # "mu_seq" (OneSearch-V1 §3.2.1) | "random_hash" (TIGER) | "none"
    use_query_sid: bool = True           # OneSearch-V1 Eq.(5) / V2 Eq.(2) live query SID `SID_q`
    explicit_cot_mode: bool = False      # OneSearch-V2 Table 3 `+ Direct Explicit CoT` ablation
    sdft_kl_weight: float = 0.15         # alpha_KL in OneSearch-V2 Eq.(5)
    rdrop_weight: float = 0.05           # alpha_R in OneSearch-V2 Eq.(6)
    focal_gamma: float = 1.0             # Focal CE gamma in OneSearch-V2 §3.3.3
    sdft_temperature: float = 1.0
    stage3_replay_ratio: float = 0.30    # Anti-forgetting Stage 1+2 replay in Stage 3
    stage12_epochs: int = 2
    stage3_epochs: int = 3
    batch_size: int = 16
    grad_accum_steps: int = 1
    lr: float = 1.5e-4
    max_prompt_tokens: int = 320
    eval_beams: int = 20
    seed: int = 42


def gather_completion_logits(logits: torch.Tensor, comp_logit_pos: torch.Tensor) -> torch.Tensor:
    """Slice `[B, Seq, V]` logits at exact target-predicting indices `[B, C]` -> `[B, C, V]`."""
    bsz, c_len = comp_logit_pos.shape
    v_size = logits.size(-1)
    safe_pos = comp_logit_pos.clamp(0, logits.size(1) - 1)
    idx_expanded = safe_pos.unsqueeze(-1).expand(bsz, c_len, v_size)
    return torch.gather(logits, dim=1, index=idx_expanded)


def compute_focal_completion_ce_loss(
    comp_logits: torch.Tensor,  # [B, C, V]
    labels: torch.Tensor,       # [B, Seq] with -100 on prompt
    comp_logit_pos: torch.Tensor,  # [B, C]
    comp_mask: torch.Tensor,       # [B, C] bool
    gamma: float = 1.0,
) -> torch.Tensor:
    """OneSearch-V2 §3.3.3: Focal Cross-Entropy over target SID tokens to mitigate head-SID imbalance."""
    # Target token at step j is at `comp_logit_pos + 1` in `labels`
    target_pos = (comp_logit_pos + 1).clamp(0, labels.size(1) - 1)
    comp_targets = torch.gather(labels, dim=1, index=target_pos)  # [B, C]
    valid = comp_mask & (comp_targets >= 0)
    safe_targets = comp_targets.clamp(min=0)

    logp = F.log_softmax(comp_logits.float(), dim=-1)
    tok_logp = torch.gather(logp, dim=-1, index=safe_targets.unsqueeze(-1)).squeeze(-1)
    tok_p = torch.exp(tok_logp).clamp(0.0, 1.0)
    focal_factor = ((1.0 - tok_p) ** gamma) if gamma > 0.0 else torch.ones_like(tok_p)
    tok_loss = -focal_factor * tok_logp

    mask_f = valid.float()
    return (tok_loss * mask_f).sum() / mask_f.sum().clamp(min=1.0)


def compute_rdrop_symmetric_kl(
    comp_logits_1: torch.Tensor,  # [B, C, V]
    comp_logits_2: torch.Tensor,  # [B, C, V]
    comp_mask: torch.Tensor,      # [B, C] bool
) -> torch.Tensor:
    """OneSearch-V2 §3.3.2 Eq. (6): Symmetric R-Drop KL consistency over valid completion positions."""
    lp1 = F.log_softmax(comp_logits_1.float(), dim=-1)
    lp2 = F.log_softmax(comp_logits_2.float(), dim=-1)
    p1 = torch.exp(lp1)
    p2 = torch.exp(lp2)
    kl_12 = F.kl_div(lp2, p1, reduction="none").sum(dim=-1)
    kl_21 = F.kl_div(lp1, p2, reduction="none").sum(dim=-1)
    sym_kl = 0.5 * (kl_12 + kl_21)
    mask_f = comp_mask.float()
    return (sym_kl * mask_f).sum() / mask_f.sum().clamp(min=1.0)


def compute_completion_kl_loss(
    student_comp_logits: torch.Tensor,  # [B, C, V]
    teacher_comp_logits: torch.Tensor,  # [B, C, V]
    comp_mask: torch.Tensor,            # [B, C] bool
    temperature: float = 1.0,
) -> torch.Tensor:
    """OneSearch-V2 §3.3.1 Eq. (4): `T^2 * KL(softmax(sg(z^T)/T) || softmax(z^S/T))`."""
    t = max(float(temperature), 1e-3)
    stu_logp = F.log_softmax(student_comp_logits.float() / t, dim=-1)
    with torch.no_grad():
        tea_prob = F.softmax(teacher_comp_logits.detach().float() / t, dim=-1)

    token_kl = F.kl_div(stu_logp, tea_prob, reduction="none").sum(dim=-1)
    mask_f = comp_mask.float()
    denom = mask_f.sum().clamp(min=1.0)
    return (t * t) * ((token_kl * mask_f).sum() / denom)


@torch.no_grad()
def evaluate_generative_search_ddp(
    model: nn.Module,
    tokenizer: Any,
    trie: GPUPrefixTrie,
    level_token_ids: Sequence[torch.LongTensor],
    sessions: Sequence[Mapping[str, Any]],
    item_sid_map: Mapping[int, str],
    corpus_records: Sequence[Mapping[str, Any]],
    device: torch.device,
    query_sid_map: Optional[Mapping[str, str]] = None,
    user_id_mode: str = "mu_seq",
    explicit_cot_mode: bool = False,
    personalized: bool = True,
    num_beams: int = 10,
    batch_size: int = 16,
    max_prompt_tokens: int = 320,
    rank: int = 0,
    world_size: int = 1,
) -> Tuple[Dict[str, Dict[str, float]], Dict[str, List[int]], Dict[str, float]]:
    """Sharded multi-GPU Constrained Beam Search evaluation with left-padded batched prompts."""
    from transformers.generation import LogitsProcessorList

    raw_model = model.module if isinstance(model, DDP) else model
    raw_model.eval()

    pb = PromptBuilder(user_id_mode=user_id_mode)
    eos_id = int(tokenizer.eos_token_id if tokenizer.eos_token_id is not None else tokenizer.pad_token_id)
    pad_id = int(tokenizer.pad_token_id if tokenizer.pad_token_id is not None else eos_id)
    vocab_size = int(raw_model.get_input_embeddings().weight.shape[0])

    my_sessions = list(sessions[rank::world_size])
    local_preds: Dict[str, List[int]] = {}
    local_latencies_ms: List[float] = []

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    t_wall0 = time.perf_counter()

    for st in tqdm(
        range(0, len(my_sessions), batch_size),
        desc=f"[GenEval beams={num_beams} cot={explicit_cot_mode}]",
        disable=(rank != 0),
        mininterval=5.0,
    ):
        batch_s = my_sessions[st : st + batch_size]
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        t_b0 = time.perf_counter()

        tok_rows: List[List[int]] = []
        for s in batch_s:
            u_pref, q_suf = pb.build_inference_pair(
                s,
                item_sid_map,
                query_sid_map=query_sid_map,
                item_to_codes=trie.item_to_codes,
                sid_vocab=trie.sid_vocab,
                personalized=personalized,
                explicit_cot_mode=explicit_cot_mode,
            )
            ids = tokenizer(u_pref + q_suf, add_special_tokens=False)["input_ids"][-max_prompt_tokens:]
            tok_rows.append(ids)

        if explicit_cot_mode:
            # OneSearch-V2 Table 3 `+ Direct Explicit CoT`: first generate text keywords freely,
            # then append/align at `"商品id:"` before constrained SID beam search!
            max_p0 = max(len(r) for r in tok_rows)
            bsz = len(batch_s)
            in0 = torch.full((bsz, max_p0), fill_value=pad_id, dtype=torch.long, device=device)
            am0 = torch.zeros((bsz, max_p0), dtype=torch.long, device=device)
            for i, r in enumerate(tok_rows):
                in0[i, max_p0 - len(r) :] = torch.tensor(r, dtype=torch.long, device=device)
                am0[i, max_p0 - len(r) :] = 1
            cot_out = raw_model.generate(
                input_ids=in0,
                attention_mask=am0,
                max_new_tokens=48,
                do_sample=False,
                pad_token_id=pad_id,
                eos_token_id=eos_id,
            )
            cot_texts = tokenizer.batch_decode(cot_out[:, max_p0:], skip_special_tokens=True)
            new_tok_rows: List[List[int]] = []
            for base_r, ct in zip(tok_rows, cot_texts):
                prefix_cot = ct.split("商品id:")[0].split("\n")[0].strip()
                bridged = f"{prefix_cot} 商品id:" if prefix_cot else "商品id:"
                b_ids = tokenizer(bridged, add_special_tokens=False)["input_ids"]
                new_tok_rows.append((base_r + b_ids)[-(max_prompt_tokens + 56) :])
            tok_rows = new_tok_rows

        max_p = max(len(r) for r in tok_rows)
        bsz = len(batch_s)
        input_ids = torch.full((bsz, max_p), fill_value=pad_id, dtype=torch.long, device=device)
        attn_mask = torch.zeros((bsz, max_p), dtype=torch.long, device=device)
        for i, r in enumerate(tok_rows):
            input_ids[i, max_p - len(r) :] = torch.tensor(r, dtype=torch.long, device=device)
            attn_mask[i, max_p - len(r) :] = 1

        proc = PrefixTrieLogitsProcessor(
            trie=trie,
            level_token_ids=level_token_ids,
            eos_token_id=eos_id,
            vocab_size=vocab_size,
            prompt_len=max_p,
            renormalize_logprobs=True,
        )

        out_seqs = raw_model.generate(
            input_ids=input_ids,
            attention_mask=attn_mask,
            max_new_tokens=trie.num_levels + 1,
            min_new_tokens=trie.num_levels,
            num_beams=num_beams,
            num_return_sequences=num_beams,
            do_sample=False,
            early_stopping=True,
            pad_token_id=pad_id,
            eos_token_id=eos_id,
            logits_processor=LogitsProcessorList([proc]),
        )

        gen_part = out_seqs[:, max_p : max_p + trie.num_levels]
        decoded = tokenizer.batch_decode(gen_part, skip_special_tokens=True)

        if device.type == "cuda":
            torch.cuda.synchronize(device)
        dt_ms = (time.perf_counter() - t_b0) * 1000.0 / max(1, bsz)
        local_latencies_ms.extend([dt_ms] * bsz)

        for i, s in enumerate(batch_s):
            sid_str = str(s["session_id"])
            beams_i = decoded[i * num_beams : (i + 1) * num_beams]
            local_preds[sid_str] = trie.decode_beams_to_ranked_items(beams_i, max_items=num_beams)

    total_wall = time.perf_counter() - t_wall0

    if world_size > 1 and dist.is_initialized():
        gathered_preds: List[Dict[str, List[int]]] = [{} for _ in range(world_size)]
        gathered_lats: List[List[float]] = [[] for _ in range(world_size)]
        dist.all_gather_object(gathered_preds, local_preds)
        dist.all_gather_object(gathered_lats, local_latencies_ms)
        merged_preds: Dict[str, List[int]] = {}
        merged_lats: List[float] = []
        for gp in gathered_preds:
            merged_preds.update(gp)
        for gl in gathered_lats:
            merged_lats.extend(gl)
    else:
        merged_preds = local_preds
        merged_lats = local_latencies_ms

    valid_corpus_ids = {int(r["item_id"]) for r in corpus_records}
    item_to_cat = {int(r["item_id"]): int(r.get("category_level2_id") or 0) for r in corpus_records}
    ks_eval = tuple(k for k in (1, 5, 10, 20, 50) if k <= num_beams) or (1, num_beams)
    metrics = evaluate_predictions(
        sessions=sessions,
        predictions=merged_preds,
        ks=ks_eval,
        item_to_cat=item_to_cat,
        valid_corpus_ids=valid_corpus_ids,
    )
    lat_summary = summarize_latencies_ms(merged_lats, len(sessions), total_wall)
    return metrics, merged_preds, lat_summary


def _run_training_loop(
    model: nn.Module,
    tokenizer: Any,
    records: Sequence[Mapping[str, str]],
    cfg: JointSDFTConfig,
    epochs: int,
    use_sdft: bool,
    device: torch.device,
    rank: int,
    world_size: int,
    stage_tag: str,
    lr_scale: float = 1.0,
) -> Dict[str, float]:
    from transformers import get_cosine_schedule_with_warmup

    if epochs <= 0 or not records:
        return {"loss": 0.0, "ce_loss": 0.0, "kl_loss": 0.0, "rdrop_loss": 0.0}

    ds = SFTMultiStageDataset(records)
    is_ddp = world_size > 1 and dist.is_initialized()
    sampler = DistributedSampler(ds, num_replicas=world_size, rank=rank, shuffle=True, seed=cfg.seed) if is_ddp else None

    if not use_sdft and stage_tag == "Stage1+2-Align" and device.type == "cuda":
        if cfg.grad_accum_steps > 1:
            eff_bs = min(48, cfg.batch_size * cfg.grad_accum_steps)
            eff_accum = 1
        elif cfg.batch_size <= 32:
            eff_bs = cfg.batch_size * 2
            eff_accum = 1
        else:
            eff_bs = cfg.batch_size
            eff_accum = cfg.grad_accum_steps
    else:
        eff_bs = cfg.batch_size
        eff_accum = cfg.grad_accum_steps
    loader = DataLoader(
        ds,
        batch_size=eff_bs,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=0,
        pin_memory=False,
        drop_last=(len(ds) >= eff_bs * world_size * 2),
        collate_fn=lambda b: collate_sft_or_sdft_batch(
            b,
            tokenizer=tokenizer,
            max_prompt_tokens=(min(cfg.max_prompt_tokens, 160) if stage_tag == "Stage1+2-Align" else cfg.max_prompt_tokens),
            max_target_tokens=(64 if cfg.explicit_cot_mode else 24),
            include_teacher=use_sdft,
        ),
    )

    steps_per_epoch = max(1, len(loader) // max(1, eff_accum))
    total_opt_steps = max(1, steps_per_epoch * epochs)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr * lr_scale, weight_decay=0.01)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=int(0.06 * total_opt_steps), num_training_steps=total_opt_steps
    )
    use_bf16 = device.type == "cuda" and torch.cuda.is_bf16_supported()

    last_stats = {"loss": 0.0, "ce_loss": 0.0, "kl_loss": 0.0, "rdrop_loss": 0.0}
    raw_m = model.module if isinstance(model, DDP) else model

    for ep in range(1, epochs + 1):
        if sampler is not None:
            sampler.set_epoch(ep)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        ep_loss = 0.0
        ep_ce = 0.0
        ep_kl = 0.0
        ep_rdrop = 0.0
        pbar = tqdm(loader, desc=f"[{stage_tag}] Ep {ep}/{epochs}", disable=(rank != 0), mininterval=5.0)

        for step_i, batch in enumerate(pbar, 1):
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attn_mask = batch["attention_mask"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)
            comp_pos = batch["comp_logit_pos"].to(device, non_blocking=True)
            c_mask = batch["comp_mask"].to(device, non_blocking=True)

            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_bf16):
                do_rdrop = use_sdft and (cfg.rdrop_weight > 0.0)
                if do_rdrop:
                    # Single fused `[2*B, Seq]` forward pass with explicit independent embedding + attention dropout
                    # so R-Drop is 100% guaranteed non-zero even on backbones whose default config had dropout=0.0!
                    cat_ids = torch.cat([input_ids, input_ids], dim=0)
                    cat_mask = torch.cat([attn_mask, attn_mask], dim=0)
                    emb_layer = raw_m.get_input_embeddings()
                    cat_embeds = F.dropout(emb_layer(cat_ids), p=0.06, training=True)
                    if cfg.focal_gamma > 0.0:
                        # Avoid allocating a 13.5GB fp32 full-sequence CE tensor inside ForCausalLM when using Focal CE on SID tokens!
                        cat_out = model(inputs_embeds=cat_embeds, attention_mask=cat_mask, use_cache=False)
                        logits_1, logits_2 = cat_out.logits.chunk(2, dim=0)
                        stu_c_logits = gather_completion_logits(logits_1, comp_pos)
                        stu_c_logits_2 = gather_completion_logits(logits_2, comp_pos)
                        del cat_out, logits_1, logits_2
                        ce_1 = compute_focal_completion_ce_loss(stu_c_logits, labels, comp_pos, c_mask, gamma=cfg.focal_gamma)
                        ce_2 = compute_focal_completion_ce_loss(stu_c_logits_2, labels, comp_pos, c_mask, gamma=cfg.focal_gamma)
                        ce_loss = 0.5 * (ce_1 + ce_2)
                    else:
                        cat_lbl = torch.cat([labels, labels], dim=0)
                        cat_out = model(inputs_embeds=cat_embeds, attention_mask=cat_mask, labels=cat_lbl, use_cache=False)
                        logits_1, logits_2 = cat_out.logits.chunk(2, dim=0)
                        stu_c_logits = gather_completion_logits(logits_1, comp_pos)
                        stu_c_logits_2 = gather_completion_logits(logits_2, comp_pos)
                        ce_loss = cat_out.loss
                        del cat_out, logits_1, logits_2
                else:
                    if cfg.focal_gamma > 0.0 and use_sdft:
                        stu_out = model(input_ids=input_ids, attention_mask=attn_mask, use_cache=False)
                        stu_c_logits = gather_completion_logits(stu_out.logits, comp_pos)
                        del stu_out
                        stu_c_logits_2 = None
                        ce_loss = compute_focal_completion_ce_loss(
                            stu_c_logits, labels, comp_pos, c_mask, gamma=cfg.focal_gamma
                        )
                    else:
                        stu_out = model(input_ids=input_ids, attention_mask=attn_mask, labels=labels, use_cache=False)
                        stu_c_logits = gather_completion_logits(stu_out.logits, comp_pos)
                        stu_c_logits_2 = None
                        ce_loss = stu_out.loss
                        del stu_out

                loss = ce_loss
                step_kl_val = 0.0
                step_rd_val = 0.0

                if use_sdft and "teacher_input_ids" in batch:
                    t_ids = batch["teacher_input_ids"].to(device, non_blocking=True)
                    t_mask = batch["teacher_attention_mask"].to(device, non_blocking=True)
                    t_pos = batch["teacher_comp_logit_pos"].to(device, non_blocking=True)

                    # OneSearch-V2 §3.3.1 Eq. (4): Teacher runs strictly under `torch.no_grad()` & `eval()`!
                    was_training = raw_m.training
                    raw_m.eval()
                    with torch.no_grad():
                        tea_out = raw_m(input_ids=t_ids, attention_mask=t_mask, use_cache=False)
                        tea_c_logits = gather_completion_logits(tea_out.logits, t_pos)
                        del tea_out
                    if was_training:
                        raw_m.train()

                    kl_loss = compute_completion_kl_loss(
                        stu_c_logits, tea_c_logits, c_mask, temperature=cfg.sdft_temperature
                    )
                    loss = loss + cfg.sdft_kl_weight * kl_loss
                    step_kl_val = float(kl_loss.detach().item())

                    if do_rdrop and stu_c_logits_2 is not None:
                        rdrop_loss = compute_rdrop_symmetric_kl(stu_c_logits, stu_c_logits_2, c_mask)
                        loss = loss + cfg.rdrop_weight * rdrop_loss
                        step_rd_val = float(rdrop_loss.detach().item())

                scaled_loss = loss / max(1, eff_accum)

            scaled_loss.backward()
            ep_loss += float(loss.detach().item())
            ep_ce += float(ce_loss.detach().item())
            ep_kl += step_kl_val
            ep_rdrop += step_rd_val

            if step_i % eff_accum == 0 or step_i == len(loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

        denom_n = float(max(1, len(loader)))
        last_stats = {
            "loss": round(ep_loss / denom_n, 4),
            "ce_loss": round(ep_ce / denom_n, 4),
            "kl_loss": round(ep_kl / denom_n, 4),
            "rdrop_loss": round(ep_rdrop / denom_n, 4),
        }
        if rank == 0:
            print(
                f"[{stage_tag}] Epoch {ep}/{epochs} | total={last_stats['loss']:.4f} "
                f"ce={last_stats['ce_loss']:.4f} kl={last_stats['kl_loss']:.4f} rdrop={last_stats['rdrop_loss']:.4f}",
                flush=True,
            )
        if is_ddp:
            dist.barrier()

    del optimizer, scheduler, loader, ds
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return last_stats


def train_generative_search_sft(
    bench_dir: str | Path,
    sid_dir: str | Path,
    out_dir: str | Path,
    cfg: JointSDFTConfig,
) -> Dict[str, Any]:
    """End-to-end Progressive SFT + Joint SDFT training and Constrained Beam Search evaluation."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    local_rank = int(os.environ.get("LOCAL_RANK", "-1"))
    is_ddp = local_rank >= 0 and int(os.environ.get("WORLD_SIZE", "1")) > 1
    if is_ddp:
        torch.cuda.set_device(local_rank)
        if not dist.is_initialized():
            from datetime import timedelta
            dist.init_process_group(backend="nccl", timeout=timedelta(minutes=45))
        device = torch.device("cuda", local_rank)
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        rank = 0
        world_size = 1

    set_global_seed(cfg.seed + rank)
    bpath = Path(bench_dir)
    spath = Path(sid_dir)
    opath = Path(out_dir)
    if rank == 0:
        opath.mkdir(parents=True, exist_ok=True)

    corpus_recs = list(iter_jsonl(bpath / "corpus.jsonl"))
    corpus_map = {int(r["item_id"]): r for r in corpus_recs}
    train_sessions = list(iter_jsonl(bpath / "train_sessions.jsonl"))
    val_sessions = list(iter_jsonl(bpath / "val_sessions.jsonl"))
    test_sessions = list(iter_jsonl(bpath / "test_sessions.jsonl"))

    trie = GPUPrefixTrie.load(spath / "trie.json")
    item_sid_map = {int(k): str(v) for k, v in json.loads((spath / "item_sid_map.json").read_text(encoding="utf-8")).items()}
    q_train_file = spath / "query_sid_map.json"
    q_all_file = spath / "query_sid_all_live.json"
    train_query_sid_map = json.loads(q_train_file.read_text(encoding="utf-8")) if (cfg.use_query_sid and q_train_file.is_file()) else None
    live_query_sid_map = (
        json.loads(q_all_file.read_text(encoding="utf-8"))
        if (cfg.use_query_sid and q_all_file.is_file())
        else train_query_sid_map
    )

    if rank == 0:
        _ = AutoTokenizer.from_pretrained(cfg.base_model, trust_remote_code=True)
        _ = AutoModelForCausalLM.from_pretrained(cfg.base_model, torch_dtype=torch.bfloat16, trust_remote_code=True)
    if is_ddp:
        dist.barrier()

    tokenizer = AutoTokenizer.from_pretrained(cfg.base_model, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype_load = torch.bfloat16 if (device.type == "cuda" and torch.cuda.is_bf16_supported()) else torch.float32
    extra_load_kwargs: Dict[str, Any] = {}
    if cfg.use_joint_sdft and cfg.rdrop_weight > 0.0:
        extra_load_kwargs["attention_dropout"] = 0.08
    try:
        base_lm = AutoModelForCausalLM.from_pretrained(
            cfg.base_model, torch_dtype=dtype_load, trust_remote_code=True, **extra_load_kwargs
        ).to(device)
    except TypeError:
        base_lm = AutoModelForCausalLM.from_pretrained(
            cfg.base_model, torch_dtype=dtype_load, trust_remote_code=True
        ).to(device)
    base_lm.config.use_cache = False
    if cfg.use_joint_sdft and cfg.rdrop_weight > 0.0:
        base_lm.config.attention_dropout = 0.08
        for mod in base_lm.modules():
            if hasattr(mod, "attention_dropout") and isinstance(getattr(mod, "attention_dropout"), float):
                mod.attention_dropout = 0.08
    if sum(p.numel() for p in base_lm.parameters()) >= 3_000_000_000:
        base_lm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    level_token_ids = initialize_sid_embeddings_semantic(
        model=base_lm,
        tokenizer=tokenizer,
        sid_vocab=trie.sid_vocab,
        corpus_records=corpus_recs,
        item_to_codes=trie.item_to_codes,
    )
    if is_ddp:
        for p in base_lm.parameters():
            dist.broadcast(p.data, src=0)
        model: nn.Module = DDP(base_lm, device_ids=[local_rank], find_unused_parameters=False)
    else:
        model = base_lm

    stage12_all: List[Dict[str, str]] = []
    if cfg.use_progressive_stages and cfg.stage12_epochs > 0:
        s1_recs = build_stage1_alignment_records(
            corpus_recs,
            item_sid_map,
            query_sid_map=train_query_sid_map,
            train_sessions=train_sessions,
            include_cot_tasks=True,
            max_items=120_000,
            seed=cfg.seed,
        )
        s2_recs = build_stage2_cooccurrence_records(
            train_sessions,
            corpus_recs,
            item_sid_map,
            query_sid_map=train_query_sid_map,
            max_records=260_000,
            seed=cfg.seed,
        )
        stage12_all = s1_recs + s2_recs
        if rank == 0:
            print(f"[SFT] Stage 1+2 records={len(stage12_all):,} (S1={len(s1_recs):,}, S2={len(s2_recs):,})", flush=True)
        _run_training_loop(
            model=model,
            tokenizer=tokenizer,
            records=stage12_all,
            cfg=cfg,
            epochs=cfg.stage12_epochs,
            use_sdft=False,
            device=device,
            rank=rank,
            world_size=world_size,
            stage_tag="Stage1+2-Align",
            lr_scale=1.0,
        )

    pb = PromptBuilder(user_id_mode=cfg.user_id_mode)
    s3_recs = build_stage3_personalized_records(
        train_sessions,
        corpus_map=corpus_map,
        item_sid_map=item_sid_map,
        query_sid_map=train_query_sid_map,
        item_to_codes=trie.item_to_codes,
        sid_vocab=trie.sid_vocab,
        prompt_builder=pb,
        personalized=cfg.personalized,
        explicit_cot_mode=cfg.explicit_cot_mode,
        seed=cfg.seed,
    )
    # Anti-forgetting multi-task replay of Stage 1+2 SID grounding records during Stage 3!
    if cfg.use_progressive_stages and cfg.stage3_replay_ratio > 0.0 and stage12_all and not cfg.explicit_cot_mode:
        import gc
        import random as _py_rng

        rr = _py_rng.Random(cfg.seed + 99)
        n_rep = min(len(stage12_all), int(len(s3_recs) * cfg.stage3_replay_ratio))
        sid_target_pool = [r for r in stage12_all if r["task"] in ("s1_i2sid", "s2_q2isid", "s2_pseudo_q2isid")]
        chosen_rep = rr.sample(sid_target_pool, min(n_rep, len(sid_target_pool)))
        for r in chosen_rep:
            s3_recs.append({"prompt": r["prompt"], "teacher_prompt": r["prompt"], "target": r["target"], "task": "s3_replay"})
        rr.shuffle(s3_recs)
        s1_recs.clear()
        s2_recs.clear()
        stage12_all.clear()
        del sid_target_pool, chosen_rep
    del train_sessions, corpus_map
    import gc as _gc
    _gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    if rank == 0:
        print(
            f"[SFT] Stage 3 total records={len(s3_recs):,} | sdft={cfg.use_joint_sdft} | uid={cfg.user_id_mode} | explicit_cot={cfg.explicit_cot_mode}",
            flush=True,
        )

    final_stats = _run_training_loop(
        model=model,
        tokenizer=tokenizer,
        records=s3_recs,
        cfg=cfg,
        epochs=cfg.stage3_epochs,
        use_sdft=(cfg.use_joint_sdft and not cfg.explicit_cot_mode),
        device=device,
        rank=rank,
        world_size=world_size,
        stage_tag="Stage3-OneSearchV2-SDFT" if cfg.use_joint_sdft else "Stage3-Pers-CE",
        lr_scale=0.5 if cfg.use_progressive_stages else 1.0,
    )

    ckpt_dir = opath / "checkpoint"
    if rank == 0:
        raw_save = model.module if isinstance(model, DDP) else model
        raw_save.config.use_cache = True
        raw_save.save_pretrained(ckpt_dir)
        tokenizer.save_pretrained(ckpt_dir)
        (opath / "prompt_config.json").write_text(
            json.dumps({"user_id_mode": cfg.user_id_mode, "use_query_sid": cfg.use_query_sid}, indent=2),
            encoding="utf-8",
        )
    if is_ddp:
        dist.barrier()

    test_metrics, test_preds, test_lat = evaluate_generative_search_ddp(
        model=model,
        tokenizer=tokenizer,
        trie=trie,
        level_token_ids=level_token_ids,
        sessions=test_sessions,
        item_sid_map=item_sid_map,
        corpus_records=corpus_recs,
        device=device,
        query_sid_map=live_query_sid_map,
        user_id_mode=cfg.user_id_mode,
        explicit_cot_mode=cfg.explicit_cot_mode,
        personalized=cfg.personalized,
        num_beams=cfg.eval_beams,
        batch_size=16,
        max_prompt_tokens=cfg.max_prompt_tokens,
        rank=rank,
        world_size=world_size,
    )

    summary: Dict[str, Any] = {}
    if rank == 0:
        sample_pairs = [
            pb.build_inference_pair(
                s,
                item_sid_map,
                query_sid_map=live_query_sid_map,
                item_to_codes=trie.item_to_codes,
                sid_vocab=trie.sid_vocab,
                personalized=cfg.personalized,
            )
            for s in test_sessions[: min(100, len(test_sessions))]
        ]
        raw_eval = model.module if isinstance(model, DDP) else model
        raw_eval.config.use_cache = True
        kv_engine = UserPrefixKVCacheEngine(raw_eval, tokenizer, trie, level_token_ids, device=device)
        _, lat_no_kv = kv_engine.benchmark_online_latency(sample_pairs, num_beams=10, use_prefix_cache=False)
        _, lat_with_kv = kv_engine.benchmark_online_latency(sample_pairs, num_beams=10, use_prefix_cache=True)

        summary = {
            "config": asdict(cfg),
            "final_train_loss": final_stats["loss"],
            "stage3_loss_diagnostics": final_stats,
            "test_metrics": test_metrics,
            "batched_eval_latency": test_lat,
            "online_single_query_no_kv_cache": lat_no_kv,
            "online_single_query_user_prefix_kv_cache": lat_with_kv,
        }
        (opath / "sft_eval_summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        (opath / "test_predictions.json").write_text(
            json.dumps(test_preds, ensure_ascii=False), encoding="utf-8"
        )
        print(f"[SFT Done] Test Overall: {test_metrics.get('overall', {})}", flush=True)
        print(f"[KV-Cache Latency] No-Cache P50={lat_no_kv['p50_ms']:.2f}ms -> Prefix-Cached P50={lat_with_kv['p50_ms']:.2f}ms", flush=True)

    if is_ddp:
        dist.barrier()
        dist.destroy_process_group()
    return summary


def main() -> None:
    p = argparse.ArgumentParser(description="Train & Evaluate Unified Generative Search SFT / OneSearch-V2 SDFT")
    p.add_argument("--bench_dir", required=True)
    p.add_argument("--sid_dir", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--base_model", default="Qwen/Qwen3-1.7B-Base")
    p.add_argument("--use_progressive_stages", type=int, default=1, choices=[0, 1])
    p.add_argument("--use_joint_sdft", type=int, default=1, choices=[0, 1])
    p.add_argument("--personalized", type=int, default=1, choices=[0, 1])
    p.add_argument("--user_id_mode", default="mu_seq", choices=["mu_seq", "random_hash", "none"])
    p.add_argument("--use_query_sid", type=int, default=1, choices=[0, 1])
    p.add_argument("--explicit_cot_mode", type=int, default=0, choices=[0, 1])
    p.add_argument("--sdft_kl_weight", type=float, default=0.15)
    p.add_argument("--rdrop_weight", type=float, default=0.05)
    p.add_argument("--focal_gamma", type=float, default=1.0)
    p.add_argument("--sdft_temperature", type=float, default=1.0)
    p.add_argument("--stage3_replay_ratio", type=float, default=0.30)
    p.add_argument("--stage12_epochs", type=int, default=2)
    p.add_argument("--stage3_epochs", type=int, default=3)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--grad_accum_steps", type=int, default=1)
    p.add_argument("--lr", type=float, default=1.5e-4)
    p.add_argument("--eval_beams", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()

    cfg = JointSDFTConfig(
        base_model=a.base_model,
        use_progressive_stages=bool(a.use_progressive_stages),
        use_joint_sdft=bool(a.use_joint_sdft),
        personalized=bool(a.personalized),
        user_id_mode=a.user_id_mode,
        use_query_sid=bool(a.use_query_sid),
        explicit_cot_mode=bool(a.explicit_cot_mode),
        sdft_kl_weight=a.sdft_kl_weight,
        rdrop_weight=a.rdrop_weight,
        focal_gamma=a.focal_gamma,
        sdft_temperature=a.sdft_temperature,
        stage3_replay_ratio=a.stage3_replay_ratio,
        stage12_epochs=a.stage12_epochs,
        stage3_epochs=a.stage3_epochs,
        batch_size=a.batch_size,
        grad_accum_steps=a.grad_accum_steps,
        lr=a.lr,
        eval_beams=a.eval_beams,
        seed=a.seed,
    )
    train_generative_search_sft(a.bench_dir, a.sid_dir, a.out_dir, cfg)


if __name__ == "__main__":
    main()
