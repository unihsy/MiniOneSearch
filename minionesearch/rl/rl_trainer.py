"""Self-Contained Distributed Search-RL Post-Training Runner (`Table 4` & `Table 1`).

Supports all 4 post-training methods starting from an SFT/SDFT checkpoint:
  - `listwise_dpo`
  - `vanilla_grpo`
  - `tpma_grpo`
  - `sp_tpma_grpo` (Ours: Ancestor-Consistent & Scale-Preserving Prefix-Tree GRPO)

Key Engineering Highlights:
  1. Uses `PrefixTrieLogitsProcessor` during online RL rollout so 100% of `G` sampled/beam
     candidates per query are valid, unique in-corpus items (no wasted invalid rollouts!).
  2. Co-locates frozen Reference Model + 4-Tier Relevance Cross-Encoder Judge in BF16 on
     each GPU (easily fitting in <25 GB VRAM on MI300X / A100 / 4090).
  3. Includes mild SFT positive anchor loss (`sft_anchor_weight = 0.08`) to stabilize
     ranking calibration while optimizing listwise/groupwise search rewards.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from tqdm import tqdm
from transformers.generation import LogitsProcessorList

from minionesearch.baselines.relevance_rm import RelevanceScorer
from minionesearch.decoding.trie import GPUPrefixTrie, PrefixTrieLogitsProcessor
from minionesearch.rl.algorithms import (
    compute_listwise_dpo_loss,
    compute_sp_tpma_advantages,
    compute_tpma_original_advantages,
    compute_vanilla_grpo_advantages,
    ppo_clipped_kl_surrogate_loss,
)
from minionesearch.rl.rewards import CompositeRewardWeights, SearchRewardEngine
from minionesearch.sft.dataset import PromptBuilder, collate_sft_or_sdft_batch
from minionesearch.sft.sft_trainer import evaluate_generative_search_ddp
from minionesearch.utils import iter_jsonl, set_global_seed


@dataclass
class RLTrainConfig:
    algo: str = "sp_tpma_grpo"  # "listwise_dpo" | "vanilla_grpo" | "tpma_grpo" | "sp_tpma_grpo"
    num_generations: int = 12
    rollout_mode: str = "hybrid"  # "beam" or "hybrid" (half constrained sampling + half beam backfill)
    rollout_temperature: float = 1.10
    max_rl_sessions: int = 12_000
    epochs: int = 1
    ppo_epochs: int = 2  # Multi-step inner updates per rollout batch so `cur_lp != old_lp` & PPO clip fires!
    use_ppo_clip: bool = True
    inject_gt_anchor: bool = True
    rl_loss_weight: float = 1.0  # Set 0.0 for SFT-Anchor-Only control ablation
    w_item: float = 0.50  # Identical item-reward weight across TPMA-GRPO and SP-TPMA-GRPO!
    batch_size: int = 8  # prompts per GPU step (`8 prompts * 12 rollouts = 96 sequences/GPU`)
    lr: float = 1.5e-5
    kl_beta: float = 0.02
    dpo_beta: float = 0.10
    sft_anchor_weight: float = 0.08
    max_prompt_tokens: int = 288
    eval_beams: int = 10
    seed: int = 42


class SessionRLDataset(Dataset):
    def __init__(self, sessions: Sequence[Mapping[str, Any]]):
        self.sessions = list(sessions)

    def __len__(self) -> int:
        return len(self.sessions)

    def __getitem__(self, idx: int) -> Mapping[str, Any]:
        return self.sessions[idx]


def compute_per_token_logps(
    model: nn.Module,
    prompt_ids: torch.Tensor,      # [N, P]
    prompt_mask: torch.Tensor,     # [N, P]
    comp_ids: torch.Tensor,        # [N, C]
    comp_mask: torch.Tensor,       # [N, C]
) -> torch.Tensor:
    """Compute per-token log-probabilities `[N, C]` of `comp_ids` conditioned on `prompt_ids`."""
    full_ids = torch.cat([prompt_ids, comp_ids], dim=1)
    full_mask = torch.cat([prompt_mask, comp_mask], dim=1)
    out = model(input_ids=full_ids, attention_mask=full_mask, use_cache=False)
    p_len = prompt_ids.size(1)
    c_len = comp_ids.size(1)
    comp_logits = out.logits[:, p_len - 1 : p_len - 1 + c_len, :]  # [N, C, V]
    log_probs = F.log_softmax(comp_logits.float(), dim=-1)
    return torch.gather(log_probs, dim=-1, index=comp_ids.unsqueeze(-1)).squeeze(-1)


@torch.no_grad()
def generate_constrained_group_rollouts(
    raw_policy: Any,
    tokenizer: Any,
    trie: GPUPrefixTrie,
    level_token_ids: Sequence[torch.LongTensor],
    prompt_ids: torch.Tensor,      # [B, P] left-padded
    prompt_mask: torch.Tensor,     # [B, P]
    num_generations: int = 12,
    mode: str = "hybrid",
    temperature: float = 1.10,
    gt_sid_strs: Optional[Sequence[Optional[str]]] = None,
) -> Tuple[torch.LongTensor, torch.LongTensor, List[List[Optional[int]]]]:
    """Generate `G` valid, unique in-corpus candidate SIDs per prompt on GPU!

    Returns:
        `comp_token_ids`: `[B * G, L]` global token IDs
        `rollout_codes`:  `[B, G, L]` local code indices `[0..K_d-1]`
        `rollout_items`:  `list[list[int]]` of shape `[B][G]`
    """
    raw_policy.eval()
    device = prompt_ids.device
    bsz, p_len = prompt_ids.shape
    l_sid = trie.num_levels
    eos_id = int(tokenizer.eos_token_id if tokenizer.eos_token_id is not None else tokenizer.pad_token_id)
    pad_id = int(tokenizer.pad_token_id if tokenizer.pad_token_id is not None else eos_id)
    vocab_size = int(raw_policy.get_input_embeddings().weight.shape[0])

    # 1. Constrained Beam Search for exact top-G unique high-probability items
    beam_proc = PrefixTrieLogitsProcessor(
        trie=trie,
        level_token_ids=level_token_ids,
        eos_token_id=eos_id,
        vocab_size=vocab_size,
        prompt_len=p_len,
        renormalize_logprobs=True,
    )
    beam_out = raw_policy.generate(
        input_ids=prompt_ids,
        attention_mask=prompt_mask,
        max_new_tokens=l_sid + 1,
        min_new_tokens=l_sid,
        num_beams=num_generations,
        num_return_sequences=num_generations,
        do_sample=False,
        early_stopping=True,
        pad_token_id=pad_id,
        eos_token_id=eos_id,
        logits_processor=LogitsProcessorList([beam_proc]),
    )
    beam_gen = beam_out[:, p_len : p_len + l_sid].view(bsz, num_generations, l_sid)

    if mode == "hybrid" and num_generations >= 4:
        # Also draw `G // 2` stochastic constrained samples for exploration & merge unique!
        n_samp = num_generations // 2
        samp_proc = PrefixTrieLogitsProcessor(
            trie=trie,
            level_token_ids=level_token_ids,
            eos_token_id=eos_id,
            vocab_size=vocab_size,
            prompt_len=p_len,
            renormalize_logprobs=True,
        )
        samp_out = raw_policy.generate(
            input_ids=prompt_ids,
            attention_mask=prompt_mask,
            max_new_tokens=l_sid + 1,
            min_new_tokens=l_sid,
            do_sample=True,
            temperature=temperature,
            num_return_sequences=n_samp,
            pad_token_id=pad_id,
            eos_token_id=eos_id,
            logits_processor=LogitsProcessorList([samp_proc]),
        )
        samp_gen = samp_out[:, p_len : p_len + l_sid].view(bsz, n_samp, l_sid)

        merged_rows: List[torch.Tensor] = []
        for b in range(bsz):
            # Interleave top-2 beams + exploratory samples + remaining beams, deduplicating exact token rows
            cands_cat = torch.cat([beam_gen[b, :2], samp_gen[b], beam_gen[b, 2:]], dim=0)
            chosen_rows: List[torch.Tensor] = []
            seen_tups = set()
            for r_idx in range(cands_cat.size(0)):
                tup = tuple(int(x) for x in cands_cat[r_idx].tolist())
                if tup not in seen_tups:
                    seen_tups.add(tup)
                    chosen_rows.append(cands_cat[r_idx])
                if len(chosen_rows) == num_generations:
                    break
            while len(chosen_rows) < num_generations:
                chosen_rows.append(beam_gen[b, len(chosen_rows) % beam_gen.size(1)])
            merged_rows.append(torch.stack(chosen_rows, dim=0))
        final_gen = torch.stack(merged_rows, dim=0)  # [B, G, L]
    else:
        final_gen = beam_gen

    # Optional positive anchor injection (controlled via `cfg.inject_gt_anchor`):
    # When enabled and none of the G sampled/beam candidates hit the primary positive SID,
    # replaces slot `G - 1` and tracks the exact injection count for transparent reporting.
    n_injected = 0
    if gt_sid_strs is not None:
        final_gen = final_gen.clone()
        for b in range(bsz):
            gt_s = gt_sid_strs[b]
            if not gt_s:
                continue
            gt_tids = tokenizer(gt_s, add_special_tokens=False)["input_ids"][:l_sid]
            if len(gt_tids) != l_sid:
                continue
            gt_row = torch.tensor(gt_tids, dtype=final_gen.dtype, device=device)
            already_hit = bool((final_gen[b] == gt_row.unsqueeze(0)).all(dim=-1).any().item())
            if not already_hit:
                final_gen[b, -1] = gt_row
                n_injected += 1

    flat_gen = final_gen.reshape(bsz * num_generations, l_sid)
    decoded_strs = tokenizer.batch_decode(flat_gen, skip_special_tokens=True)

    rollout_items: List[List[Optional[int]]] = [[] for _ in range(bsz)]
    rollout_codes_np = torch.zeros((bsz, num_generations, l_sid), dtype=torch.long)

    for idx, s_str in enumerate(decoded_strs):
        b = idx // num_generations
        g = idx % num_generations
        iid = trie.decode_sid_str_to_item(s_str)
        rollout_items[b].append(iid)
        codes_tup = trie.sid_vocab.str_to_codes(s_str)
        if codes_tup is not None:
            rollout_codes_np[b, g] = torch.tensor(codes_tup, dtype=torch.long)

    return flat_gen, rollout_codes_np.to(device), rollout_items, n_injected


def train_generative_search_rl(
    bench_dir: str | Path,
    sid_dir: str | Path,
    sft_ckpt_dir: str | Path,
    out_dir: str | Path,
    cfg: RLTrainConfig,
    rel_rm_ckpt: Optional[str | Path] = None,
) -> Dict[str, Any]:
    """Run Search-RL post-training (`listwise_dpo` | `vanilla_grpo` | `tpma_grpo` | `sp_tpma_grpo`)."""
    from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup

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
    train_sessions_all = list(iter_jsonl(bpath / "train_sessions.jsonl"))
    val_sessions = list(iter_jsonl(bpath / "val_sessions.jsonl"))
    test_sessions = list(iter_jsonl(bpath / "test_sessions.jsonl"))

    # Sample most informative recent/diverse RL sessions up to `cfg.max_rl_sessions`
    rng = random.Random(cfg.seed)
    rl_sessions = list(train_sessions_all)
    rng.shuffle(rl_sessions)
    if cfg.max_rl_sessions > 0 and len(rl_sessions) > cfg.max_rl_sessions:
        rl_sessions = rl_sessions[: cfg.max_rl_sessions]

    trie = GPUPrefixTrie.load(spath / "trie.json")
    item_sid_map = {int(k): str(v) for k, v in json.loads((spath / "item_sid_map.json").read_text(encoding="utf-8")).items()}
    q_all_file = spath / "query_sid_all_live.json"
    q_train_file = spath / "query_sid_map.json"
    live_query_sid_map = (
        json.loads(q_all_file.read_text(encoding="utf-8"))
        if q_all_file.is_file()
        else (json.loads(q_train_file.read_text(encoding="utf-8")) if q_train_file.is_file() else None)
    )
    p_cfg_file = Path(sft_ckpt_dir) / "prompt_config.json"
    if not p_cfg_file.is_file():
        p_cfg_file = Path(sft_ckpt_dir).parent / "prompt_config.json"
    p_cfg = json.loads(p_cfg_file.read_text(encoding="utf-8")) if p_cfg_file.is_file() else {}
    user_id_mode = str(p_cfg.get("user_id_mode", "mu_seq"))
    use_query_sid = bool(p_cfg.get("use_query_sid", True))
    if not use_query_sid:
        live_query_sid_map = None
    all_corpus_ids = [int(r["item_id"]) for r in corpus_recs]

    tokenizer = AutoTokenizer.from_pretrained(sft_ckpt_dir, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    level_token_ids = trie.sid_vocab.register_with_tokenizer(tokenizer)

    dtype_load = torch.bfloat16 if (device.type == "cuda" and torch.cuda.is_bf16_supported()) else torch.float32
    policy_base = AutoModelForCausalLM.from_pretrained(
        sft_ckpt_dir, torch_dtype=dtype_load, trust_remote_code=True
    ).to(device)
    if sum(p.numel() for p in policy_base.parameters()) >= 3_000_000_000:
        policy_base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    ref_model = AutoModelForCausalLM.from_pretrained(
        sft_ckpt_dir, torch_dtype=dtype_load, trust_remote_code=True
    ).to(device)
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad = False

    rel_scorer = None
    if rel_rm_ckpt:
        if not Path(rel_rm_ckpt).exists():
            raise FileNotFoundError(f"Explicitly requested --rel_rm_ckpt does not exist: {rel_rm_ckpt}")
        rel_scorer = RelevanceScorer(rel_rm_ckpt, device=device)
    reward_engine = SearchRewardEngine(corpus_recs, rel_scorer=rel_scorer)

    model: nn.Module = DDP(policy_base, device_ids=[local_rank], find_unused_parameters=False) if is_ddp else policy_base

    ds = SessionRLDataset(rl_sessions)
    sampler = DistributedSampler(ds, num_replicas=world_size, rank=rank, shuffle=True, seed=cfg.seed) if is_ddp else None
    loader = DataLoader(
        ds,
        batch_size=cfg.batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        collate_fn=lambda x: list(x),
        drop_last=(len(ds) >= cfg.batch_size * world_size * 2),
    )

    inner_k = 1 if cfg.algo == "listwise_dpo" else max(1, cfg.ppo_epochs)
    total_steps = max(1, len(loader) * cfg.epochs * inner_k)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=0.01)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=int(0.08 * total_steps), num_training_steps=total_steps
    )
    use_bf16 = device.type == "cuda" and torch.cuda.is_bf16_supported()
    pb = PromptBuilder(user_id_mode=user_id_mode)
    eos_id = int(tokenizer.eos_token_id if tokenizer.eos_token_id is not None else tokenizer.pad_token_id)
    pad_id = int(tokenizer.pad_token_id if tokenizer.pad_token_id is not None else eos_id)

    step_logs: List[Dict[str, float]] = []
    val_history: List[Dict[str, Any]] = []
    best_ckpt_tag = "none"
    total_rollout_prompts = 0
    total_injected_prompts = 0
    ckpt_dir = opath / "checkpoint"
    best_val_score = -1.0
    eval_interval = max(25, len(loader) // 2)

    def _maybe_save_best_on_val(tag: str) -> None:
        nonlocal best_val_score, best_ckpt_tag
        if not val_sessions:
            return
        val_m, _, _ = evaluate_generative_search_ddp(
            model=model,
            tokenizer=tokenizer,
            trie=trie,
            level_token_ids=level_token_ids,
            sessions=val_sessions,
            item_sid_map=item_sid_map,
            corpus_records=corpus_recs,
            device=device,
            query_sid_map=live_query_sid_map,
            user_id_mode=user_id_mode,
            personalized=True,
            num_beams=10,
            batch_size=16,
            max_prompt_tokens=cfg.max_prompt_tokens,
            rank=rank,
            world_size=world_size,
        )
        cur_ndcg = float(val_m.get("overall", {}).get("NDCG@10", 0.0))
        cur_rec = float(val_m.get("overall", {}).get("Recall@10", 0.0))
        is_new_best = cur_ndcg >= best_val_score
        if rank == 0:
            print(f"[RL:{cfg.algo} ValCheck {tag}] val_NDCG@10={cur_ndcg * 100:.2f}% (best={best_val_score * 100:.2f}%)", flush=True)
            val_history.append({
                "tag": tag,
                "val_NDCG@10": round(cur_ndcg, 6),
                "val_Recall@10": round(cur_rec, 6),
                "is_new_best": bool(is_new_best),
            })
            if is_new_best:
                best_val_score = cur_ndcg
                best_ckpt_tag = tag
                raw_s = model.module if isinstance(model, DDP) else model
                raw_s.config.use_cache = True
                raw_s.save_pretrained(ckpt_dir)
                tokenizer.save_pretrained(ckpt_dir)
        else:
            if is_new_best:
                best_val_score = cur_ndcg
                best_ckpt_tag = tag
        if is_ddp:
            dist.barrier()

    # Baseline val check at step 0 before updates
    _maybe_save_best_on_val("step0_init")

    for ep in range(1, cfg.epochs + 1):
        if sampler is not None:
            sampler.set_epoch(ep)
        pbar = tqdm(loader, desc=f"[RL:{cfg.algo}] Ep {ep}/{cfg.epochs}", disable=(rank != 0), mininterval=5.0)

        for step_idx, batch_sess in enumerate(pbar, 1):
            bsz = len(batch_sess)
            # Left-pad prompts for generation & log-prob calculation
            p_rows: List[List[int]] = []
            sft_anchor_rows: List[Dict[str, str]] = []
            gt_code_sets: List[List[Tuple[int, ...]]] = []

            for s in batch_sess:
                u_pref, q_suf = pb.build_inference_pair(
                    s,
                    item_sid_map,
                    query_sid_map=live_query_sid_map,
                    item_to_codes=trie.item_to_codes,
                    sid_vocab=trie.sid_vocab,
                    personalized=True,
                )
                p_str = u_pref + q_suf
                p_ids = tokenizer(p_str, add_special_tokens=False)["input_ids"][-cfg.max_prompt_tokens :]
                p_rows.append(p_ids)

                pos_ids = list(dict.fromkeys((s.get("purchased_item_ids") or []) + (s.get("clicked_item_ids") or [])))
                pos_codes = [trie.item_to_codes[int(iid)] for iid in pos_ids if int(iid) in trie.item_to_codes]
                gt_code_sets.append(pos_codes)
                if pos_ids and int(pos_ids[0]) in item_sid_map:
                    sft_anchor_rows.append({"prompt": p_str, "target": item_sid_map[int(pos_ids[0])]})

            max_p = max(len(r) for r in p_rows)
            prompt_ids = torch.full((bsz, max_p), fill_value=pad_id, dtype=torch.long, device=device)
            prompt_mask = torch.zeros((bsz, max_p), dtype=torch.long, device=device)
            for i, r in enumerate(p_rows):
                prompt_ids[i, max_p - len(r) :] = torch.tensor(r, dtype=torch.long, device=device)
                prompt_mask[i, max_p - len(r) :] = 1

            raw_pol = model.module if isinstance(model, DDP) else model

            if cfg.algo == "listwise_dpo":
                # Build `[B, 1 + K]` candidate SIDs (`1` positive + `K=num_generations-1` hard negatives)
                k_neg = max(3, cfg.num_generations - 1)
                cand_sids_flat: List[List[int]] = []
                for s in batch_sess:
                    pos_ids = list(dict.fromkeys((s.get("purchased_item_ids") or []) + (s.get("clicked_item_ids") or [])))
                    pos_set = {int(x) for x in pos_ids}
                    pos_iid = int(pos_ids[0])
                    imp_negs = [int(x) for x in (s.get("impressed_item_ids") or []) if int(x) in item_sid_map and int(x) not in pos_set]
                    chosen_negs = imp_negs[:k_neg]
                    while len(chosen_negs) < k_neg:
                        rc = rng.choice(all_corpus_ids)
                        if rc not in pos_set:
                            chosen_negs.append(rc)
                    for cid in [pos_iid] + chosen_negs:
                        t_ids = tokenizer(item_sid_map[cid], add_special_tokens=False)["input_ids"][: trie.num_levels]
                        cand_sids_flat.append(t_ids)

                n_group = 1 + k_neg
                rep_p_ids = prompt_ids.repeat_interleave(n_group, dim=0)
                rep_p_mask = prompt_mask.repeat_interleave(n_group, dim=0)
                comp_ids = torch.tensor(cand_sids_flat, dtype=torch.long, device=device)
                comp_mask = torch.ones_like(comp_ids)

                model.train()
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_bf16):
                    pol_tok_lp = compute_per_token_logps(model, rep_p_ids, rep_p_mask, comp_ids, comp_mask)
                    with torch.no_grad():
                        ref_tok_lp = compute_per_token_logps(ref_model, rep_p_ids, rep_p_mask, comp_ids, comp_mask)
                    pol_seq_lp = pol_tok_lp.sum(dim=-1).view(bsz, n_group)
                    ref_seq_lp = ref_tok_lp.sum(dim=-1).view(bsz, n_group)
                    loss, step_m = compute_listwise_dpo_loss(pol_seq_lp, ref_seq_lp, beta=cfg.dpo_beta)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                step_logs.append(step_m)
                if step_idx % eval_interval == 0:
                    _maybe_save_best_on_val(f"ep{ep}_step{step_idx}")
                continue

            # Online Constrained Group Rollout (`G` valid unique items per prompt!)
            g_size = cfg.num_generations
            gt_primary_sids: Optional[List[Optional[str]]] = None
            if cfg.inject_gt_anchor:
                gt_primary_sids = []
                for s in batch_sess:
                    pos_ids = list(dict.fromkeys((s.get("purchased_item_ids") or []) + (s.get("clicked_item_ids") or [])))
                    gt_primary_sids.append(item_sid_map.get(int(pos_ids[0])) if pos_ids else None)

            comp_ids, rollout_codes, rollout_items, n_inj = generate_constrained_group_rollouts(
                raw_policy=raw_pol,
                tokenizer=tokenizer,
                trie=trie,
                level_token_ids=level_token_ids,
                prompt_ids=prompt_ids,
                prompt_mask=prompt_mask,
                num_generations=g_size,
                mode=cfg.rollout_mode,
                temperature=cfg.rollout_temperature,
                gt_sid_strs=gt_primary_sids,
            )
            total_rollout_prompts += bsz
            total_injected_prompts += n_inj
            comp_mask = torch.ones_like(comp_ids)
            rep_p_ids = prompt_ids.repeat_interleave(g_size, dim=0)
            rep_p_mask = prompt_mask.repeat_interleave(g_size, dim=0)

            # Score composite search rewards `[B, G]`
            item_rewards, rew_m = reward_engine.score_rollout_groups(batch_sess, rollout_items)
            item_rewards = item_rewards.to(device)

            # Compute token-level advantages & gates `[B, G, L]` with IDENTICAL `w_item` across TPMA & SP-TPMA!
            extra_m: Dict[str, float] = {}
            if cfg.algo == "vanilla_grpo":
                tok_adv, tok_gates = compute_vanilla_grpo_advantages(item_rewards, num_levels=trie.num_levels)
            elif cfg.algo == "tpma_grpo":
                tok_adv, tok_gates = compute_tpma_original_advantages(
                    rollout_codes, gt_code_sets, item_rewards, w_item=cfg.w_item
                )
            elif cfg.algo == "sp_tpma_grpo":
                tok_adv, tok_gates, extra_m = compute_sp_tpma_advantages(
                    rollout_codes, gt_code_sets, item_rewards, w_item=cfg.w_item
                )
            else:
                raise ValueError(f"Unknown RL algo: {cfg.algo}")

            flat_adv = tok_adv.view(bsz * g_size, trie.num_levels)
            flat_gates = tok_gates.view(bsz * g_size, trie.num_levels)

            # Cache `old_lp` and `ref_lp` ONCE before the inner PPO loop so step >= 2 has genuine `ratio != 1.0`!
            with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_bf16):
                old_lp = compute_per_token_logps(raw_pol, rep_p_ids, rep_p_mask, comp_ids, comp_mask).detach()
                ref_lp = compute_per_token_logps(ref_model, rep_p_ids, rep_p_mask, comp_ids, comp_mask).detach()

            anchor_pack = (
                collate_sft_or_sdft_batch(
                    sft_anchor_rows, tokenizer, max_prompt_tokens=cfg.max_prompt_tokens, include_teacher=False
                )
                if (cfg.sft_anchor_weight > 0 and sft_anchor_rows)
                else None
            )

            pg_m: Dict[str, float] = {}
            for _ppo_k in range(max(1, cfg.ppo_epochs)):
                model.train()
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_bf16):
                    cur_lp = compute_per_token_logps(model, rep_p_ids, rep_p_mask, comp_ids, comp_mask)
                    rl_loss, pg_m = ppo_clipped_kl_surrogate_loss(
                        cur_per_token_logps=cur_lp,
                        old_per_token_logps=old_lp,
                        ref_per_token_logps=ref_lp,
                        token_advantages=flat_adv,
                        token_gates=flat_gates,
                        completion_mask=comp_mask,
                        kl_beta=cfg.kl_beta,
                        use_ppo_clip=cfg.use_ppo_clip,
                    )
                    loss = cfg.rl_loss_weight * rl_loss
                    if anchor_pack is not None:
                        anc_out = model(
                            input_ids=anchor_pack["input_ids"].to(device),
                            attention_mask=anchor_pack["attention_mask"].to(device),
                            labels=anchor_pack["labels"].to(device),
                            use_cache=False,
                        )
                        loss = loss + cfg.sft_anchor_weight * anc_out.loss

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()

            merged_step_m = {**rew_m, **pg_m, **extra_m}
            step_logs.append(merged_step_m)
            if rank == 0 and step_idx % 25 == 0:
                print(
                    f"[RL:{cfg.algo}] step={step_idx}/{len(loader)} "
                    f"R_mean={rew_m['reward/total_mean']:.3f} pos_hit={rew_m['reward/pos_hit_rate']:.3f} "
                    f"kl={pg_m.get('rl/kl_div', 0.0):.4f} clip_frac={pg_m.get('rl/clip_frac', 0.0):.3f}",
                    flush=True,
                )
            if step_idx % eval_interval == 0:
                _maybe_save_best_on_val(f"ep{ep}_step{step_idx}")

        if is_ddp:
            dist.barrier()

    # Final validation check & restore best val checkpoint onto `model` before test evaluation!
    _maybe_save_best_on_val("final")
    if ckpt_dir.exists():
        best_loaded = AutoModelForCausalLM.from_pretrained(
            ckpt_dir, torch_dtype=dtype_load, trust_remote_code=True
        ).to(device)
        raw_target = model.module if isinstance(model, DDP) else model
        raw_target.load_state_dict(best_loaded.state_dict())
        del best_loaded
    elif rank == 0:
        raw_save = model.module if isinstance(model, DDP) else model
        raw_save.config.use_cache = True
        raw_save.save_pretrained(ckpt_dir)
        tokenizer.save_pretrained(ckpt_dir)
    if is_ddp:
        dist.barrier()

    # Final Test Evaluation via Constrained Beam Search (+ 4-Tier Relevance Judge scoring!)
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
        user_id_mode=user_id_mode,
        personalized=True,
        num_beams=cfg.eval_beams,
        batch_size=16,
        max_prompt_tokens=cfg.max_prompt_tokens,
        rank=rank,
        world_size=world_size,
    )

    summary: Dict[str, Any] = {}
    if rank == 0:
        if rel_scorer is not None:
            from minionesearch.baselines.run_phase1_eval import score_top10_relevance_for_methods
            from minionesearch.evaluation import evaluate_predictions

            item_text_map = {int(r["item_id"]): str(r["structured_text"]) for r in corpus_recs}
            item_to_cat = {int(r["item_id"]): int(r.get("category_level2_id") or 0) for r in corpus_recs}
            valid_c = {int(r["item_id"]) for r in corpus_recs}
            rel_map = score_top10_relevance_for_methods(
                rel_scorer, test_sessions, {cfg.algo: test_preds}, item_text_map, batch_size=512
            )
            ks_eval = tuple(k for k in (1, 5, 10, 20, 50) if k <= cfg.eval_beams) or (1, cfg.eval_beams)
            test_metrics = evaluate_predictions(
                test_sessions, test_preds, ks=ks_eval, item_to_cat=item_to_cat, valid_corpus_ids=valid_c, session_item_rel_scores=rel_map
            )

        clip_vals = [float(x.get("rl/clip_frac", 0.0)) for x in step_logs if "rl/clip_frac" in x]
        summary = {
            "code_version": "v3_clean_orthogonal",
            "config": asdict(cfg),
            "inherited_prompt_config": {
                "user_id_mode": user_id_mode,
                "use_query_sid": bool(use_query_sid),
            },
            "reward_source": "neural_4tier_rm" if rel_scorer is not None else "heuristic_lexical_fallback",
            "actual_gt_injection_rate": round(total_injected_prompts / max(1, total_rollout_prompts), 6),
            "mean_ppo_clip_frac": round(sum(clip_vals) / max(1, len(clip_vals)), 6) if clip_vals else 0.0,
            "best_val_ckpt_tag": best_ckpt_tag,
            "val_history": val_history,
            "test_metrics": test_metrics,
            "latency": test_lat,
            "train_tail_metrics": step_logs[-5:] if step_logs else [],
        }
        (opath / "rl_eval_summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        (opath / "test_predictions.json").write_text(
            json.dumps(test_preds, ensure_ascii=False), encoding="utf-8"
        )
        print(f"[RL:{cfg.algo} Done] Test Overall: {test_metrics.get('overall', {})}", flush=True)

    if is_ddp:
        dist.barrier()
        dist.destroy_process_group()
    return summary


def main() -> None:
    p = argparse.ArgumentParser(description="Run Search-RL Post-Training Suite")
    p.add_argument("--bench_dir", required=True)
    p.add_argument("--sid_dir", required=True)
    p.add_argument("--sft_ckpt_dir", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--rel_rm_ckpt", default="")
    p.add_argument("--algo", default="sp_tpma_grpo", choices=["listwise_dpo", "vanilla_grpo", "tpma_grpo", "sp_tpma_grpo"])
    p.add_argument("--num_generations", type=int, default=12)
    p.add_argument("--rollout_mode", default="hybrid", choices=["beam", "hybrid"])
    p.add_argument("--max_rl_sessions", type=int, default=12000)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--ppo_epochs", type=int, default=2)
    p.add_argument("--use_ppo_clip", type=int, default=1)
    p.add_argument("--inject_gt_anchor", type=int, default=1)
    p.add_argument("--rl_loss_weight", type=float, default=1.0)
    p.add_argument("--w_item", type=float, default=0.50)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1.5e-5)
    p.add_argument("--kl_beta", type=float, default=0.02)
    p.add_argument("--sft_anchor_weight", type=float, default=0.08)
    p.add_argument("--eval_beams", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()

    cfg = RLTrainConfig(
        algo=a.algo,
        num_generations=a.num_generations,
        rollout_mode=a.rollout_mode,
        max_rl_sessions=a.max_rl_sessions,
        epochs=a.epochs,
        ppo_epochs=a.ppo_epochs,
        use_ppo_clip=bool(a.use_ppo_clip),
        inject_gt_anchor=bool(a.inject_gt_anchor),
        rl_loss_weight=a.rl_loss_weight,
        w_item=a.w_item,
        batch_size=a.batch_size,
        lr=a.lr,
        kl_beta=a.kl_beta,
        sft_anchor_weight=a.sft_anchor_weight,
        eval_beams=a.eval_beams,
        seed=a.seed,
    )
    train_generative_search_rl(
        bench_dir=a.bench_dir,
        sid_dir=a.sid_dir,
        sft_ckpt_dir=a.sft_ckpt_dir,
        out_dir=a.out_dir,
        cfg=cfg,
        rel_rm_ckpt=a.rel_rm_ckpt or None,
    )


if __name__ == "__main__":
    main()
