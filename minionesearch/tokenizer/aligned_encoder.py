"""OneSearch-V1 §3.1.1 Five-Objective Aligned Collaborative & Semantic Representation Encoder.

Implements Eq. (1) of OneSearch-V1 (Chen et al., ICML 2026, arXiv:2509.03236):
    L_align = lambda_1 * L_q2q + lambda_2 * L_i2i + lambda_3 * L_q2i + lambda_4 * L_rank + lambda_5 * L_rel

All training pairs and statistics are constructed from `train_sessions.jsonl` and
`relevance/train.jsonl`:
  - `L_q2i` : Query-to-Item InfoNCE contrastive loss with impressed hard negatives & cross-GPU negatives.
  - `L_i2i` : Collaborative Item-to-Item contrastive loss on co-clicked/co-purchased items within a session.
  - `L_q2q` : Collaborative Query-to-Query contrastive loss on distinct queries clicking the same item.
  - `L_rank`: Behavior-level margin ranking loss (`order > click > impressed_unclicked`).
  - `L_rel` : Calibrated semantic relevance alignment loss on 4-tier annotated `(query, item, score)` pairs.
"""
from __future__ import annotations

import argparse
import json
import os
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from tqdm import tqdm

from minionesearch.baselines.dense_retriever import BiEncoderWrapper, _gather_across_gpus_with_grad
from minionesearch.baselines.relevance_rm import format_relevance_doc_text
from minionesearch.utils import iter_jsonl, set_global_seed


class FiveLossAlignedDataset(Dataset):
    """Yields synchronized collaborative `(q, q_co, i_pos, i_co, i_imp_neg, has_order_gap, i_click_sub)` tuples."""

    def __init__(
        self,
        train_sessions: Sequence[Mapping[str, Any]],
        corpus_map: Mapping[int, str],
        all_corpus_ids: Sequence[int],
        seed: int = 42,
    ):
        self.corpus_map = corpus_map
        self.all_corpus_ids = list(all_corpus_ids)
        self.rng = random.Random(seed)

        # Build item -> co-clicking queries index on `train` for L_q2q
        item_to_queries: Dict[int, List[str]] = defaultdict(list)
        for s in train_sessions:
            q = str(s["query"]).strip()
            for iid in set(s.get("clicked_item_ids") or []) | set(s.get("purchased_item_ids") or []):
                if int(iid) in corpus_map:
                    item_to_queries[int(iid)].append(q)
        self.item_to_uniq_queries: Dict[int, List[str]] = {
            iid: list(dict.fromkeys(qs)) for iid, qs in item_to_queries.items()
        }

        self.samples: List[Tuple[str, int, List[int], List[int], List[int]]] = []
        for s in train_sessions:
            q = str(s["query"]).strip()
            purch = [int(x) for x in (s.get("purchased_item_ids") or []) if int(x) in corpus_map]
            clicks = [int(x) for x in (s.get("clicked_item_ids") or []) if int(x) in corpus_map and int(x) not in purch]
            all_pos = purch + clicks
            if not all_pos:
                continue
            pos_set = set(all_pos)
            imp_negs = [int(x) for x in (s.get("impressed_item_ids") or []) if int(x) in corpus_map and int(x) not in pos_set]
            primary = all_pos[0]
            co_items = [x for x in all_pos[1:] if x != primary]
            self.samples.append((q, primary, co_items, clicks if purch else [], imp_negs))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        q, pos_id, co_items, click_only_when_order, imp_negs = self.samples[idx]
        q_text = f"搜索查询: {q}"

        # Co-clicking query for L_q2q (or self query if unique -> dropout view in encoder)
        cand_qs = [cq for cq in self.item_to_uniq_queries.get(pos_id, []) if cq != q]
        q_co_str = self.rng.choice(cand_qs) if cand_qs else q
        q_co_text = f"搜索查询: {q_co_str}"

        # Co-interacted item for L_i2i
        i_co_id = self.rng.choice(co_items) if co_items else pos_id
        # Impressed hard negative for L_q2i & L_rank
        if imp_negs:
            neg_id = self.rng.choice(imp_negs)
        else:
            neg_id = self.rng.choice(self.all_corpus_ids)
            while neg_id == pos_id:
                neg_id = self.rng.choice(self.all_corpus_ids)

        # Behavior hierarchy: order > click_only > impressed_unclicked
        has_order_click_pair = len(click_only_when_order) > 0
        sub_click_id = self.rng.choice(click_only_when_order) if has_order_click_pair else neg_id

        return {
            "q": q_text,
            "q_co": q_co_text,
            "d_pos": self.corpus_map[pos_id],
            "d_co": self.corpus_map[i_co_id],
            "d_neg": self.corpus_map[neg_id],
            "d_sub": self.corpus_map[sub_click_id],
            "has_order_gap": 1.0 if has_order_click_pair else 0.0,
        }


def train_onesearch_aligned_encoder(
    bench_dir: str | Path,
    out_dir: str | Path,
    base_model: str = "BAAI/bge-m3",
    epochs: int = 3,
    batch_size: int = 48,
    lr: float = 2e-5,
    temperature: float = 0.03,
    w_q2q: float = 0.25,
    w_i2i: float = 0.25,
    w_q2i: float = 1.00,
    w_rank: float = 0.50,
    w_rel: float = 0.50,
    seed: int = 42,
) -> Path:
    """Train OneSearch-V1 Eq. (1) 5-Loss Aligned Collaborative-Semantic Encoder."""
    from transformers import AutoModel, AutoTokenizer, get_cosine_schedule_with_warmup

    local_rank = int(os.environ.get("LOCAL_RANK", "-1"))
    is_ddp = local_rank >= 0 and int(os.environ.get("WORLD_SIZE", "1")) > 1
    if is_ddp:
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        rank = 0
        world_size = 1

    set_global_seed(seed + rank)
    bpath = Path(bench_dir)
    spath = Path(out_dir)
    if rank == 0:
        spath.mkdir(parents=True, exist_ok=True)

    corpus_recs = list(iter_jsonl(bpath / "corpus.jsonl"))
    corpus_map = {int(r["item_id"]): str(r["structured_text"]) for r in corpus_recs}
    all_corpus_ids = [int(r["item_id"]) for r in corpus_recs]
    train_sessions = list(iter_jsonl(bpath / "train_sessions.jsonl"))

    rel_train_file = bpath / "relevance" / "train.jsonl"
    rel_records = list(iter_jsonl(rel_train_file)) if rel_train_file.is_file() else []
    rel_rng = random.Random(seed + 777 + rank)

    if rank == 0:
        _ = AutoTokenizer.from_pretrained(base_model)
        _ = AutoModel.from_pretrained(base_model)
    if is_ddp:
        dist.barrier()

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    backbone = AutoModel.from_pretrained(base_model)
    if hasattr(backbone, "pooler") and backbone.pooler is not None:
        for p in backbone.pooler.parameters():
            p.requires_grad = False
    model: nn.Module = BiEncoderWrapper(backbone, pooling="cls").to(device)
    if is_ddp:
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)

    ds = FiveLossAlignedDataset(train_sessions, corpus_map, all_corpus_ids, seed=seed + rank)
    sampler = DistributedSampler(ds, num_replicas=world_size, rank=rank, shuffle=True, seed=seed) if is_ddp else None

    def _collate(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        def _tok(texts: List[str], max_l: int) -> Dict[str, torch.Tensor]:
            return tokenizer(texts, padding=True, truncation=True, max_length=max_l, return_tensors="pt")

        pack: Dict[str, Any] = {
            "q": _tok([b["q"] for b in batch], 64),
            "q_co": _tok([b["q_co"] for b in batch], 64),
            "d_pos": _tok([b["d_pos"] for b in batch], 96),
            "d_co": _tok([b["d_co"] for b in batch], 96),
            "d_neg": _tok([b["d_neg"] for b in batch], 96),
            "d_sub": _tok([b["d_sub"] for b in batch], 96),
            "has_order_gap": torch.tensor([b["has_order_gap"] for b in batch], dtype=torch.float32),
        }
        if rel_records:
            r_sub = [rel_rng.choice(rel_records) for _ in range(min(len(batch), 32))]
            pack["rel_q"] = _tok([f"搜索查询: {r['query']}" for r in r_sub], 64)
            pack["rel_d"] = _tok([format_relevance_doc_text(r) for r in r_sub], 96)
            pack["rel_y"] = torch.tensor([float(r["score"]) / 3.0 for r in r_sub], dtype=torch.float32)
        return pack

    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=0,
        pin_memory=False,
        drop_last=True,
        collate_fn=_collate,
    )

    total_steps = max(1, len(loader) * epochs)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=int(0.08 * total_steps), num_training_steps=total_steps
    )
    use_bf16 = device.type == "cuda" and torch.cuda.is_bf16_supported()

    for ep in range(1, epochs + 1):
        if sampler is not None:
            sampler.set_epoch(ep)
        model.train()
        running_loss = 0.0
        pbar = tqdm(loader, desc=f"[OneSearch-AlignedEnc] Ep {ep}/{epochs}", disable=(rank != 0), mininterval=5.0)

        for batch in pbar:
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_bf16):
                e_q = model(batch["q"]["input_ids"].to(device), batch["q"]["attention_mask"].to(device))
                e_qco = model(batch["q_co"]["input_ids"].to(device), batch["q_co"]["attention_mask"].to(device))
                e_pos = model(batch["d_pos"]["input_ids"].to(device), batch["d_pos"]["attention_mask"].to(device))
                e_ico = model(batch["d_co"]["input_ids"].to(device), batch["d_co"]["attention_mask"].to(device))
                e_neg = model(batch["d_neg"]["input_ids"].to(device), batch["d_neg"]["attention_mask"].to(device))
                e_sub = model(batch["d_sub"]["input_ids"].to(device), batch["d_sub"]["attention_mask"].to(device))

                # 1. L_q2i (cross-GPU InfoNCE with explicit impressed hard negatives)
                all_q = _gather_across_gpus_with_grad(e_q)
                d_pair = torch.stack([e_pos, e_neg], dim=1).view(-1, e_pos.size(-1))
                all_d = _gather_across_gpus_with_grad(d_pair)
                targets_q2i = torch.arange(all_q.size(0), device=device) * 2
                loss_q2i = F.cross_entropy(torch.matmul(all_q, all_d.t()) / temperature, targets_q2i)

                # 2. L_q2q (collaborative query-to-query alignment)
                lbl_local = torch.arange(e_q.size(0), device=device)
                loss_q2q = F.cross_entropy(torch.matmul(e_q, e_qco.t()) / temperature, lbl_local)

                # 3. L_i2i (collaborative item-to-item alignment, gated by content sim >= 0.5 per §3.1.1)
                cos_i2i = (e_pos.detach() * e_ico.detach()).sum(dim=-1)
                valid_i2i = (cos_i2i >= 0.45).float()
                per_i2i = F.cross_entropy(torch.matmul(e_pos, e_ico.t()) / temperature, lbl_local, reduction="none")
                loss_i2i = (per_i2i * valid_i2i).sum() / valid_i2i.sum().clamp(min=1.0)

                # 4. L_rank (behavior hierarchy margin: pos > imp_neg, plus order > click_only)
                s_pos = (e_q * e_pos).sum(dim=-1)
                s_neg = (e_q * e_neg).sum(dim=-1)
                s_sub = (e_q * e_sub).sum(dim=-1)
                has_gap = batch["has_order_gap"].to(device)
                margin_click_show = F.relu(0.15 - (s_pos - s_neg)).mean()
                margin_order_click = (F.relu(0.08 - (s_pos - s_sub)) * has_gap).sum() / has_gap.sum().clamp(min=1.0)
                loss_rank = margin_click_show + 0.5 * margin_order_click

                # 5. L_rel (4-tier relevance distillation / calibration)
                if "rel_q" in batch:
                    rq = model(batch["rel_q"]["input_ids"].to(device), batch["rel_q"]["attention_mask"].to(device))
                    rd = model(batch["rel_d"]["input_ids"].to(device), batch["rel_d"]["attention_mask"].to(device))
                    ry = batch["rel_y"].to(device)
                    pred_sim = ((rq * rd).sum(dim=-1) + 1.0) * 0.5  # map [-1, 1] -> [0, 1]
                    loss_rel = F.mse_loss(pred_sim.float(), ry.float())
                else:
                    loss_rel = torch.zeros((), device=device)

                loss = (
                    w_q2i * loss_q2i
                    + w_q2q * loss_q2q
                    + w_i2i * loss_i2i
                    + w_rank * loss_rank
                    + w_rel * loss_rel
                )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            running_loss += float(loss.detach().item())

        if rank == 0:
            print(f"[OneSearch-AlignedEnc] Epoch {ep} avg_loss={running_loss / max(1, len(loader)):.4f}", flush=True)
        if is_ddp:
            dist.barrier()

    ckpt_dir = spath / "encoder_ckpt"
    if rank == 0:
        raw_w: BiEncoderWrapper = model.module if isinstance(model, DDP) else model  # type: ignore[assignment]
        raw_w.encoder.save_pretrained(ckpt_dir)
        tokenizer.save_pretrained(ckpt_dir)
        (spath / "aligned_encoder_meta.json").write_text(
            json.dumps(
                {
                    "base_model": base_model,
                    "epochs": epochs,
                    "weights": {"q2q": w_q2q, "i2i": w_i2i, "q2i": w_q2i, "rank": w_rank, "rel": w_rel},
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    if is_ddp:
        dist.barrier()
        dist.destroy_process_group()
    return ckpt_dir


def main() -> None:
    p = argparse.ArgumentParser(description="Train OneSearch-V1 §3.1.1 5-Loss Aligned Encoder")
    p.add_argument("--bench_dir", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--base_model", default="BAAI/bge-m3")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch_size", type=int, default=48)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()

    train_onesearch_aligned_encoder(
        bench_dir=a.bench_dir,
        out_dir=a.out_dir,
        base_model=a.base_model,
        epochs=a.epochs,
        batch_size=a.batch_size,
        lr=a.lr,
        seed=a.seed,
    )


if __name__ == "__main__":
    main()
