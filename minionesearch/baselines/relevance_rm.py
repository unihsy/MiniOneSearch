"""4-Tier Search Relevance Cross-Encoder (Judge & RL Reward Model).

Trained on KuaiSearch's 46,422 human-annotated 4-level relevance records (`score` in {0,1,2,3}):
  - 3: Excellent (Exact match on entity & key attributes)
  - 2: Related (Category/intent match, minor attribute difference)
  - 1: Mismatch (Weakly related or critical attribute mismatch)
  - 0: Irrelevant

Trains with calibrated hybrid classification + ordinal expected-value loss:
  L = CE(logits, y) + ordinal_weight * MSE(sum_c c * P(y=c|x), y)

Supports:
  1. Multi-GPU DDP training (`torchrun --nproc_per_node=N -m minionesearch.baselines.relevance_rm ...`)
  2. Batched/sharded GPU scoring for:
       - Two-stage `Dense Top-100 -> Cross-Encoder Rerank` baseline
       - Model-based `Rel@K` / `StrictRelRate@K` evaluation on generated SIDs
       - Online/offline `R_Rel` reward computation in Search-RL
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import spearmanr
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from tqdm import tqdm

from minionesearch.utils import iter_jsonl, set_global_seed


def format_relevance_doc_text(rec: Mapping[str, Any]) -> str:
    """Format item text consistently between `relevance` table and `corpus.jsonl`."""
    if "structured_text" in rec and rec["structured_text"]:
        return str(rec["structured_text"])
    title = str(rec.get("item_title") or rec.get("title") or "").strip()
    brand = str(rec.get("brand_name") or rec.get("brand") or "").strip()
    attr = str(rec.get("attr_value") or rec.get("category_level2_name") or "").strip()
    parts = [title] if title else []
    if brand and brand not in {"无品牌", "其他/other", "UNKNOWN"}:
        parts.append(f"品牌:{brand}")
    if attr and attr not in {"UNKNOWN"}:
        parts.append(f"属性:{attr}")
    return " | ".join(parts)


class RelevancePairDataset(Dataset):
    def __init__(self, records: Sequence[Mapping[str, Any]]):
        self.records = list(records)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Tuple[str, str, int]:
        r = self.records[idx]
        q = str(r["query"])
        doc = format_relevance_doc_text(r)
        label = int(r["score"])
        return q, doc, label


def expected_relevance_from_logits(logits: torch.Tensor) -> torch.Tensor:
    """Convert `[B, 4]` logits into calibrated continuous relevance score in `[0.0, 3.0]`."""
    probs = F.softmax(logits.float(), dim=-1)
    levels = torch.arange(logits.size(-1), device=logits.device, dtype=torch.float32)
    return (probs * levels).sum(dim=-1)


def compute_relevance_eval_metrics(
    gold_labels: Sequence[int],
    pred_classes: Sequence[int],
    pred_expected: Sequence[float],
) -> Dict[str, float]:
    y_true = np.asarray(gold_labels, dtype=np.int64)
    y_cls = np.asarray(pred_classes, dtype=np.int64)
    y_exp = np.asarray(pred_expected, dtype=np.float64)

    acc = float(accuracy_score(y_true, y_cls))
    macro_f1 = float(f1_score(y_true, y_cls, average="macro", zero_division=0))
    mae = float(np.mean(np.abs(y_exp - y_true)))
    rho_res = spearmanr(y_true, y_exp)
    rho = float(rho_res.statistic) if not np.isnan(rho_res.statistic) else 0.0

    y_strict = (y_true >= 2).astype(np.int64)
    if len(np.unique(y_strict)) > 1:
        strict_auc = float(roc_auc_score(y_strict, y_exp))
    else:
        strict_auc = 0.0

    return {
        "accuracy": round(acc, 4),
        "macro_f1": round(macro_f1, 4),
        "mae": round(mae, 4),
        "spearman_rho": round(rho, 4),
        "strict_rel_auc": round(strict_auc, 4),
        "num_samples": int(len(y_true)),
    }


@torch.no_grad()
def evaluate_relevance_model(
    model: nn.Module,
    tokenizer: Any,
    records: Sequence[Mapping[str, Any]],
    device: torch.device,
    batch_size: int = 128,
    max_length: int = 192,
) -> Dict[str, float]:
    model.eval()
    raw_model = model.module if isinstance(model, DDP) else model
    gold_all: List[int] = []
    cls_all: List[int] = []
    exp_all: List[float] = []
    use_bf16 = device.type == "cuda" and torch.cuda.is_bf16_supported()

    for start in range(0, len(records), batch_size):
        batch = records[start : start + batch_size]
        queries = [str(r["query"]) for r in batch]
        docs = [format_relevance_doc_text(r) for r in batch]
        labels = [int(r["score"]) for r in batch]

        enc = tokenizer(
            queries,
            docs,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        ).to(device)

        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_bf16):
            logits = raw_model(**enc).logits

        exp_scores = expected_relevance_from_logits(logits)
        pred_cls = logits.argmax(dim=-1)

        gold_all.extend(labels)
        cls_all.extend(pred_cls.cpu().tolist())
        exp_all.extend(exp_scores.cpu().tolist())

    return compute_relevance_eval_metrics(gold_all, cls_all, exp_all)


def train_relevance_rm(
    rel_dir: str | Path,
    out_dir: str | Path,
    base_model: str = "BAAI/bge-reranker-v2-m3",
    epochs: int = 3,
    batch_size: int = 64,
    lr: float = 3e-5,
    max_length: int = 192,
    ordinal_weight: float = 0.25,
    seed: int = 42,
) -> Dict[str, Any]:
    """Train 4-tier Relevance Cross-Encoder (single-GPU or `torchrun` DDP)."""
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_cosine_schedule_with_warmup

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
    rel_path = Path(rel_dir)
    save_path = Path(out_dir)
    if rank == 0:
        save_path.mkdir(parents=True, exist_ok=True)

    train_recs = list(iter_jsonl(rel_path / "train.jsonl"))
    val_recs = list(iter_jsonl(rel_path / "val.jsonl"))
    test_recs = list(iter_jsonl(rel_path / "test.jsonl"))

    # Rank 0 downloads/loads first to populate cache cleanly before barrier
    if rank == 0:
        tok0 = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
        _ = AutoModelForSequenceClassification.from_pretrained(
            base_model, num_labels=4, ignore_mismatched_sizes=True, trust_remote_code=True
        )
    if is_ddp:
        dist.barrier()

    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model_raw = AutoModelForSequenceClassification.from_pretrained(
        base_model, num_labels=4, ignore_mismatched_sizes=True, trust_remote_code=True
    )
    if getattr(model_raw.config, "pad_token_id", None) is None:
        model_raw.config.pad_token_id = tokenizer.pad_token_id
    model: nn.Module = model_raw.to(device)

    if is_ddp:
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)

    ds_train = RelevancePairDataset(train_recs)
    sampler = DistributedSampler(ds_train, num_replicas=world_size, rank=rank, shuffle=True, seed=seed) if is_ddp else None

    def _collate(batch: List[Tuple[str, str, int]]) -> Dict[str, torch.Tensor]:
        qs, ds, ys = zip(*batch)
        enc = tokenizer(
            list(qs),
            list(ds),
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        enc["labels_cls"] = torch.tensor(ys, dtype=torch.long)
        return enc

    loader = DataLoader(
        ds_train,
        batch_size=batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=0,
        pin_memory=False,
        collate_fn=_collate,
    )

    total_steps = max(1, len(loader) * epochs)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=int(0.08 * total_steps), num_training_steps=total_steps
    )
    ce_loss_fn = nn.CrossEntropyLoss(label_smoothing=0.05)
    use_bf16 = device.type == "cuda" and torch.cuda.is_bf16_supported()

    best_rho = -1.0
    history: List[Dict[str, Any]] = []

    for epoch in range(1, epochs + 1):
        if sampler is not None:
            sampler.set_epoch(epoch)
        model.train()
        running_loss = 0.0
        pbar = tqdm(loader, desc=f"[RelRM] Epoch {epoch}/{epochs}", disable=(rank != 0), mininterval=5.0)
        for batch in pbar:
            labels_cls = batch.pop("labels_cls").to(device)
            batch = {k: v.to(device) for k, v in batch.items()}

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_bf16):
                logits = model(**batch).logits
                loss_ce = ce_loss_fn(logits.float(), labels_cls)
                exp_rel = expected_relevance_from_logits(logits)
                loss_ord = F.mse_loss(exp_rel, labels_cls.float())
                loss = loss_ce + ordinal_weight * loss_ord

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            running_loss += float(loss.detach().item())

        if is_ddp:
            dist.barrier()

        if rank == 0:
            val_metrics = evaluate_relevance_model(model, tokenizer, val_recs, device, batch_size=batch_size * 2, max_length=max_length)
            val_metrics["epoch"] = epoch
            val_metrics["train_loss"] = round(running_loss / max(1, len(loader)), 4)
            history.append(val_metrics)
            print(f"[RelRM] Epoch {epoch} Val: {val_metrics}", flush=True)
            if val_metrics["spearman_rho"] >= best_rho:
                best_rho = val_metrics["spearman_rho"]
                raw_m = model.module if isinstance(model, DDP) else model
                raw_m.save_pretrained(save_path / "best_model")
                tokenizer.save_pretrained(save_path / "best_model")

        if is_ddp:
            dist.barrier()

    summary: Dict[str, Any] = {}
    if rank == 0:
        from transformers import AutoModelForSequenceClassification

        best_model = AutoModelForSequenceClassification.from_pretrained(save_path / "best_model").to(device)
        test_metrics = evaluate_relevance_model(best_model, tokenizer, test_recs, device, batch_size=batch_size * 2, max_length=max_length)
        summary = {
            "base_model": base_model,
            "best_val_spearman_rho": best_rho,
            "val_history": history,
            "test_metrics": test_metrics,
        }
        (save_path / "relevance_rm_metrics.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"[RelRM] Held-out Unseen-Query Test Metrics: {test_metrics}", flush=True)

    if is_ddp:
        dist.barrier()
        dist.destroy_process_group()
    return summary


class RelevanceScorer:
    """Fast batched GPU/CPU Cross-Encoder scorer for Reranking, Eval (`Rel@K`), and RL (`R_Rel`)."""

    def __init__(
        self,
        model_path: str | Path,
        device: Optional[str | torch.device] = None,
        max_length: int = 192,
    ):
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.device = torch.device(device) if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path, trust_remote_code=True).to(self.device)
        if getattr(self.model.config, "pad_token_id", None) is None:
            self.model.config.pad_token_id = self.tokenizer.pad_token_id
        self.model.eval()
        self.max_length = max_length
        self.use_bf16 = self.device.type == "cuda" and torch.cuda.is_bf16_supported()

    @torch.no_grad()
    def score_pairs(
        self,
        queries: Sequence[str],
        doc_texts: Sequence[str],
        batch_size: int = 256,
    ) -> np.ndarray:
        """Return calibrated expected relevance scores in `[0.0, 3.0]` for `(query, doc)` pairs."""
        if not queries:
            return np.empty(0, dtype=np.float32)
        out = np.zeros(len(queries), dtype=np.float32)
        for start in range(0, len(queries), batch_size):
            q_b = list(queries[start : start + batch_size])
            d_b = list(doc_texts[start : start + batch_size])
            enc = self.tokenizer(
                q_b,
                d_b,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            ).to(self.device)
            with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16, enabled=self.use_bf16):
                logits = self.model(**enc).logits
            scores = expected_relevance_from_logits(logits).cpu().numpy()
            out[start : start + len(scores)] = scores
        return out

    def rerank_predictions(
        self,
        sessions: Sequence[Mapping[str, Any]],
        initial_preds: Mapping[str, Sequence[int]],
        item_text_map: Mapping[int, str],
        rerank_depth: int = 100,
        batch_size: int = 256,
    ) -> Tuple[Dict[str, List[int]], Dict[str, Dict[int, float]], float]:
        """Rerank top-`rerank_depth` candidates per session and return `(reranked_preds, rel_cache, wall_sec)`."""
        t0 = time.perf_counter()
        flat_q: List[str] = []
        flat_d: List[str] = []
        spans: List[Tuple[str, List[int], List[int], int, int]] = []

        for sess in sessions:
            sid = str(sess["session_id"])
            q = str(sess["query"])
            cands = list(initial_preds.get(sid, []))
            head = cands[:rerank_depth]
            tail = cands[rerank_depth:]
            st = len(flat_q)
            for iid in head:
                flat_q.append(q)
                flat_d.append(item_text_map.get(int(iid), ""))
            ed = len(flat_q)
            spans.append((sid, head, tail, st, ed))

        all_scores = self.score_pairs(flat_q, flat_d, batch_size=batch_size)
        wall_sec = time.perf_counter() - t0

        reranked: Dict[str, List[int]] = {}
        rel_cache: Dict[str, Dict[int, float]] = {}
        for sid, head, tail, st, ed in spans:
            if ed > st:
                sc = all_scores[st:ed]
                order = np.argsort(-sc, kind="stable")
                sorted_head = [int(head[i]) for i in order]
                rel_cache[sid] = {int(head[i]): float(sc[i]) for i in range(len(head))}
            else:
                sorted_head = []
                rel_cache[sid] = {}
            reranked[sid] = sorted_head + [int(x) for x in tail]

        return reranked, rel_cache, wall_sec


def main() -> None:
    p = argparse.ArgumentParser(description="Train & evaluate 4-Tier Relevance Cross-Encoder")
    p.add_argument("--rel_dir", required=True, help="Directory with relevance/{train,val,test}.jsonl")
    p.add_argument("--out_dir", required=True, help="Output directory for checkpoints & metrics")
    p.add_argument("--base_model", default="BAAI/bge-reranker-v2-m3")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-5)
    p.add_argument("--max_length", type=int, default=192)
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()

    train_relevance_rm(
        rel_dir=a.rel_dir,
        out_dir=a.out_dir,
        base_model=a.base_model,
        epochs=a.epochs,
        batch_size=a.batch_size,
        lr=a.lr,
        max_length=a.max_length,
        seed=a.seed,
    )


if __name__ == "__main__":
    main()
