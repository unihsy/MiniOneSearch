"""Dense Bi-Encoder Search Baselines (Zero-Shot, Query-Only FT, and Personalized FT).

Implements industrial-grade Dense Retrieval (`BAAI/bge-m3` by default):
  1. Zero-Shot Dense Retrieval
  2. Fine-Tuned Query-Only Dense Retrieval (cross-GPU in-batch + impressed hard-negative InfoNCE)
  3. Fine-Tuned Personalized Dense Retrieval (`[Query] + [Recent User Queries & Clicked Titles]`)
  4. Exact GPU BF16 MIPS Top-K search with P50/P95/P99 latency and QPS profiling
"""
from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from tqdm import tqdm

from minionesearch.evaluation import summarize_latencies_ms
from minionesearch.utils import iter_jsonl, set_global_seed


def format_dense_query_input(
    sess: Mapping[str, Any],
    item_title_map: Optional[Mapping[int, str]] = None,
    personalized: bool = False,
    max_hist_q: int = 5,
    max_hist_i: int = 5,
) -> str:
    """Format query text (optionally enriched with causal user search & click history)."""
    q = str(sess["query"]).strip()
    if not personalized:
        return f"搜索查询: {q}"

    parts = [f"当前搜索: {q}"]
    hist_q = [str(x).strip() for x in (sess.get("hist_queries") or [])[-max_hist_q:] if str(x).strip()]
    if hist_q:
        parts.append("历史搜索: " + " ; ".join(hist_q))

    if item_title_map is not None:
        hist_i_ids = (sess.get("hist_clicked_item_ids") or [])[-max_hist_i:]
        hist_titles = [item_title_map[int(i)][:28] for i in hist_i_ids if int(i) in item_title_map]
        if hist_titles:
            parts.append("历史点击: " + " ; ".join(hist_titles))

    return " | ".join(parts)


class BiEncoderWrapper(nn.Module):
    """Normalized CLS/mean pooling bi-encoder wrapper compatible with BGE/Qwen/BERT encoders."""

    def __init__(self, encoder: nn.Module, pooling: str = "cls"):
        super().__init__()
        self.encoder = encoder
        self.pooling = pooling

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        hidden = out.last_hidden_state
        if self.pooling == "cls":
            emb = hidden[:, 0]
        else:
            mask = attention_mask.unsqueeze(-1).float()
            emb = (hidden.float() * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-6)
        return F.normalize(emb.float(), p=2, dim=-1)


class ContrastiveSearchDataset(Dataset):
    """Training dataset yielding `(query_text, pos_doc_text, list_of_hard_neg_doc_texts)`."""

    def __init__(
        self,
        train_sessions: Sequence[Mapping[str, Any]],
        corpus_map: Mapping[int, str],
        item_title_map: Mapping[int, str],
        all_corpus_ids: Sequence[int],
        personalized: bool = False,
        num_hard_negs: int = 3,
        seed: int = 42,
    ):
        self.samples: List[Tuple[str, int, List[int]]] = []
        self.corpus_map = corpus_map
        self.all_corpus_ids = list(all_corpus_ids)
        self.num_hard_negs = num_hard_negs
        self.rng = random.Random(seed)

        for sess in train_sessions:
            pos_list = list(dict.fromkeys((sess.get("purchased_item_ids") or []) + (sess.get("clicked_item_ids") or [])))
            pos_set = {int(x) for x in pos_list if int(x) in corpus_map}
            if not pos_set:
                continue
            q_text = format_dense_query_input(sess, item_title_map=item_title_map, personalized=personalized)
            imp_negs = [
                int(x) for x in (sess.get("impressed_item_ids") or [])
                if int(x) in corpus_map and int(x) not in pos_set
            ]
            # Pick primary positive (prefer purchased if available, plus sample across clicks)
            primary_pos = int(pos_list[0])
            if primary_pos in corpus_map:
                self.samples.append((q_text, primary_pos, imp_negs))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[str, str, List[str]]:
        q_text, pos_id, imp_negs = self.samples[idx]
        pos_doc = self.corpus_map[pos_id]

        chosen_negs: List[int] = []
        if len(imp_negs) >= self.num_hard_negs:
            chosen_negs = self.rng.sample(imp_negs, self.num_hard_negs)
        else:
            chosen_negs = list(imp_negs)
            while len(chosen_negs) < self.num_hard_negs:
                cand = self.rng.choice(self.all_corpus_ids)
                if cand != pos_id:
                    chosen_negs.append(cand)

        neg_docs = [self.corpus_map[nid] for nid in chosen_negs]
        return q_text, pos_doc, neg_docs


def _gather_across_gpus_with_grad(x: torch.Tensor) -> torch.Tensor:
    if not (dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1):
        return x
    ws = dist.get_world_size()
    rk = dist.get_rank()
    gathered = [torch.zeros_like(x) for _ in range(ws)]
    dist.all_gather(gathered, x.contiguous())
    gathered[rk] = x  # preserve local autograd graph
    return torch.cat(gathered, dim=0)


@torch.no_grad()
def encode_texts_single_or_sharded(
    model: BiEncoderWrapper,
    tokenizer: Any,
    texts: Sequence[str],
    device: torch.device,
    batch_size: int = 256,
    max_length: int = 128,
    desc: str = "Encoding",
) -> np.ndarray:
    """Encode a list of texts into L2-normalized FP32 embeddings `[N, D]`."""
    model.eval()
    raw_model = model.module if isinstance(model, DDP) else model
    use_bf16 = device.type == "cuda" and torch.cuda.is_bf16_supported()
    chunks: List[np.ndarray] = []

    for start in tqdm(range(0, len(texts), batch_size), desc=desc, mininterval=5.0):
        b_texts = list(texts[start : start + batch_size])
        enc = tokenizer(
            b_texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        ).to(device)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_bf16):
            emb = raw_model(enc["input_ids"], enc["attention_mask"])
        chunks.append(emb.cpu().numpy().astype(np.float32))

    return np.vstack(chunks) if chunks else np.zeros((0, 1), dtype=np.float32)


@torch.no_grad()
def gpu_exact_mips_retrieve(
    model: BiEncoderWrapper,
    tokenizer: Any,
    sessions: Sequence[Mapping[str, Any]],
    corpus_embs: np.ndarray,
    corpus_item_ids: np.ndarray,
    device: torch.device,
    item_title_map: Optional[Mapping[int, str]] = None,
    personalized: bool = False,
    top_k: int = 100,
    query_batch_size: int = 128,
    max_q_length: int = 128,
) -> Tuple[Dict[str, List[int]], Dict[str, float]]:
    """Run end-to-end query encoding + GPU exact MIPS Top-K retrieval with latency profiling."""
    model.eval()
    raw_model = model.module if isinstance(model, DDP) else model
    use_bf16 = device.type == "cuda" and torch.cuda.is_bf16_supported()
    eff_k = min(top_k, len(corpus_item_ids))

    corpus_dtype = torch.bfloat16 if use_bf16 else torch.float32
    corpus_tensor_t = torch.from_numpy(corpus_embs).to(device=device, dtype=corpus_dtype).t().contiguous()

    predictions: Dict[str, List[int]] = {}
    latencies_ms: List[float] = []

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    t_wall0 = time.perf_counter()

    for start in range(0, len(sessions), query_batch_size):
        batch = sessions[start : start + query_batch_size]
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        t_b0 = time.perf_counter()

        q_texts = [
            format_dense_query_input(s, item_title_map=item_title_map, personalized=personalized)
            for s in batch
        ]
        enc = tokenizer(
            q_texts,
            padding=True,
            truncation=True,
            max_length=max_q_length,
            return_tensors="pt",
        ).to(device)

        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_bf16):
            q_emb = raw_model(enc["input_ids"], enc["attention_mask"]).to(corpus_dtype)
            scores = torch.matmul(q_emb, corpus_tensor_t)  # [B, N_corpus]
            _, top_indices = torch.topk(scores, k=eff_k, dim=-1, largest=True, sorted=True)

        top_idx_np = top_indices.cpu().numpy()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        dt_per_q_ms = (time.perf_counter() - t_b0) * 1000.0 / max(1, len(batch))
        latencies_ms.extend([dt_per_q_ms] * len(batch))

        for b_i, sess in enumerate(batch):
            predictions[str(sess["session_id"])] = corpus_item_ids[top_idx_np[b_i]].tolist()

    total_wall = time.perf_counter() - t_wall0
    latency_stats = summarize_latencies_ms(latencies_ms, len(sessions), total_wall)
    return predictions, latency_stats


def train_dense_retriever(
    bench_dir: str | Path,
    out_dir: str | Path,
    base_model: str = "BAAI/bge-m3",
    personalized: bool = False,
    epochs: int = 2,
    batch_size: int = 64,
    lr: float = 2e-5,
    temperature: float = 0.03,
    num_hard_negs: int = 3,
    max_q_len: int = 128,
    max_d_len: int = 96,
    seed: int = 42,
) -> Path:
    """Fine-tune Bi-Encoder on `train_sessions.jsonl` using DDP InfoNCE + impressed hard negatives."""
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
    item_title_map = {int(r["item_id"]): str(r["item_title"]) for r in corpus_recs}
    all_corpus_ids = [int(r["item_id"]) for r in corpus_recs]
    train_sessions = list(iter_jsonl(bpath / "train_sessions.jsonl"))

    if rank == 0:
        _ = AutoTokenizer.from_pretrained(base_model)
        _ = AutoModel.from_pretrained(base_model)
    if is_ddp:
        dist.barrier()

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    backbone = AutoModel.from_pretrained(base_model)
    # Freeze unused pooler head (if present, e.g. XLM-RoBERTa / BERT) so DDP reduction is 100% clean
    if hasattr(backbone, "pooler") and backbone.pooler is not None:
        for p in backbone.pooler.parameters():
            p.requires_grad = False
    model: nn.Module = BiEncoderWrapper(backbone, pooling="cls").to(device)
    if is_ddp:
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)

    ds = ContrastiveSearchDataset(
        train_sessions=train_sessions,
        corpus_map=corpus_map,
        item_title_map=item_title_map,
        all_corpus_ids=all_corpus_ids,
        personalized=personalized,
        num_hard_negs=num_hard_negs,
        seed=seed + rank,
    )
    sampler = DistributedSampler(ds, num_replicas=world_size, rank=rank, shuffle=True, seed=seed) if is_ddp else None

    def _collate(batch: List[Tuple[str, str, List[str]]]) -> Dict[str, torch.Tensor]:
        qs, pdocs, ndocs_nested = zip(*batch)
        flat_docs: List[str] = []
        for p_d, n_list in zip(pdocs, ndocs_nested):
            flat_docs.append(p_d)
            flat_docs.extend(n_list)
        q_enc = tokenizer(list(qs), padding=True, truncation=True, max_length=max_q_len, return_tensors="pt")
        d_enc = tokenizer(flat_docs, padding=True, truncation=True, max_length=max_d_len, return_tensors="pt")
        return {
            "q_input_ids": q_enc["input_ids"],
            "q_attention_mask": q_enc["attention_mask"],
            "d_input_ids": d_enc["input_ids"],
            "d_attention_mask": d_enc["attention_mask"],
        }

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
    group_docs_per_q = 1 + num_hard_negs

    for epoch in range(1, epochs + 1):
        if sampler is not None:
            sampler.set_epoch(epoch)
        model.train()
        running_loss = 0.0
        pbar = tqdm(
            loader,
            desc=f"[DenseFT pers={personalized}] Epoch {epoch}/{epochs}",
            disable=(rank != 0),
            mininterval=5.0,
        )
        for batch in pbar:
            q_ids = batch["q_input_ids"].to(device)
            q_mask = batch["q_attention_mask"].to(device)
            d_ids = batch["d_input_ids"].to(device)
            d_mask = batch["d_attention_mask"].to(device)

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_bf16):
                q_local = model(q_ids, q_mask)   # [B, D]
                d_local = model(d_ids, d_mask)   # [B * (1 + K), D]

                all_q = _gather_across_gpus_with_grad(q_local)
                all_d = _gather_across_gpus_with_grad(d_local)

                logits = torch.matmul(all_q, all_d.t()) / temperature  # [W*B, W*B*(1+K)]
                targets = torch.arange(all_q.size(0), device=device) * group_docs_per_q
                loss = F.cross_entropy(logits, targets)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            running_loss += float(loss.detach().item())

        if rank == 0:
            print(
                f"[DenseFT pers={personalized}] Epoch {epoch} avg_loss={running_loss / max(1, len(loader)):.4f}",
                flush=True,
            )
        if is_ddp:
            dist.barrier()

    ckpt_dir = spath / "encoder_ckpt"
    if rank == 0:
        raw_wrapper: BiEncoderWrapper = model.module if isinstance(model, DDP) else model  # type: ignore[assignment]
        raw_wrapper.encoder.save_pretrained(ckpt_dir)
        tokenizer.save_pretrained(ckpt_dir)
        (spath / "train_meta.json").write_text(
            json.dumps({"base_model": base_model, "personalized": personalized, "epochs": epochs}, indent=2),
            encoding="utf-8",
        )
    if is_ddp:
        dist.barrier()
        dist.destroy_process_group()
    return ckpt_dir


def main() -> None:
    p = argparse.ArgumentParser(description="Train Dense Bi-Encoder Search Baseline")
    p.add_argument("--bench_dir", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--base_model", default="BAAI/bge-m3")
    p.add_argument("--personalized", type=int, default=0, choices=[0, 1])
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--num_hard_negs", type=int, default=3)
    p.add_argument("--max_q_len", type=int, default=128)
    p.add_argument("--max_d_len", type=int, default=96)
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()

    train_dense_retriever(
        bench_dir=a.bench_dir,
        out_dir=a.out_dir,
        base_model=a.base_model,
        personalized=bool(a.personalized),
        epochs=a.epochs,
        batch_size=a.batch_size,
        lr=a.lr,
        num_hard_negs=a.num_hard_negs,
        max_q_len=a.max_q_len,
        max_d_len=a.max_d_len,
        seed=a.seed,
    )


if __name__ == "__main__":
    main()
