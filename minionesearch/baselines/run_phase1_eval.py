"""Unified Phase 1 Evaluation Suite for Traditional Search Baselines.

Evaluates and compares on identical `test_sessions.jsonl` (and `val_sessions.jsonl`):
  1. `BM25` (Hybrid Word + Char-Bigram Sparse Retrieval)
  2. `BGE-M3 (Zero-Shot)`
  3. `FT-BGE-M3 (Query-Only)`
  4. `FT-BGE-M3-Pers (Query + User History)`
  5. `FT-BGE-M3-Pers + Cross-Encoder Top-100 Rerank`

Also uses the trained 4-Tier Relevance Cross-Encoder (`RelevanceScorer`) to compute
model-based `Rel@1/5/10` and `StrictRelRate@1/5/10` across all methods, and exports
publication-ready JSON + Markdown tables (including 4 diagnostic slices & P50/P99 latency).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch

from minionesearch.baselines.bm25 import BM25Retriever
from minionesearch.baselines.dense_retriever import (
    BiEncoderWrapper,
    encode_texts_single_or_sharded,
    gpu_exact_mips_retrieve,
)
from minionesearch.baselines.relevance_rm import RelevanceScorer
from minionesearch.evaluation import evaluate_predictions
from minionesearch.utils import iter_jsonl


def score_top10_relevance_for_methods(
    rel_scorer: RelevanceScorer,
    sessions: Sequence[Mapping[str, Any]],
    method_preds: Mapping[str, Mapping[str, Sequence[int]]],
    item_text_map: Mapping[int, str],
    batch_size: int = 256,
) -> Dict[str, Dict[int, float]]:
    """Batch-score unique `(session_id, item_id)` pairs appearing in any method's Top-10."""
    unique_pairs: List[Tuple[str, int, str, str]] = []
    seen_keys = set()
    for sess in sessions:
        sid = str(sess["session_id"])
        q = str(sess["query"])
        for preds in method_preds.values():
            for iid in preds.get(sid, [])[:10]:
                key = (sid, int(iid))
                if key not in seen_keys and int(iid) in item_text_map:
                    seen_keys.add(key)
                    unique_pairs.append((sid, int(iid), q, item_text_map[int(iid)]))

    qs = [x[2] for x in unique_pairs]
    ds = [x[3] for x in unique_pairs]
    scores = rel_scorer.score_pairs(qs, ds, batch_size=batch_size)

    session_rel_map: Dict[str, Dict[int, float]] = {}
    for (sid, iid, _, _), sc in zip(unique_pairs, scores):
        session_rel_map.setdefault(sid, {})[iid] = float(sc)
    return session_rel_map


def render_markdown_table(results: Mapping[str, Any]) -> str:
    """Render clean publication-ready Markdown tables for overall, slices, and latency."""
    lines: List[str] = ["# Phase 1 Traditional Search Baselines Benchmark\n"]
    lines.append("## 1. Overall Test Performance & Latency\n")
    headers = [
        "Method",
        "Recall@1",
        "Recall@10",
        "Recall@50",
        "NDCG@10",
        "GradedNDCG@10",
        "MRR@10",
        "Rel@10",
        "StrictRel@10",
        "CatDiv@10",
        "P50 (ms)",
        "P99 (ms)",
        "QPS",
    ]
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("|" + "|".join(["---"] * len(headers)) + "|")

    for method_name, mdata in results.items():
        ov = mdata["metrics"].get("overall", {})
        lat = mdata.get("latency", {})
        row = [
            f"**{method_name}**",
            f"{ov.get('Recall@1', 0.0) * 100:.2f}%",
            f"{ov.get('Recall@10', 0.0) * 100:.2f}%",
            f"{ov.get('Recall@50', 0.0) * 100:.2f}%",
            f"{ov.get('NDCG@10', 0.0) * 100:.2f}%",
            f"{ov.get('GradedNDCG@10', 0.0) * 100:.2f}%",
            f"{ov.get('MRR@10', 0.0) * 100:.2f}%",
            f"{ov.get('Rel@10', 0.0):.3f}",
            f"{ov.get('StrictRelRate@10', 0.0) * 100:.2f}%",
            f"{ov.get('CatDiv@10', 0.0) * 100:.2f}%",
            f"{lat.get('p50_ms', 0.0):.2f}",
            f"{lat.get('p99_ms', 0.0):.2f}",
            f"{lat.get('qps', 0.0):.1f}",
        ]
        lines.append("| " + " | ".join(row) + " |")

    lines.append("\n## 2. Diagnostic Slice Breakdown (`NDCG@10` / `Recall@10`)\n")
    slice_names = ["head_exact", "personalized_ambiguous", "long_tail_complex", "cold_start_item"]
    s_headers = ["Method"] + [f"`{s}` (N/R@10)" for s in slice_names]
    lines.append("| " + " | ".join(s_headers) + " |")
    lines.append("|" + "|".join(["---"] * len(s_headers)) + "|")

    for method_name, mdata in results.items():
        s_row = [f"**{method_name}**"]
        for sn in slice_names:
            sm = mdata["metrics"].get(f"slice:{sn}", {})
            s_row.append(f"{sm.get('NDCG@10', 0.0) * 100:.2f}% / {sm.get('Recall@10', 0.0) * 100:.2f}%")
        lines.append("| " + " | ".join(s_row) + " |")

    return "\n".join(lines) + "\n"


def run_all_phase1_evaluations(
    bench_dir: str | Path,
    out_dir: str | Path,
    zeroshot_model: str = "BAAI/bge-m3",
    ft_query_ckpt: Optional[str | Path] = None,
    ft_pers_ckpt: Optional[str | Path] = None,
    rel_rm_ckpt: Optional[str | Path] = None,
    split: str = "test",
    save_zeroshot_corpus_emb: Optional[str | Path] = None,
) -> Dict[str, Any]:
    from transformers import AutoModel, AutoTokenizer

    bpath = Path(bench_dir)
    opath = Path(out_dir)
    opath.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    corpus_recs = list(iter_jsonl(bpath / "corpus.jsonl"))
    eval_sessions = list(iter_jsonl(bpath / f"{split}_sessions.jsonl"))

    corpus_item_ids = np.array([int(r["item_id"]) for r in corpus_recs], dtype=np.int64)
    valid_corpus_set = set(corpus_item_ids.tolist())
    item_to_cat = {int(r["item_id"]): int(r.get("category_level2_id") or 0) for r in corpus_recs}
    item_text_map = {int(r["item_id"]): str(r["structured_text"]) for r in corpus_recs}
    item_title_map = {int(r["item_id"]): str(r["item_title"]) for r in corpus_recs}
    corpus_texts = [str(r["structured_text"]) for r in corpus_recs]

    method_preds: Dict[str, Dict[str, List[int]]] = {}
    method_latencies: Dict[str, Dict[str, float]] = {}

    # 1. BM25
    print(f"[Phase1-Eval] Indexing & evaluating BM25 on {len(corpus_recs):,} items / {len(eval_sessions):,} queries...")
    bm25 = BM25Retriever(k1=1.5, b=0.75, use_char_bigrams=True)
    bm25.index_corpus(corpus_recs)
    bm25_preds, bm25_lat = bm25.retrieve_sessions(eval_sessions, top_k=100, batch_size=128)
    method_preds["BM25"] = bm25_preds
    method_latencies["BM25"] = bm25_lat

    # Helper to evaluate a dense encoder checkpoint
    def _eval_dense(name: str, ckpt: str | Path, personalized: bool, save_emb_path: Optional[Path] = None) -> None:
        print(f"[Phase1-Eval] Encoding corpus & evaluating {name} ({ckpt})...")
        tok = AutoTokenizer.from_pretrained(ckpt)
        enc_model = BiEncoderWrapper(AutoModel.from_pretrained(ckpt), pooling="cls").to(device)
        c_embs = encode_texts_single_or_sharded(
            enc_model, tok, corpus_texts, device=device, batch_size=512, max_length=96, desc=f"Corpus [{name}]"
        )
        if save_emb_path is not None:
            save_emb_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(save_emb_path, c_embs)
        preds, lat = gpu_exact_mips_retrieve(
            enc_model,
            tok,
            eval_sessions,
            corpus_embs=c_embs,
            corpus_item_ids=corpus_item_ids,
            device=device,
            item_title_map=item_title_map,
            personalized=personalized,
            top_k=100,
            query_batch_size=64,
            max_q_length=128 if personalized else 64,
        )
        # Also measure TRUE empirical single-query (`batch_size=1`) latency on up to 100 queries after warmup!
        probe_s = list(eval_sessions[: min(100, len(eval_sessions))])
        if len(probe_s) >= 5:
            _ = gpu_exact_mips_retrieve(
                enc_model, tok, probe_s[:3], corpus_embs=c_embs, corpus_item_ids=corpus_item_ids,
                device=device, item_title_map=item_title_map, personalized=personalized, top_k=100,
                query_batch_size=1, max_q_length=128 if personalized else 64,
            )
            _, online_b1_lat = gpu_exact_mips_retrieve(
                enc_model, tok, probe_s, corpus_embs=c_embs, corpus_item_ids=corpus_item_ids,
                device=device, item_title_map=item_title_map, personalized=personalized, top_k=100,
                query_batch_size=1, max_q_length=128 if personalized else 64,
            )
            lat["online_b1"] = online_b1_lat
        method_preds[name] = preds
        method_latencies[name] = lat
        del enc_model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # 2. Zero-Shot BGE-M3 (also caches corpus embeddings for Phase 2 SID construction if requested!)
    zs_emb_out = Path(save_zeroshot_corpus_emb) if save_zeroshot_corpus_emb else None
    _eval_dense("BGE-M3 (Zero-Shot)", zeroshot_model, personalized=False, save_emb_path=zs_emb_out)

    # 3. Fine-Tuned Query-Only BGE-M3
    if ft_query_ckpt and Path(ft_query_ckpt).exists():
        _eval_dense("FT-BGE-M3 (Query-Only)", ft_query_ckpt, personalized=False)

    # 4. Fine-Tuned Personalized BGE-M3
    if ft_pers_ckpt and Path(ft_pers_ckpt).exists():
        _eval_dense("FT-BGE-M3-Pers (Query+Hist)", ft_pers_ckpt, personalized=True)

    # 5. Two-Stage Cascade Rerank + Model-based Rel@10 scoring
    session_rel_map: Optional[Dict[str, Dict[int, float]]] = None
    if rel_rm_ckpt and Path(rel_rm_ckpt).exists():
        print(f"[Phase1-Eval] Loading 4-Tier Relevance Cross-Encoder from {rel_rm_ckpt}...")
        rel_scorer = RelevanceScorer(rel_rm_ckpt, device=device)

        # Rerank strongest dense retriever Top-100
        base_for_rerank = (
            "FT-BGE-M3-Pers (Query+Hist)"
            if "FT-BGE-M3-Pers (Query+Hist)" in method_preds
            else ("FT-BGE-M3 (Query-Only)" if "FT-BGE-M3 (Query-Only)" in method_preds else "BGE-M3 (Zero-Shot)")
        )
        reranked_preds, rel_cache_100, rerank_wall_sec = rel_scorer.rerank_predictions(
            sessions=eval_sessions,
            initial_preds=method_preds[base_for_rerank],
            item_text_map=item_text_map,
            rerank_depth=100,
            batch_size=512,
        )
        cascade_name = f"{base_for_rerank} + CrossEncoder-Rerank@100"
        method_preds[cascade_name] = reranked_preds
        base_lat = method_latencies[base_for_rerank]
        rerank_ms_per_q = (rerank_wall_sec * 1000.0) / max(1, len(eval_sessions))
        total_ms_per_q = base_lat["mean_ms"] + rerank_ms_per_q

        # Measure TRUE empirical single-query (`1 query x 100 candidates`) Cross-Encoder latency on up to 100 queries!
        import time as _time
        probe_s = list(eval_sessions[: min(100, len(eval_sessions))])
        ce_b1_lats: List[float] = []
        base_b1_p50 = float(base_lat.get("online_b1", base_lat).get("p50_ms", base_lat["p50_ms"]))
        for idx_p, ps in enumerate(probe_s):
            sid_p = str(ps["session_id"])
            cands_p = method_preds[base_for_rerank].get(sid_p, [])[:100]
            qs_p = [str(ps["query"])] * len(cands_p)
            ds_p = [item_text_map.get(int(iid), "") for iid in cands_p]
            if not qs_p:
                continue
            if idx_p < 3:
                _ = rel_scorer.score_pairs(qs_p, ds_p, batch_size=100)
                continue
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            t_ce0 = _time.perf_counter()
            _ = rel_scorer.score_pairs(qs_p, ds_p, batch_size=100)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            ce_b1_lats.append(base_b1_p50 + (_time.perf_counter() - t_ce0) * 1000.0)

        ce_arr = np.asarray(ce_b1_lats, dtype=np.float64) if ce_b1_lats else np.asarray([total_ms_per_q])
        online_b1_cascade = {
            "mean_ms": round(float(np.mean(ce_arr)), 3),
            "p50_ms": round(float(np.percentile(ce_arr, 50)), 3),
            "p95_ms": round(float(np.percentile(ce_arr, 95)), 3),
            "p99_ms": round(float(np.percentile(ce_arr, 99)), 3),
            "qps": round(1000.0 / max(float(np.mean(ce_arr)), 1e-6), 2),
        }
        method_latencies[cascade_name] = {
            "mean_ms": round(total_ms_per_q, 3),
            "p50_ms": online_b1_cascade["p50_ms"],
            "p95_ms": online_b1_cascade["p95_ms"],
            "p99_ms": online_b1_cascade["p99_ms"],
            "qps": round(1000.0 / max(total_ms_per_q, 1e-6), 2),
            "online_b1": online_b1_cascade,
        }

        print("[Phase1-Eval] Scoring Top-10 Relevance across all compared methods...")
        session_rel_map = score_top10_relevance_for_methods(
            rel_scorer=rel_scorer,
            sessions=eval_sessions,
            method_preds=method_preds,
            item_text_map=item_text_map,
            batch_size=512,
        )
        for sid, cmap in rel_cache_100.items():
            session_rel_map.setdefault(sid, {}).update(cmap)

    # Aggregate final metrics & save per-query Top-20 predictions for paired statistical testing!
    bpred_dir = opath / f"baseline_preds_{split}"
    bpred_dir.mkdir(parents=True, exist_ok=True)
    import re as _re
    final_results: Dict[str, Any] = {}
    for m_name, preds in method_preds.items():
        metrics = evaluate_predictions(
            sessions=eval_sessions,
            predictions=preds,
            item_to_cat=item_to_cat,
            valid_corpus_ids=valid_corpus_set,
            session_item_rel_scores=session_rel_map,
        )
        slug = _re.sub(r"[^A-Za-z0-9_.-]+", "_", m_name).strip("_")
        pred_file = bpred_dir / f"{slug}.json"
        compact_preds = {str(k): [int(x) for x in v[:20]] for k, v in preds.items()}
        pred_file.write_text(json.dumps(compact_preds, ensure_ascii=False), encoding="utf-8")
        final_results[m_name] = {
            "metrics": metrics,
            "latency": method_latencies[m_name],
            "predictions_file": f"baseline_preds_{split}/{slug}.json",
        }

    (opath / f"phase1_baselines_{split}.json").write_text(
        json.dumps(final_results, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    md_text = render_markdown_table(final_results)
    (opath / f"phase1_baselines_{split}.md").write_text(md_text, encoding="utf-8")
    print("\n" + md_text, flush=True)
    return final_results


def main() -> None:
    p = argparse.ArgumentParser(description="Run Phase 1 Baseline Evaluation Suite")
    p.add_argument("--bench_dir", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--zeroshot_model", default="BAAI/bge-m3")
    p.add_argument("--ft_query_ckpt", default="")
    p.add_argument("--ft_pers_ckpt", default="")
    p.add_argument("--rel_rm_ckpt", default="")
    p.add_argument("--split", default="test", choices=["val", "test"])
    p.add_argument("--save_zeroshot_corpus_emb", default="")
    a = p.parse_args()

    run_all_phase1_evaluations(
        bench_dir=a.bench_dir,
        out_dir=a.out_dir,
        zeroshot_model=a.zeroshot_model,
        ft_query_ckpt=a.ft_query_ckpt or None,
        ft_pers_ckpt=a.ft_pers_ckpt or None,
        rel_rm_ckpt=a.rel_rm_ckpt or None,
        split=a.split,
        save_zeroshot_corpus_emb=a.save_zeroshot_corpus_emb or None,
    )


if __name__ == "__main__":
    main()
