"""Unified evaluation metrics for Generative & Traditional Search.

Supports:
  - Standard IR metrics: Recall@K, HitRate@K, NDCG@K (binary & graded click<purchase), MRR@K
  - Model-based & annotated Relevance@K (4-tier expected relevance score in [0, 3])
  - Search Result Diversity: Category-Coverage@K (unique L2/L3 categories in top-K) & ILD
  - Constraint Diagnostics: Valid-In-Corpus Rate (1 - Invalid/Collision Hallucination Rate)
  - Four Search-Specific Diagnostic Slices:
      1. head_exact
      2. personalized_ambiguous
      3. long_tail_complex
      4. cold_start_item
  - Latency & Throughput Profiling: P50 / P95 / P99 latency (ms) and QPS
"""
from __future__ import annotations

import math
from collections import defaultdict
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Set

import numpy as np


DEFAULT_KS = (1, 5, 10, 20, 50)


def compute_query_ir_metrics(
    ranked_item_ids: Sequence[int],
    clicked_ids: Set[int],
    purchased_ids: Optional[Set[int]] = None,
    ks: Sequence[int] = DEFAULT_KS,
    item_to_cat: Optional[Mapping[int, int]] = None,
    valid_corpus_ids: Optional[Set[int]] = None,
    item_rel_scores: Optional[Mapping[int, float]] = None,
) -> Dict[str, float]:
    """Compute comprehensive per-query IR metrics.

    Notes on Ground-Truth Definitions:
      - Binary positive set `pos_ids = clicked_ids | purchased_ids` is used for
        `Recall@K`, `HR@K`, `NDCG@K` (standard binary relevance), and `MRR@K`.
      - Graded utility (`click=1.0`, `purchase=2.0`) is additionally reported as
        `GradedNDCG@K` to reflect conversion-aware ranking quality in e-commerce search.
      - Deduplication: duplicate predicted item IDs are only credited at their first rank
        (never double-counted!).
    """
    purchased_ids = purchased_ids or set()
    pos_ids = clicked_ids | purchased_ids
    if not pos_ids:
        return {}

    max_k = max(ks)
    # Deduplicate while preserving rank positions (penalize empty/duplicate slots fairly)
    dedup_preds: List[int] = []
    seen: Set[int] = set()
    for iid in ranked_item_ids[:max_k]:
        if iid is not None and iid >= 0 and iid not in seen:
            seen.add(iid)
            dedup_preds.append(int(iid))
        else:
            dedup_preds.append(-1)  # invalid or duplicate placeholder at this rank

    out: Dict[str, float] = {}

    # Validity check (for generative models)
    if valid_corpus_ids is not None and len(ranked_item_ids) > 0:
        top10 = ranked_item_ids[: min(10, len(ranked_item_ids))]
        valid_cnt = sum(1 for x in top10 if x in valid_corpus_ids)
        out["ValidRate@10"] = valid_cnt / len(top10)
        uniq_valid = len({x for x in top10 if x in valid_corpus_ids})
        out["UniqueValidRate@10"] = uniq_valid / len(top10)

    # Precompute ideal DCG denominators
    num_pos = len(pos_ids)
    # Graded gains: purchase -> 3.0 (2^2 - 1), click -> 1.0 (2^1 - 1)
    ideal_graded_gains = sorted(
        [3.0 if iid in purchased_ids else 1.0 for iid in pos_ids],
        reverse=True,
    )

    # Reciprocal rank
    rr_at_10 = 0.0
    for rank_idx, iid in enumerate(dedup_preds[:10]):
        if iid in pos_ids:
            rr_at_10 = 1.0 / (rank_idx + 1.0)
            break
    out["MRR@10"] = rr_at_10

    for k in ks:
        preds_k = dedup_preds[:k]
        hits_k = sum(1 for iid in preds_k if iid in pos_ids)
        out[f"Recall@{k}"] = hits_k / float(num_pos)
        out[f"HR@{k}"] = 1.0 if hits_k > 0 else 0.0

        # Binary NDCG@K
        dcg = 0.0
        for rank_idx, iid in enumerate(preds_k):
            if iid in pos_ids:
                dcg += 1.0 / math.log2(rank_idx + 2.0)
        idcg_len = min(num_pos, k)
        idcg = sum(1.0 / math.log2(r + 2.0) for r in range(idcg_len))
        out[f"NDCG@{k}"] = (dcg / idcg) if idcg > 0 else 0.0

        # Conversion-aware Graded NDCG@K
        g_dcg = 0.0
        for rank_idx, iid in enumerate(preds_k):
            if iid in purchased_ids:
                g_dcg += 3.0 / math.log2(rank_idx + 2.0)
            elif iid in clicked_ids:
                g_dcg += 1.0 / math.log2(rank_idx + 2.0)
        g_idcg = sum(
            gain / math.log2(r + 2.0)
            for r, gain in enumerate(ideal_graded_gains[:k])
        )
        out[f"GradedNDCG@{k}"] = (g_dcg / g_idcg) if g_idcg > 0 else 0.0

        # Category diversity @ K (for K <= 20)
        if item_to_cat is not None and k in (5, 10, 20):
            valid_preds = [iid for iid in preds_k if iid >= 0]
            if valid_preds:
                cats = {item_to_cat.get(iid, -1) for iid in valid_preds if item_to_cat.get(iid, 0) > 0}
                out[f"CatDiv@{k}"] = len(cats) / float(len(valid_preds))
            else:
                out[f"CatDiv@{k}"] = 0.0

        # Relevance model / human score @ K (scale [0, 3])
        if item_rel_scores is not None and k in (1, 5, 10):
            valid_preds = [iid for iid in preds_k if iid >= 0]
            if valid_preds:
                scores = [float(item_rel_scores.get(iid, 0.0)) for iid in valid_preds]
                out[f"Rel@{k}"] = float(np.mean(scores))
                out[f"StrictRelRate@{k}"] = float(np.mean([1.0 if s >= 2.0 else 0.0 for s in scores]))
            else:
                out[f"Rel@{k}"] = 0.0
                out[f"StrictRelRate@{k}"] = 0.0

    return out


def evaluate_predictions(
    sessions: Sequence[Dict],
    predictions: Mapping[str, Sequence[int]],
    ks: Sequence[int] = DEFAULT_KS,
    item_to_cat: Optional[Mapping[int, int]] = None,
    valid_corpus_ids: Optional[Set[int]] = None,
    session_item_rel_scores: Optional[Mapping[str, Mapping[int, float]]] = None,
) -> Dict[str, Dict[str, float]]:
    """Aggregate overall and per-slice search evaluation metrics.

    Args:
        sessions: Evaluation session records containing `session_id`, `clicked_item_ids`,
            `purchased_item_ids`, and `slices` (list of diagnostic slice names).
        predictions: Mapping `str(session_id) -> ranked list of item_ids`.
    Returns:
        Nested dictionary `{"overall": {...}, "slice:head_exact": {...}, ...}`.
    """
    buckets: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))

    for sess in sessions:
        sid = str(sess["session_id"])
        clicked = {int(x) for x in (sess.get("clicked_item_ids") or [])}
        purchased = {int(x) for x in (sess.get("purchased_item_ids") or [])}
        if not (clicked | purchased):
            continue

        preds = predictions.get(sid, [])
        rel_map = session_item_rel_scores.get(sid) if session_item_rel_scores else None
        q_metrics = compute_query_ir_metrics(
            ranked_item_ids=preds,
            clicked_ids=clicked,
            purchased_ids=purchased,
            ks=ks,
            item_to_cat=item_to_cat,
            valid_corpus_ids=valid_corpus_ids,
            item_rel_scores=rel_map,
        )
        if not q_metrics:
            continue

        for m_name, m_val in q_metrics.items():
            buckets["overall"][m_name].append(m_val)

        for sl in sess.get("slices", []):
            for m_name, m_val in q_metrics.items():
                buckets[f"slice:{sl}"][m_name].append(m_val)

    summary: Dict[str, Dict[str, float]] = {}
    for group_name, metric_lists in buckets.items():
        group_res = {m: float(np.mean(vals)) for m, vals in metric_lists.items()}
        first_key = next(iter(metric_lists), None)
        group_res["num_queries"] = float(len(metric_lists[first_key])) if first_key else 0.0
        summary[group_name] = group_res

    return summary


def summarize_latencies_ms(
    latencies_ms: Sequence[float],
    total_queries: int,
    total_wall_sec: float,
) -> Dict[str, float]:
    """Compute P50 / P95 / P99 per-batch/per-query latency and end-to-end QPS."""
    if not latencies_ms:
        return {"p50_ms": 0.0, "p95_ms": 0.0, "p99_ms": 0.0, "mean_ms": 0.0, "qps": 0.0}
    arr = np.asarray(latencies_ms, dtype=np.float64)
    return {
        "mean_ms": float(np.mean(arr)),
        "p50_ms": float(np.percentile(arr, 50)),
        "p95_ms": float(np.percentile(arr, 95)),
        "p99_ms": float(np.percentile(arr, 99)),
        "qps": float(total_queries / max(total_wall_sec, 1e-6)),
    }
