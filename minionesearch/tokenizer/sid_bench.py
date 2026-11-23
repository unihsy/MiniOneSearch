"""SID-Bench: Intrinsic Diagnostic Evaluation Suite for Search Semantic ID Tokenizers.

Evaluates any `codes` matrix `[N_items, L]` BEFORE expensive LLM training on 6 search criteria:
  1. Codebook Utilization (%) per layer & mean across layers
  2. Normalized Codebook Perplexity (`exp(H_l) / K_l` in `[0, 1]`, measuring load balance)
  3. Item Collision Rate (`1 - unique_leaf_SIDs / N_items`) & Max Bucket Size
  4. Hierarchical Category Prefix Purity (`L1@Depth1`, `L2@Depth2`, `L3@Depth3`)
  5. Search Co-Click Prefix Alignment (`QueryAlign@Depth1`, `QueryAlign@Depth2`):
     Probability that two items clicked under the same query share the same SID prefix!
  6. Cold-Start Item Semantic Placement (`ColdStartCatPurity@Depth2`):
     Category match rate between cold-start items (`train_pos_count == 0`) and warm neighbors
     sharing the same `<a_i><b_j>` prefix!
"""
from __future__ import annotations

import math
from collections import Counter, defaultdict
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np


def _weighted_cluster_purity(prefixes: Sequence[Tuple[int, ...]], labels: Sequence[int]) -> float:
    buckets: Dict[Tuple[int, ...], Counter] = defaultdict(Counter)
    valid_total = 0
    for pref, lbl in zip(prefixes, labels):
        if int(lbl) <= 0:
            continue
        buckets[pref][int(lbl)] += 1
        valid_total += 1
    if valid_total == 0:
        return 0.0
    correct = sum(cnt.most_common(1)[0][1] for cnt in buckets.values())
    return float(correct / valid_total)


def evaluate_sid_assignment(
    codes: np.ndarray,
    codebook_sizes: Sequence[int],
    corpus_records: Sequence[Mapping[str, Any]],
    eval_sessions: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    """Compute comprehensive intrinsic Search-SID quality metrics (`Table 2`)."""
    codes = np.asarray(codes, dtype=np.int32)
    n_items, n_levels = codes.shape
    if n_items != len(corpus_records):
        raise ValueError(f"codes rows ({n_items}) != corpus_records ({len(corpus_records)})")

    # 1. Layer-wise Codebook Utilization & Normalized Perplexity
    layer_util: List[float] = []
    layer_norm_ppl: List[float] = []
    for l in range(n_levels):
        k_l = int(codebook_sizes[l])
        cnt = np.bincount(codes[:, l], minlength=k_l).astype(np.float64)
        used = int(np.sum(cnt > 0))
        layer_util.append(used / float(k_l))

        probs = cnt[cnt > 0] / float(n_items)
        entropy_nats = float(-np.sum(probs * np.log(probs)))
        ppl = math.exp(entropy_nats)
        layer_norm_ppl.append(ppl / float(k_l))

    # 2. Item Collision Rate & Bucket Statistics (both 3-layer coarse prefix & full 5-layer leaf!)
    full_tuples = [tuple(int(x) for x in row) for row in codes]
    d3_tuples = [row[: min(3, n_levels)] for row in full_tuples]
    d3_unique = len(set(d3_tuples))
    d3_collision_rate = 1.0 - (d3_unique / float(max(1, n_items)))

    leaf_counts = Counter(full_tuples)
    n_unique_sids = len(leaf_counts)
    collision_rate = 1.0 - (n_unique_sids / float(max(1, n_items)))
    max_bucket_size = max(leaf_counts.values()) if leaf_counts else 0

    # 3. Hierarchical Category Prefix Purity
    cat1 = [int(r.get("category_level1_id") or 0) for r in corpus_records]
    cat2 = [int(r.get("category_level2_id") or 0) for r in corpus_records]
    cat3 = [int(r.get("category_level3_id") or 0) for r in corpus_records]

    pref_d1 = [(row[0],) for row in full_tuples]
    pref_d2 = [row[: min(2, n_levels)] for row in full_tuples]
    pref_d3 = [row[: min(3, n_levels)] for row in full_tuples]

    purity_l1_d1 = _weighted_cluster_purity(pref_d1, cat1)
    purity_l2_d2 = _weighted_cluster_purity(pref_d2, cat2)
    purity_l3_d3 = _weighted_cluster_purity(pref_d3, cat3)

    # 4. Search Co-Click Prefix Alignment (`QueryCoClick@Depth1`, `@Depth2`)
    item_id_to_row = {int(r["item_id"]): idx for idx, r in enumerate(corpus_records)}
    query_to_clicked_rows: Dict[str, List[int]] = defaultdict(list)
    for sess in eval_sessions:
        q = str(sess.get("query", "")).strip()
        if not q:
            continue
        pos_ids = set(sess.get("clicked_item_ids") or []) | set(sess.get("purchased_item_ids") or [])
        for iid in pos_ids:
            r_idx = item_id_to_row.get(int(iid))
            if r_idx is not None:
                query_to_clicked_rows[q].append(r_idx)

    pair_cnt = 0
    same_d1 = 0
    same_d2 = 0
    for q, r_list in query_to_clicked_rows.items():
        uniq_r = list(dict.fromkeys(r_list))[:25]
        if len(uniq_r) < 2:
            continue
        for i in range(len(uniq_r)):
            for j in range(i + 1, len(uniq_r)):
                ri, rj = uniq_r[i], uniq_r[j]
                pair_cnt += 1
                if pref_d1[ri] == pref_d1[rj]:
                    same_d1 += 1
                if pref_d2[ri] == pref_d2[rj]:
                    same_d2 += 1

    coclick_d1 = (same_d1 / float(pair_cnt)) if pair_cnt > 0 else 0.0
    coclick_d2 = (same_d2 / float(pair_cnt)) if pair_cnt > 0 else 0.0

    # 5. Cold-Start Item Zero-Shot Prefix Placement (`ColdStartCatPurity@Depth2`)
    warm_d2_cat_votes: Dict[Tuple[int, ...], Counter] = defaultdict(Counter)
    cold_rows: List[int] = []
    for idx, r in enumerate(corpus_records):
        c2 = cat2[idx]
        if int(r.get("train_pos_count") or 0) > 0:
            if c2 > 0:
                warm_d2_cat_votes[pref_d2[idx]][c2] += 1
        else:
            if c2 > 0:
                cold_rows.append(idx)

    cold_hits = 0
    cold_eval_cnt = 0
    for idx in cold_rows:
        votes = warm_d2_cat_votes.get(pref_d2[idx])
        if not votes:
            continue
        cold_eval_cnt += 1
        majority_cat = votes.most_common(1)[0][0]
        if majority_cat == cat2[idx]:
            cold_hits += 1
    cold_start_d2_acc = (cold_hits / float(cold_eval_cnt)) if cold_eval_cnt > 0 else 0.0

    return {
        "mean_codebook_util": round(float(np.mean(layer_util)), 4),
        "layer_codebook_util": [round(u, 4) for u in layer_util],
        "mean_norm_perplexity": round(float(np.mean(layer_norm_ppl)), 4),
        "layer_norm_perplexity": [round(p, 4) for p in layer_norm_ppl],
        "icr": round(float(1.0 - collision_rate), 5),
        "d3_collision_rate": round(float(d3_collision_rate), 5),
        "collision_rate": round(float(collision_rate), 5),
        "unique_sids": int(n_unique_sids),
        "max_bucket_size": int(max_bucket_size),
        "cat_purity_l1_d1": round(purity_l1_d1, 4),
        "cat_purity_l2_d2": round(purity_l2_d2, 4),
        "cat_purity_l3_d3": round(purity_l3_d3, 4),
        "query_coclick_align_d1": round(coclick_d1, 4),
        "query_coclick_align_d2": round(coclick_d2, 4),
        "cold_start_d2_cat_acc": round(cold_start_d2_acc, 4),
    }


def render_sid_bench_markdown(bench_results: Mapping[str, Mapping[str, Any]]) -> str:
    """Render publication-ready Markdown Table 2 (`SID-Bench`, reporting OneSearch CUR & ICR!)."""
    headers = [
        "Tokenizer",
        "CUR (Util) ↑",
        "Norm PPL ↑",
        "D3 Coarse Coll.",
        "L5 ICR (Unique) ↑",
        "L5 Collision ↓",
        "Max Bucket ↓",
        "CatPurity (D1/D2/D3) ↑",
        "QueryCoClick (D1/D2) ↑",
        "ColdStart D2 Acc ↑",
    ]
    lines = ["# Phase 2 SID-Bench: Search Semantic ID Tokenizer Comparison\n"]
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("|" + "|".join(["---"] * len(headers)) + "|")

    for name, m in bench_results.items():
        icr_val = float(m.get("icr", 1.0 - float(m["collision_rate"])))
        d3_coll = float(m.get("d3_collision_rate", 0.0))
        cat_s = (
            f"{m['cat_purity_l1_d1'] * 100:.1f}% / "
            f"{m['cat_purity_l2_d2'] * 100:.1f}% / "
            f"{m['cat_purity_l3_d3'] * 100:.1f}%"
        )
        q_s = f"{m['query_coclick_align_d1'] * 100:.1f}% / {m['query_coclick_align_d2'] * 100:.1f}%"
        row = [
            f"**{name}**",
            f"{m['mean_codebook_util'] * 100:.2f}%",
            f"{m['mean_norm_perplexity'] * 100:.2f}%",
            f"{d3_coll * 100:.2f}%",
            f"{icr_val * 100:.2f}%",
            f"{m['collision_rate'] * 100:.2f}%",
            f"{m['max_bucket_size']}",
            cat_s,
            q_s,
            f"{m['cold_start_d2_cat_acc'] * 100:.2f}%",
        ]
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines) + "\n"
