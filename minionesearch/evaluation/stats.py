"""Statistical Significance Testing & Multi-Seed Aggregation for Search Benchmarks.

Implements paired bootstrap inference and multiple-testing adjustment for user-session search logs:
  1. **User-Clustered Paired Bootstrap (`n_boot = 10,000`)**:
     Accounts for intra-user correlation when multiple test sessions belong to the same
     `user_id` by resampling unique users with replacement and weighting by user session sums.
  2. **Session-Level Paired Bootstrap**:
     Standard query-level paired bootstrap for comparison.
  3. **Holm-Bonferroni Step-Down Multiple-Testing Correction**:
     Controls Family-Wise Error Rate (FWER) across hypothesis families.
  4. **Multi-Seed Mean ± Std Summary**:
     Aggregates independent training seeds (`seed=42, 43, 44`).
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np

from minionesearch.evaluation import compute_query_ir_metrics


def compute_per_session_metric_map(
    sessions: Sequence[Mapping[str, Any]],
    predictions: Mapping[str, Sequence[int]],
    metric_key: str = "NDCG@10",
    ks: Sequence[int] = (1, 5, 10, 20),
) -> Dict[str, float]:
    """Return `session_id -> scalar metric` across all valid positive evaluation sessions."""
    out: Dict[str, float] = {}
    for sess in sessions:
        sid = str(sess["session_id"])
        clicked = {int(x) for x in (sess.get("clicked_item_ids") or [])}
        purchased = {int(x) for x in (sess.get("purchased_item_ids") or [])}
        if not (clicked | purchased):
            continue
        qm = compute_query_ir_metrics(predictions.get(sid, []), clicked, purchased, ks=ks)
        if metric_key in qm:
            out[sid] = float(qm[metric_key])
    return out


def user_clustered_paired_bootstrap(
    sessions: Sequence[Mapping[str, Any]],
    preds_a: Mapping[str, Sequence[int]],
    preds_b: Mapping[str, Sequence[int]],
    metric_key: str = "NDCG@10",
    n_boot: int = 10_000,
    seed: int = 42,
) -> Dict[str, float]:
    """Compute paired difference `A - B` with 95% User-Clustered AND Session-Level Bootstrap CIs.

    Vectorized cluster resampling:
      Let user cluster `u` have `n_u` sessions with difference sum `S_u = sum_{i in u} (a_i - b_i)`.
      A cluster bootstrap draw samples `U` users with replacement (counts `w_u` ~ Multinomial)
      and forms `Delta* = (sum_u w_u S_u) / (sum_u w_u n_u)`.
    """
    map_a = compute_per_session_metric_map(sessions, preds_a, metric_key=metric_key)
    map_b = compute_per_session_metric_map(sessions, preds_b, metric_key=metric_key)

    diffs_sess: List[float] = []
    vals_a: List[float] = []
    vals_b: List[float] = []
    user_diff_sums: Dict[int, float] = defaultdict(float)
    user_sess_cnts: Dict[int, int] = defaultdict(int)

    for sess in sessions:
        sid = str(sess["session_id"])
        if sid not in map_a or sid not in map_b:
            continue
        va, vb = map_a[sid], map_b[sid]
        d = va - vb
        vals_a.append(va)
        vals_b.append(vb)
        diffs_sess.append(d)
        uid = int(sess.get("user_id", hash(sid)))
        user_diff_sums[uid] += d
        user_sess_cnts[uid] += 1

    n_sess = len(diffs_sess)
    if n_sess == 0:
        return {
            "mean_a": 0.0, "mean_b": 0.0, "diff": 0.0,
            "ci95_low_cluster": 0.0, "ci95_high_cluster": 0.0, "p_two_sided_cluster": 1.0,
            "ci95_low_session": 0.0, "ci95_high_session": 0.0, "p_two_sided_session": 1.0,
            "num_sessions": 0.0, "num_users": 0.0,
        }

    arr_d = np.asarray(diffs_sess, dtype=np.float64)
    obs_diff = float(np.mean(arr_d))
    rng = np.random.default_rng(seed)

    # 1. User-clustered bootstrap (`[n_boot, n_users]` integer index draw in chunks)
    uids = sorted(user_diff_sums.keys())
    n_users = len(uids)
    u_sums = np.asarray([user_diff_sums[u] for u in uids], dtype=np.float64)
    u_cnts = np.asarray([user_sess_cnts[u] for u in uids], dtype=np.float64)

    boot_cluster = np.empty(n_boot, dtype=np.float64)
    chunk = 1000
    for st in range(0, n_boot, chunk):
        ed = min(n_boot, st + chunk)
        idx = rng.integers(0, n_users, size=(ed - st, n_users))
        num = u_sums[idx].sum(axis=1)
        den = np.maximum(u_cnts[idx].sum(axis=1), 1.0)
        boot_cluster[st:ed] = num / den

    ci_low_c, ci_high_c = float(np.percentile(boot_cluster, 2.5)), float(np.percentile(boot_cluster, 97.5))
    # Centered bootstrap two-sided p-value (Hall & Wilson 1991 / Efron & Tibshirani 1993)
    centered_c = boot_cluster - obs_diff
    p_val_c = float((np.sum(np.abs(centered_c) >= abs(obs_diff)) + 1.0) / (n_boot + 1.0))

    # 2. Session-level paired bootstrap
    boot_sess = np.empty(n_boot, dtype=np.float64)
    for st in range(0, n_boot, chunk):
        ed = min(n_boot, st + chunk)
        s_idx = rng.integers(0, n_sess, size=(ed - st, n_sess))
        boot_sess[st:ed] = arr_d[s_idx].mean(axis=1)

    ci_low_s, ci_high_s = float(np.percentile(boot_sess, 2.5)), float(np.percentile(boot_sess, 97.5))
    centered_s = boot_sess - obs_diff
    p_val_s = float((np.sum(np.abs(centered_s) >= abs(obs_diff)) + 1.0) / (n_boot + 1.0))

    return {
        "mean_a": float(np.mean(vals_a)),
        "mean_b": float(np.mean(vals_b)),
        "diff": obs_diff,
        "ci95_low_cluster": ci_low_c,
        "ci95_high_cluster": ci_high_c,
        "p_two_sided_cluster": min(1.0, p_val_c),
        "ci95_low_session": ci_low_s,
        "ci95_high_session": ci_high_s,
        "p_two_sided_session": min(1.0, p_val_s),
        "num_sessions": float(n_sess),
        "num_users": float(n_users),
    }


def holm_bonferroni_adjust(p_values: Sequence[float]) -> List[float]:
    """Compute step-down Holm-Bonferroni FWER-adjusted p-values."""
    m = len(p_values)
    if m == 0:
        return []
    order = sorted(range(m), key=lambda i: p_values[i])
    adjusted = [0.0] * m
    running_max = 0.0
    for rank_idx, orig_i in enumerate(order):
        mult = m - rank_idx
        val = min(1.0, float(p_values[orig_i]) * mult)
        running_max = max(running_max, val)
        adjusted[orig_i] = running_max
    return adjusted


def run_hypothesis_family(
    sessions: Sequence[Mapping[str, Any]],
    comparisons: Sequence[Tuple[str, Mapping[str, Sequence[int]], Mapping[str, Sequence[int]]]],
    metric_key: str = "NDCG@10",
    n_boot: int = 10_000,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """Evaluate a family of paired comparisons and attach Holm-Bonferroni adjusted p-values."""
    rows: List[Dict[str, Any]] = []
    for idx, (label, pa, pb) in enumerate(comparisons):
        res = user_clustered_paired_bootstrap(
            sessions=sessions,
            preds_a=pa,
            preds_b=pb,
            metric_key=metric_key,
            n_boot=n_boot,
            seed=seed + idx * 13,
        )
        rows.append({"comparison": label, "metric": metric_key, **res})

    raw_ps = [r["p_two_sided_cluster"] for r in rows]
    holm_ps = holm_bonferroni_adjust(raw_ps)
    for r, hp in zip(rows, holm_ps):
        r["p_holm_cluster"] = hp
    return rows


def aggregate_multiseed_metrics(seed_metric_dicts: Sequence[Mapping[str, float]]) -> Dict[str, Dict[str, float]]:
    """Aggregate `{metric: val}` across multiple independent training seeds into `{metric: {mean, std, n}}`."""
    if not seed_metric_dicts:
        return {}
    keys = sorted(seed_metric_dicts[0].keys())
    out: Dict[str, Dict[str, float]] = {}
    for k in keys:
        vals = [float(d[k]) for d in seed_metric_dicts if k in d]
        if vals:
            out[k] = {
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
                "min": float(np.min(vals)),
                "max": float(np.max(vals)),
                "n_seeds": float(len(vals)),
            }
    return out
