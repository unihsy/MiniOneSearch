"""Composite Multi-Objective Search Reward Engine for Generative Search RL.

Combines four complementary search ranking signals over `G` candidate rollouts per query:
  1. `R_conv`: Behavioral Conversion Hierarchy (`purchase=4.0 > click=3.0 > unimpressed=0.0 > impressed_unclicked=-0.3`)
  2. `R_rel`: Calibrated 4-Tier Semantic Relevance (`[0.0, 3.0]` from trained Cross-Encoder Judge,
     mitigating offline exposure bias on relevant unexposed items & penalizing attribute mismatches!)
  3. `R_ctr`: Bayesian Smoothed Train-Split Item Posterior CTR Prior (`[0.0, 1.0]`)
  4. `R_div`: Rank-Ordered Sub-Category Diversity Term (penalizes redundant L2/L3 category flooding
     among non-positive beams while rewarding relevant diverse coverage)
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple

import numpy as np
import torch


@dataclass
class CompositeRewardWeights:
    w_conv: float = 1.0
    w_rel: float = 0.8
    w_ctr: float = 0.4
    w_div: float = 0.3
    v_purchase: float = 4.0
    v_click: float = 3.0
    v_impressed_neg: float = -0.30
    redundancy_cat_cap: int = 2


class SearchRewardEngine:
    """Computes composite item-level rewards `[B, G]` and diagnostic sub-metrics."""

    def __init__(
        self,
        corpus_records: Sequence[Mapping[str, Any]],
        weights: Optional[CompositeRewardWeights] = None,
        rel_scorer: Optional[Any] = None,
    ):
        self.weights = weights or CompositeRewardWeights()
        self.rel_scorer = rel_scorer

        self.item_text_map: Dict[int, str] = {}
        self.item_ctr_map: Dict[int, float] = {}
        self.item_cat_map: Dict[int, int] = {}

        max_ctr = 1e-6
        raw_ctrs: Dict[int, float] = {}
        for r in corpus_records:
            iid = int(r["item_id"])
            self.item_text_map[iid] = str(r.get("structured_text") or r.get("item_title") or "")
            c_val = float(r.get("posterior_ctr") or 0.0)
            raw_ctrs[iid] = c_val
            if c_val > max_ctr:
                max_ctr = c_val
            cat = int(r.get("category_level3_id") or r.get("category_level2_id") or 0)
            self.item_cat_map[iid] = cat

        # Normalize posterior CTR to `[0, 1]` using 99th percentile cap so outliers don't dominate
        p99 = float(np.percentile(list(raw_ctrs.values()), 99)) if raw_ctrs else 1.0
        denom = max(p99, 1e-4)
        for iid, c_val in raw_ctrs.items():
            self.item_ctr_map[iid] = float(min(1.0, max(0.0, c_val / denom)))

    def score_rollout_groups(
        self,
        sessions: Sequence[Mapping[str, Any]],
        rollout_item_ids: Sequence[Sequence[Optional[int]]],  # shape [B][G], ordered by policy score
        rel_batch_size: int = 256,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Score `[B, G]` candidate item IDs and return `(rewards_tensor [B, G], metrics_dict)`."""
        bsz = len(sessions)
        g_size = len(rollout_item_ids[0]) if bsz > 0 else 0
        w = self.weights

        r_conv = np.zeros((bsz, g_size), dtype=np.float32)
        r_rel = np.zeros((bsz, g_size), dtype=np.float32)
        r_ctr = np.zeros((bsz, g_size), dtype=np.float32)
        r_div = np.zeros((bsz, g_size), dtype=np.float32)

        # Batch-collect unique `(query, item_id)` pairs for fast GPU Cross-Encoder relevance scoring
        pair_keys: List[Tuple[str, int]] = []
        seen_pairs: Set[Tuple[str, int]] = set()
        if self.rel_scorer is not None:
            for b_i, sess in enumerate(sessions):
                q = str(sess["query"])
                for iid in rollout_item_ids[b_i]:
                    if iid is not None and iid in self.item_text_map:
                        pk = (q, int(iid))
                        if pk not in seen_pairs:
                            seen_pairs.add(pk)
                            pair_keys.append(pk)

        rel_lookup: Dict[Tuple[str, int], float] = {}
        if self.rel_scorer is not None and pair_keys:
            qs = [pk[0] for pk in pair_keys]
            ds = [self.item_text_map[pk[1]] for pk in pair_keys]
            scores_arr = self.rel_scorer.score_pairs(qs, ds, batch_size=rel_batch_size)
            for pk, sc in zip(pair_keys, scores_arr):
                rel_lookup[pk] = float(sc)

        for b_i, sess in enumerate(sessions):
            q = str(sess["query"])
            purch_set = {int(x) for x in (sess.get("purchased_item_ids") or [])}
            click_set = {int(x) for x in (sess.get("clicked_item_ids") or [])}
            pos_set = purch_set | click_set
            imp_neg_set = {int(x) for x in (sess.get("impressed_item_ids") or [])} - pos_set

            cat_seen_counts: Counter = Counter()
            item_seen_in_group: Set[int] = set()

            for g_j, iid_opt in enumerate(rollout_item_ids[b_i]):
                if iid_opt is None or int(iid_opt) not in self.item_text_map:
                    r_conv[b_i, g_j] = -1.0
                    continue
                iid = int(iid_opt)

                # Penalize exact duplicate rollout if unconstrained sampling was used
                if iid in item_seen_in_group:
                    r_div[b_i, g_j] = -1.0
                    continue
                item_seen_in_group.add(iid)

                # 1. Conversion hierarchy
                if iid in purch_set:
                    r_conv[b_i, g_j] = w.v_purchase
                elif iid in click_set:
                    r_conv[b_i, g_j] = w.v_click
                elif iid in imp_neg_set:
                    r_conv[b_i, g_j] = w.v_impressed_neg
                else:
                    r_conv[b_i, g_j] = 0.0

                # 2. 4-Tier Relevance [0, 3]
                if self.rel_scorer is not None:
                    r_rel[b_i, g_j] = rel_lookup.get((q, iid), 0.0)
                else:
                    # Fallback lexical/category proxy when running lightweight unit tests without GPU RM
                    r_rel[b_i, g_j] = 2.5 if iid in pos_set else 0.5

                # 3. Calibrated posterior CTR [0, 1]
                r_ctr[b_i, g_j] = self.item_ctr_map.get(iid, 0.0)

                # 4. Rank-ordered sub-category diversity
                cat_id = self.item_cat_map.get(iid, 0)
                if cat_id > 0:
                    prev_cat_cnt = cat_seen_counts[cat_id]
                    if prev_cat_cnt == 0 and r_rel[b_i, g_j] >= 1.8:
                        r_div[b_i, g_j] = 0.4  # reward relevant fresh sub-category
                    elif prev_cat_cnt >= w.redundancy_cat_cap and iid not in pos_set:
                        r_div[b_i, g_j] = -0.5  # penalize repetitive SKU flooding
                    cat_seen_counts[cat_id] += 1

        total = w.w_conv * r_conv + w.w_rel * r_rel + w.w_ctr * r_ctr + w.w_div * r_div
        metrics = {
            "reward/total_mean": float(np.mean(total)),
            "reward/total_std": float(np.std(total)),
            "reward/conv_mean": float(np.mean(r_conv)),
            "reward/pos_hit_rate": float(np.mean(r_conv >= w.v_click)),
            "reward/rel_mean": float(np.mean(r_rel)),
            "reward/ctr_mean": float(np.mean(r_ctr)),
            "reward/div_mean": float(np.mean(r_div)),
        }
        return torch.from_numpy(total), metrics
