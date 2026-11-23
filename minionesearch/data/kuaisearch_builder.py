"""Deterministic chronological KuaiSearch benchmark builder (`core5`, `mini`, and `scale` tiers).

Constructs standardized Generative Search benchmarks from `benchen4395/KuaiSearch`:
  1. Causal user search & interaction histories (`hist_time < cur_time`).
  2. Chronological validation holdout from native `train`, preserving native `test`.
  3. Training-split statistics for Bayesian posterior CTR, cold-start item tagging,
     and query intent entropy.
  4. Four diagnostic evaluation slices on val/test:
       - `head_exact`
       - `personalized_ambiguous`
       - `long_tail_complex`
       - `cold_start_item`
"""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

import numpy as np
from tqdm import tqdm

from minionesearch.utils import iter_jsonl, set_global_seed, stable_hash_int, write_jsonl


PLACEHOLDER_TOKENS = {"", "UNKNOWN", "unknown", "无品牌", "其他/other", "其他", "\\N", "None", "null"}


@dataclass
class BuildConfig:
    tier: str = "mini"  # "core5" (true iterative 5-core), "mini" (100K sparse), or "scale" (500K industrial)
    target_corpus_size: int = 100_000
    max_users: Optional[int] = 16_000  # None for full scale/core5 tier
    core_item_k: int = 5
    core_user_k: int = 1
    max_hist_queries: int = 15
    max_hist_items: int = 20
    val_ratio: float = 0.05
    seed: int = 42
    use_rank_prehistory: bool = False  # Pure timestamped `< cur_t` causal history by default!


def clean_field(val: Any) -> str:
    if val is None:
        return ""
    s = str(val).strip()
    return "" if s in PLACEHOLDER_TOKENS else s


def format_item_structured_text(item: Dict[str, Any]) -> str:
    """Format an item record into clean structured text for embedding & lexical indexing."""
    title = clean_field(item.get("item_title") or item.get("title"))
    brand = clean_field(item.get("brand_name") or item.get("brand"))
    cats = [
        clean_field(item.get("category_level1_name")),
        clean_field(item.get("category_level2_name")),
        clean_field(item.get("category_level3_name")),
    ]
    cat_path = "/".join([c for c in cats if c])
    parts = [title] if title else []
    if brand:
        parts.append(f"品牌:{brand}")
    if cat_path:
        parts.append(f"类目:{cat_path}")
    return " | ".join(parts)


def download_kuaisearch_raw(
    raw_dir: str | Path,
    include_rank_lite: bool = False,
    repo_id: str = "benchen4395/KuaiSearch",
) -> Path:
    """Download required KuaiSearch Lite + Relevance JSONL tables from HuggingFace Hub."""
    from huggingface_hub import snapshot_download

    raw_path = Path(raw_dir)
    raw_path.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=str(raw_path),
        allow_patterns=["users_lite/*", "items_lite/*", "recall_lite/*", "relevance/*"],
    )
    if include_rank_lite:
        try:
            snapshot_download(
                repo_id=repo_id,
                repo_type="dataset",
                local_dir=str(raw_path),
                allow_patterns=["rank_lite/*"],
            )
        except Exception as exc:
            print(f"[WARN] Optional rank_lite download skipped ({exc}); proceeding with causal recall_lite history.")
    return raw_path


def _resolve_table_file(raw_dir: Path, candidates: Sequence[str]) -> Optional[Path]:
    for rel in candidates:
        p = raw_dir / rel
        if p.is_file():
            return p
    return None


def _compute_normalized_entropy(counts: Counter) -> float:
    total = sum(counts.values())
    if total <= 0 or len(counts) <= 1:
        return 0.0
    ent = -sum((c / total) * math.log2(c / total) for c in counts.values() if c > 0)
    return ent


def _load_rank_prehistory_for_sessions(
    rank_file: Path,
    target_session_ids: Set[str],
) -> Dict[str, Tuple[List[int], List[int]]]:
    """Stream `rank_lite/train.jsonl` once (with compact session-level cache) to extract pre-session histories."""
    prehist: Dict[str, Tuple[List[int], List[int]]] = {}
    if not rank_file or not rank_file.is_file():
        return prehist

    cache_file = rank_file.parent / "session_prehist_compact.jsonl"
    if cache_file.is_file():
        for rec in iter_jsonl(cache_file):
            sid = str(rec["sid"])
            if sid in target_session_ids:
                prehist[sid] = (rec["rc"], rec["rp"])
        return prehist

    all_compact: List[Dict[str, Any]] = []
    seen_all: Set[str] = set()
    last_sid_str = ""
    for rec in tqdm(iter_jsonl(rank_file), desc="Streaming rank pre-history (one-time compact index)", mininterval=5.0):
        sid = str(rec.get("session_id", ""))
        if not sid or sid == last_sid_str or sid in seen_all:
            last_sid_str = sid
            continue
        last_sid_str = sid
        seen_all.add(sid)
        rc = [int(x) for x in (rec.get("recently_clicked_item_ids") or []) if x is not None and int(x) > 0]
        rp = [int(x) for x in (rec.get("recently_purchased_item_ids") or []) if x is not None and int(x) > 0]
        all_compact.append({"sid": sid, "rc": rc, "rp": rp})
        if sid in target_session_ids:
            prehist[sid] = (rc, rp)

    try:
        write_jsonl(cache_file, all_compact)
    except OSError:
        pass
    return prehist


def build_benchmark_from_raw(
    raw_dir: str | Path,
    out_dir: str | Path,
    cfg: BuildConfig,
) -> Dict[str, Any]:
    """Build deterministic chronological `KuaiSearch` benchmark (`core5`, `mini`, or `scale`)."""
    set_global_seed(cfg.seed)
    raw_path = Path(raw_dir)
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    users_file = _resolve_table_file(raw_path, ["users_lite/train.jsonl", "users.jsonl"])
    items_file = _resolve_table_file(raw_path, ["items_lite/train.jsonl", "items.jsonl"])
    recall_file = _resolve_table_file(raw_path, ["recall_lite/train.jsonl", "recall.jsonl"])
    rel_file = _resolve_table_file(raw_path, ["relevance/train.jsonl", "relevance.jsonl"])
    rank_file = _resolve_table_file(raw_path, ["rank_lite/train.jsonl", "rank.jsonl"])

    if not items_file or not recall_file:
        raise FileNotFoundError(
            f"Missing required items/recall tables under {raw_path}. Found items={items_file}, recall={recall_file}"
        )

    # 1. Scan valid item IDs & compact metadata from items table
    print(f"[Builder:{cfg.tier}] Scanning item catalog from {items_file} ...")
    all_items: Dict[int, Dict[str, Any]] = {}
    cat1_buckets: Dict[int, List[int]] = defaultdict(list)
    for obj in tqdm(iter_jsonl(items_file), desc="Loading items", mininterval=5.0):
        iid_raw = obj.get("item_id")
        title = clean_field(obj.get("item_title") or obj.get("title"))
        if iid_raw is None or not title:
            continue
        iid = int(iid_raw)
        c1_id = int(obj.get("category_level1_id") or 0)
        c2_id = int(obj.get("category_level2_id") or 0)
        c3_id = int(obj.get("category_level3_id") or 0)
        item_rec = {
            "item_id": iid,
            "item_title": title,
            "brand_name": clean_field(obj.get("brand_name") or obj.get("brand")),
            "seller_name": clean_field(obj.get("seller_name") or obj.get("seller")),
            "category_level1_id": c1_id,
            "category_level1_name": clean_field(obj.get("category_level1_name")),
            "category_level2_id": c2_id,
            "category_level2_name": clean_field(obj.get("category_level2_name")),
            "category_level3_id": c3_id,
            "category_level3_name": clean_field(obj.get("category_level3_name")),
        }
        all_items[iid] = item_rec
        cat1_buckets[c1_id].append(iid)

    print(f"[Builder:{cfg.tier}] Usable items in catalog: {len(all_items):,}")

    # 2. Load users
    users_map: Dict[int, Dict[str, Any]] = {}
    if users_file:
        for u in iter_jsonl(users_file):
            uid = u.get("user_id")
            if uid is not None:
                users_map[int(uid)] = {
                    "user_id": int(uid),
                    "gender": clean_field(u.get("gender")),
                    "age_bucket": clean_field(u.get("age_bucket") or u.get("age")),
                    "province": clean_field(u.get("fre_province") or u.get("location")),
                    "city": clean_field(u.get("fre_city")),
                }

    # 3. Load recall sessions & filter to positive sessions whose clicked/purchased items exist in catalog
    user_sessions_raw: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    seen_session_ids: Set[str] = set()
    for rec in tqdm(iter_jsonl(recall_file), desc="Loading recall sessions", mininterval=5.0):
        sid = rec.get("session_id")
        uid = rec.get("user_id")
        query = clean_field(rec.get("query"))
        raw_split = str(rec.get("split", "train")).lower()
        if sid is None or uid is None or not query or raw_split not in ("train", "test"):
            continue
        sid_str = str(sid)
        if sid_str in seen_session_ids:
            continue

        clicked = [int(x) for x in (rec.get("clicked_item_ids") or []) if int(x) in all_items]
        purchased = [int(x) for x in (rec.get("purchased_item_ids") or []) if int(x) in all_items]
        if not clicked and not purchased:
            continue

        impressed = [int(x) for x in (rec.get("impressed_item_ids") or []) if int(x) in all_items]
        t_idx = int(rec.get("time_index") or 0)
        seen_session_ids.add(sid_str)
        user_sessions_raw[int(uid)].append(
            {
                "session_id": sid_str,
                "user_id": int(uid),
                "time_index": t_idx,
                "query": query,
                "clicked_item_ids": list(dict.fromkeys(clicked)),
                "purchased_item_ids": list(dict.fromkeys(purchased)),
                "impressed_item_ids": list(dict.fromkeys(impressed)),
                "raw_split": raw_split,
            }
        )

    # 4. Select User Universe & Target Item Pool for the requested tier
    if cfg.tier == "mini":
        # Prioritize active users who appear in both train and test for rich personalization evaluation
        both_users: List[int] = []
        other_active_users: List[int] = []
        for uid, s_list in user_sessions_raw.items():
            n_tr = sum(1 for s in s_list if s["raw_split"] == "train")
            n_te = sum(1 for s in s_list if s["raw_split"] == "test")
            if n_tr >= 2 and n_te >= 1:
                both_users.append(uid)
            elif len(s_list) >= 2:
                other_active_users.append(uid)

        both_users.sort(key=lambda u: stable_hash_int(f"mini_both_{u}_{cfg.seed}"))
        other_active_users.sort(key=lambda u: stable_hash_int(f"mini_other_{u}_{cfg.seed}"))
        ordered_candidates = both_users + other_active_users
        if not ordered_candidates:
            ordered_candidates = sorted(
                user_sessions_raw.keys(),
                key=lambda u: stable_hash_int(f"mini_any_{u}_{cfg.seed}"),
            )

        pos_budget = int(cfg.target_corpus_size * 0.55)
        selected_users: Set[int] = set()
        required_pos_items: Set[int] = set()
        for uid in ordered_candidates:
            if cfg.max_users is not None and len(selected_users) >= cfg.max_users:
                break
            u_pos: Set[int] = set()
            for s in user_sessions_raw[uid]:
                u_pos.update(s["clicked_item_ids"])
                u_pos.update(s["purchased_item_ids"])
            if len(required_pos_items | u_pos) > pos_budget and len(selected_users) >= 100:
                continue
            selected_users.add(uid)
            required_pos_items.update(u_pos)
            if len(required_pos_items) >= pos_budget:
                break
    elif cfg.tier == "core5":
        # Iterative bipartite k-core filtering on the final `train` split (`cur_t < val_time_threshold`):
        # Retains items with `train_pos_count >= core_item_k` and users with `>= core_user_k` positive sessions.
        all_tr_times = sorted(
            s["time_index"]
            for s_list in user_sessions_raw.values()
            for s in s_list
            if s["raw_split"] == "train"
        )
        if len(all_tr_times) >= 20 and cfg.val_ratio > 0:
            c5_cut = max(1, int(len(all_tr_times) * (1.0 - cfg.val_ratio)))
            pre_val_threshold: Optional[int] = all_tr_times[min(c5_cut, len(all_tr_times) - 1)]
            min_tr_t0 = all_tr_times[0]
        else:
            pre_val_threshold = None
            min_tr_t0 = 0

        def _is_c5_final_train(sess_obj: Mapping[str, Any]) -> bool:
            if sess_obj["raw_split"] != "train":
                return False
            if pre_val_threshold is None:
                return True
            ct = int(sess_obj["time_index"])
            if ct >= pre_val_threshold:
                if ct > min_tr_t0:
                    return False
                return (stable_hash_int(f"val_{sess_obj['session_id']}_{cfg.seed}") % 100) >= int(cfg.val_ratio * 100)
            return True

        active_items: Set[int] = set(all_items.keys())
        active_users: Set[int] = set(user_sessions_raw.keys())
        peel_round = 0
        while True:
            peel_round += 1
            next_users: Set[int] = set()
            item_tr_cnt: Counter = Counter()
            for uid in active_users:
                u_valid_tr = 0
                u_pos_per_sess: List[Set[int]] = []
                for s in user_sessions_raw[uid]:
                    if _is_c5_final_train(s):
                        p_in = {iid for iid in (s["clicked_item_ids"] + s["purchased_item_ids"]) if iid in active_items}
                        if p_in:
                            u_valid_tr += 1
                            u_pos_per_sess.append(p_in)
                if u_valid_tr >= cfg.core_user_k:
                    next_users.add(uid)
                    for p_in in u_pos_per_sess:
                        for iid in p_in:
                            item_tr_cnt[iid] += 1

            next_items = {iid for iid, cnt in item_tr_cnt.items() if cnt >= cfg.core_item_k}
            if cfg.target_corpus_size > 0 and len(next_items) > cfg.target_corpus_size:
                top_iids = [iid for iid, cnt in item_tr_cnt.most_common(cfg.target_corpus_size) if cnt >= cfg.core_item_k]
                next_items = set(top_iids)

            if next_items == active_items and next_users == active_users:
                break
            active_items = next_items
            active_users = next_users

        print(
            f"[Builder:core5] Iterative ({cfg.core_user_k},{cfg.core_item_k})-core converged in {peel_round} rounds: "
            f"{len(active_users):,} users, {len(active_items):,} items."
        )

        filtered_user_sessions: Dict[int, List[Dict[str, Any]]] = {}
        for uid in sorted(active_users):
            kept_s: List[Dict[str, Any]] = []
            for s in user_sessions_raw[uid]:
                c_f = [iid for iid in s["clicked_item_ids"] if iid in active_items]
                p_f = [iid for iid in s["purchased_item_ids"] if iid in active_items]
                if c_f or p_f:
                    ns = dict(s)
                    ns["clicked_item_ids"] = c_f
                    ns["purchased_item_ids"] = p_f
                    ns["impressed_item_ids"] = [iid for iid in s["impressed_item_ids"] if iid in active_items]
                    kept_s.append(ns)
            if kept_s:
                filtered_user_sessions[uid] = kept_s

        user_sessions_raw = filtered_user_sessions
        selected_users = set(filtered_user_sessions.keys())
        required_pos_items = set(active_items)
    else:
        # Scale tier: retain users up to target_corpus_size positive budget (or all users if within budget)
        ordered_users = sorted(
            user_sessions_raw.keys(),
            key=lambda u: (-len(user_sessions_raw[u]), stable_hash_int(f"scale_{u}_{cfg.seed}")),
        )
        pos_budget = int(cfg.target_corpus_size * 0.72)
        selected_users = set()
        required_pos_items = set()
        for uid in ordered_users:
            if cfg.max_users is not None and len(selected_users) >= cfg.max_users:
                break
            u_pos = set()
            for s in user_sessions_raw[uid]:
                u_pos.update(s["clicked_item_ids"])
                u_pos.update(s["purchased_item_ids"])
            if len(required_pos_items | u_pos) > pos_budget and len(selected_users) >= 1000:
                continue
            selected_users.add(uid)
            required_pos_items.update(u_pos)

    selected_session_ids = {
        s["session_id"] for uid in selected_users for s in user_sessions_raw[uid]
    }
    print(
        f"[Builder:{cfg.tier}] Selected {len(selected_users):,} users, "
        f"{len(selected_session_ids):,} sessions, {len(required_pos_items):,} positive items."
    )

    # Optional pre-session history from rank table
    rank_prehist: Dict[str, Tuple[List[int], List[int]]] = {}
    if cfg.use_rank_prehistory and rank_file is not None:
        rank_prehist = _load_rank_prehistory_for_sessions(rank_file, selected_session_ids)

    # 5. Fill Corpus up to `cfg.target_corpus_size` (for open-catalog `mini`/`scale` tiers;
    #    for `core5`, retain strictly the active 5-core warm item catalog with zero filler padding):
    corpus_set: Set[int] = set(required_pos_items)

    if cfg.tier != "core5":
        # (b) Pre-history items (up to 70% of target_corpus_size)
        hist_cap = int(cfg.target_corpus_size * 0.70)
        if rank_prehist and len(corpus_set) < hist_cap:
            hist_counter: Counter = Counter()
            for rc, rp in rank_prehist.values():
                for iid in rp + rc:
                    if iid in all_items and iid not in corpus_set:
                        hist_counter[iid] += 1
            for iid, _ in hist_counter.most_common(hist_cap - len(corpus_set)):
                corpus_set.add(iid)

        # (c) Impressed unclicked hard negatives from train sessions (up to 88% of target_corpus_size)
        hard_neg_cap = int(cfg.target_corpus_size * 0.88)
        if len(corpus_set) < hard_neg_cap:
            imp_counter: Counter = Counter()
            for uid in selected_users:
                for s in user_sessions_raw[uid]:
                    if s["raw_split"] != "train":
                        continue
                    pos_s = set(s["clicked_item_ids"]) | set(s["purchased_item_ids"])
                    for iid in s["impressed_item_ids"]:
                        if iid not in pos_s and iid not in corpus_set:
                            imp_counter[iid] += 1
            for iid, _ in imp_counter.most_common(hard_neg_cap - len(corpus_set)):
                corpus_set.add(iid)

        # (d) Stratified L1 category fill up to min(target_corpus_size, len(all_items))
        final_target = min(cfg.target_corpus_size, len(all_items))
        if len(corpus_set) < final_target:
            rng = np.random.default_rng(cfg.seed)
            c1_keys = sorted(cat1_buckets.keys())
            shuffled_buckets = []
            for c1 in c1_keys:
                cands = [iid for iid in cat1_buckets[c1] if iid not in corpus_set]
                if cands:
                    perm = rng.permutation(len(cands))
                    shuffled_buckets.append([cands[i] for i in perm])
            ptr = 0
            while len(corpus_set) < final_target and shuffled_buckets:
                non_empty = []
                for b in shuffled_buckets:
                    if ptr < len(b) and len(corpus_set) < final_target:
                        corpus_set.add(b[ptr])
                    if ptr + 1 < len(b):
                        non_empty.append(b)
                shuffled_buckets = non_empty
                ptr += 1

    # 6. Chronological Train / Val / Test Split Assignment
    # Native `test` stays `test`; latest `val_ratio` of native `train` becomes `val`
    if cfg.tier == "core5":
        val_time_threshold = pre_val_threshold
        min_native_tr_time = min_tr_t0
    else:
        native_train_times = sorted(
            s["time_index"]
            for uid in selected_users
            for s in user_sessions_raw[uid]
            if s["raw_split"] == "train"
        )
        if len(native_train_times) >= 20 and cfg.val_ratio > 0:
            val_cut_idx = max(1, int(len(native_train_times) * (1.0 - cfg.val_ratio)))
            val_time_threshold = native_train_times[min(val_cut_idx, len(native_train_times) - 1)]
            min_native_tr_time = native_train_times[0]
        else:
            val_time_threshold = None
            min_native_tr_time = 0

    # 7. Reconstruct Strictly Causal User Histories (`hist_time < cur_time`)
    all_built_sessions: List[Dict[str, Any]] = []
    for uid in sorted(selected_users):
        s_list = sorted(
            user_sessions_raw[uid],
            key=lambda s: (s["time_index"], stable_hash_int(s["session_id"])),
        )
        # Only compute `suffix_positives` if optional legacy `use_rank_prehistory` is explicitly enabled;
        # in default pure-causal mode (`use_rank_prehistory=False`), zero future state is ever inspected!
        suffix_positives: List[Set[int]] = []
        if cfg.use_rank_prehistory:
            suffix_positives = [set() for _ in range(len(s_list) + 1)]
            for idx in range(len(s_list) - 1, -1, -1):
                cur_pos = set(s_list[idx]["clicked_item_ids"]) | set(s_list[idx]["purchased_item_ids"])
                suffix_positives[idx] = suffix_positives[idx + 1] | cur_pos

        causal_queries: List[str] = []
        causal_query_times: List[int] = []
        causal_clicked: List[int] = []
        causal_clicked_times: List[int] = []
        causal_purchased: List[int] = []
        causal_purchased_times: List[int] = []
        past_positives_seen: Set[int] = set()

        # Group simultaneous timestamps so sessions at identical `time_index` NEVER see each other!
        i = 0
        while i < len(s_list):
            j = i
            cur_t = s_list[i]["time_index"]
            while j < len(s_list) and s_list[j]["time_index"] == cur_t:
                j += 1

            # Build all sessions in [i, j) using ONLY strictly earlier history (< cur_t)
            for k in range(i, j):
                s = s_list[k]
                sid_str = s["session_id"]

                if cfg.use_rank_prehistory:
                    forbidden_future = suffix_positives[i] - past_positives_seen
                    pre_c, pre_p = rank_prehist.get(sid_str, ([], []))
                    clean_pre_c = [iid for iid in pre_c if iid in corpus_set and iid not in forbidden_future]
                    clean_pre_p = [iid for iid in pre_p if iid in corpus_set and iid not in forbidden_future]
                    pre_c_times = [-1] * len(clean_pre_c)
                    pre_p_times = [-1] * len(clean_pre_p)
                else:
                    clean_pre_c, clean_pre_p = [], []
                    pre_c_times, pre_p_times = [], []

                # Merge causal in-window clicked items (deduplicated keeping latest timestamp < cur_t)
                merged_clicks_rev: List[int] = []
                merged_click_times_rev: List[int] = []
                seen_h: Set[int] = set()
                for iid, itime in zip(reversed(clean_pre_c + causal_clicked), reversed(pre_c_times + causal_clicked_times)):
                    if iid not in seen_h:
                        seen_h.add(iid)
                        merged_clicks_rev.append(iid)
                        merged_click_times_rev.append(itime)
                    if len(merged_clicks_rev) >= cfg.max_hist_items:
                        break
                hist_clicks = list(reversed(merged_clicks_rev))
                hist_click_times = list(reversed(merged_click_times_rev))

                merged_purch_rev: List[int] = []
                merged_purch_times_rev: List[int] = []
                seen_hp: Set[int] = set()
                for iid, itime in zip(reversed(clean_pre_p + causal_purchased), reversed(pre_p_times + causal_purchased_times)):
                    if iid not in seen_hp:
                        seen_hp.add(iid)
                        merged_purch_rev.append(iid)
                        merged_purch_times_rev.append(itime)
                    if len(merged_purch_rev) >= cfg.max_hist_items:
                        break
                hist_purch = list(reversed(merged_purch_rev))
                hist_purch_times = list(reversed(merged_purch_times_rev))

                hist_q = causal_queries[-cfg.max_hist_queries :]
                hist_q_times = causal_query_times[-cfg.max_hist_queries :]

                # Determine final split
                if s["raw_split"] == "test":
                    final_split = "test"
                elif val_time_threshold is not None and cur_t >= val_time_threshold:
                    if cur_t > min_native_tr_time:
                        final_split = "val"
                    else:
                        final_split = (
                            "val"
                            if (stable_hash_int(f"val_{sid_str}_{cfg.seed}") % 100) < int(cfg.val_ratio * 100)
                            else "train"
                        )
                else:
                    final_split = "train"

                imp_in_corpus = [iid for iid in s["impressed_item_ids"] if iid in corpus_set]
                u_prof = users_map.get(uid, {"user_id": uid, "gender": "", "age_bucket": "", "province": "", "city": ""})

                all_built_sessions.append(
                    {
                        "session_id": sid_str,
                        "user_id": uid,
                        "time_index": cur_t,
                        "split": final_split,
                        "query": s["query"],
                        "user_profile": u_prof,
                        "hist_queries": hist_q,
                        "hist_query_times": hist_q_times,
                        "hist_clicked_item_ids": hist_clicks,
                        "hist_clicked_times": hist_click_times,
                        "hist_purchased_item_ids": hist_purch,
                        "hist_purchased_times": hist_purch_times,
                        "clicked_item_ids": s["clicked_item_ids"],
                        "purchased_item_ids": s["purchased_item_ids"],
                        "impressed_item_ids": imp_in_corpus,
                    }
                )

            # Now advance causal state past `cur_t`
            for k in range(i, j):
                s = s_list[k]
                causal_queries.append(s["query"])
                causal_query_times.append(cur_t)
                for iid in s["clicked_item_ids"]:
                    if iid in corpus_set:
                        causal_clicked.append(iid)
                        causal_clicked_times.append(cur_t)
                        past_positives_seen.add(iid)
                for iid in s["purchased_item_ids"]:
                    if iid in corpus_set:
                        causal_purchased.append(iid)
                        causal_purchased_times.append(cur_t)
                        past_positives_seen.add(iid)
            i = j

    # 8. Compute Training-Split Item & Query Statistics
    train_query_freq: Counter = Counter()
    train_query_l2_cats: Dict[str, Counter] = defaultdict(Counter)
    train_item_show: Counter = Counter()
    train_item_click: Counter = Counter()
    train_item_order: Counter = Counter()
    train_item_pos: Counter = Counter()
    item_co_queries: Dict[int, Counter] = defaultdict(Counter)

    for sess in all_built_sessions:
        if sess["split"] != "train":
            continue
        q = sess["query"]
        train_query_freq[q] += 1
        c_set = set(sess["clicked_item_ids"])
        p_set = set(sess["purchased_item_ids"])
        pos_set = c_set | p_set

        for iid in sess["impressed_item_ids"]:
            train_item_show[iid] += 1
        for iid in c_set:
            train_item_click[iid] += 1
        for iid in p_set:
            train_item_order[iid] += 1
        for iid in pos_set:
            train_item_pos[iid] += 1
            item_co_queries[iid][q] += 1
            c2 = all_items[iid]["category_level2_id"]
            if c2 > 0:
                train_query_l2_cats[q][c2] += 1

    seen_freqs = [cnt for cnt in train_query_freq.values() if cnt > 0]
    head_freq_thresh = max(3, int(np.percentile(seen_freqs, 75))) if seen_freqs else 3

    # 9. Assign Four Search-Specific Diagnostic Slices to Every Session
    slice_counts: Dict[str, Counter] = defaultdict(Counter)
    for sess in all_built_sessions:
        q = sess["query"]
        pos_ids = set(sess["clicked_item_ids"]) | set(sess["purchased_item_ids"])
        q_freq = train_query_freq.get(q, 0)
        q_ent = _compute_normalized_entropy(train_query_l2_cats[q]) if q in train_query_l2_cats else 0.0
        hist_richness = len(sess["hist_clicked_item_ids"]) + len(sess["hist_queries"])

        # Explicit brand match check
        has_brand_match = any(
            all_items[iid]["brand_name"] and all_items[iid]["brand_name"].lower() in q.lower()
            for iid in pos_ids
        )

        slices: List[str] = []
        if any(train_item_pos.get(iid, 0) == 0 for iid in pos_ids):
            slices.append("cold_start_item")
        if q_freq >= head_freq_thresh and (q_ent <= 0.6 or has_brand_match):
            slices.append("head_exact")
        if hist_richness >= 3 and (q_ent > 0.6 or len(train_query_l2_cats.get(q, {})) >= 2):
            slices.append("personalized_ambiguous")
        if q_freq <= 1 or len(q) >= 10:
            slices.append("long_tail_complex")

        sess["slices"] = slices
        for sl in slices:
            slice_counts[sess["split"]][sl] += 1

    # 10. Write Standardized Corpus, Train/Val/Test Sessions, and Relevance Split
    sorted_corpus_ids = sorted(corpus_set)
    corpus_records: List[Dict[str, Any]] = []
    for dense_idx, iid in enumerate(sorted_corpus_ids):
        base = dict(all_items[iid])
        shows = train_item_show.get(iid, 0)
        clicks = train_item_click.get(iid, 0)
        orders = train_item_order.get(iid, 0)
        # Ensure shows >= clicks when impressions list was truncated
        eff_shows = max(shows, clicks)
        posterior_ctr = float((clicks + 1.0) / (eff_shows + 25.0))
        top_queries = [q for q, _ in item_co_queries[iid].most_common(5)]

        base["corpus_idx"] = dense_idx
        base["structured_text"] = format_item_structured_text(base)
        base["train_pos_count"] = int(train_item_pos.get(iid, 0))
        base["posterior_ctr"] = round(posterior_ctr, 6)
        base["train_top_queries"] = top_queries
        corpus_records.append(base)

    write_jsonl(out_path / "corpus.jsonl", corpus_records)

    split_buckets: Dict[str, List[Dict[str, Any]]] = {"train": [], "val": [], "test": []}
    for sess in all_built_sessions:
        split_buckets[sess["split"]].append(sess)

    for sp_name, recs in split_buckets.items():
        write_jsonl(out_path / f"{sp_name}_sessions.jsonl", recs)

    # Also build grouped deterministic Train/Val/Test split for 4-Tier Relevance table (if present)
    rel_stats: Dict[str, int] = {}
    if rel_file and rel_file.is_file():
        rel_splits: Dict[str, List[Dict[str, Any]]] = {"train": [], "val": [], "test": []}
        for r_obj in iter_jsonl(rel_file):
            q_text = clean_field(r_obj.get("query"))
            i_title = clean_field(r_obj.get("item_title") or r_obj.get("title"))
            score = r_obj.get("score")
            if not q_text or not i_title or score is None:
                continue
            rec_clean = {
                "query": q_text,
                "item_title": i_title,
                "brand_name": clean_field(r_obj.get("brand") or r_obj.get("brand_name")),
                "seller_name": clean_field(r_obj.get("seller_name")),
                "attr_value": clean_field(r_obj.get("attr_value") or r_obj.get("attribute")),
                "score": int(score),
            }
            # Group by query hash so val/test evaluate unseen-query relevance generalization
            bucket_id = stable_hash_int(f"rel_q_{q_text}_{cfg.seed}") % 100
            if bucket_id < 5:
                rel_splits["test"].append(rec_clean)
            elif bucket_id < 10:
                rel_splits["val"].append(rec_clean)
            else:
                rel_splits["train"].append(rec_clean)

        rel_out_dir = out_path / "relevance"
        for rsp, rlist in rel_splits.items():
            write_jsonl(rel_out_dir / f"{rsp}.jsonl", rlist)
            rel_stats[rsp] = len(rlist)

    pos_counts_arr = [int(r["train_pos_count"]) for r in corpus_records]
    stats = {
        "tier": cfg.tier,
        "config": asdict(cfg),
        "num_corpus_items": len(corpus_records),
        "num_selected_users": len(selected_users),
        "users_per_split": {k: len({int(s["user_id"]) for s in v}) for k, v in split_buckets.items()},
        "sessions_per_split": {k: len(v) for k, v in split_buckets.items()},
        "corpus_train_pos_summary": {
            "min": int(min(pos_counts_arr)) if pos_counts_arr else 0,
            "median": float(np.median(pos_counts_arr)) if pos_counts_arr else 0.0,
            "max": int(max(pos_counts_arr)) if pos_counts_arr else 0,
            "ge5_items": int(sum(1 for c in pos_counts_arr if c >= 5)),
            "zero_pos_items": int(sum(1 for c in pos_counts_arr if c == 0)),
        },
        "slice_counts": {sp: dict(cnts) for sp, cnts in slice_counts.items()},
        "relevance_splits": rel_stats,
        "head_query_freq_threshold": head_freq_thresh,
    }
    (out_path / "stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # 11. Verify Benchmark Schema & Timestamp Ordering
    verify_benchmark_integrity(out_path)
    print(f"[Builder:{cfg.tier}] Completed & verified -> {out_path}")
    return stats


def verify_benchmark_integrity(bench_dir: str | Path) -> Dict[str, Any]:
    """Verify corpus completeness and chronological timestamp ordering for a built benchmark."""
    bpath = Path(bench_dir)
    corpus_ids: Set[int] = set()
    train_pos_from_corpus: Dict[int, int] = {}
    for item in iter_jsonl(bpath / "corpus.jsonl"):
        iid = int(item["item_id"])
        corpus_ids.add(iid)
        train_pos_from_corpus[iid] = int(item["train_pos_count"])

    observed_train_pos: Counter = Counter()
    for sp in ("train", "val", "test"):
        sfile = bpath / f"{sp}_sessions.jsonl"
        if not sfile.is_file():
            continue
        for sess in iter_jsonl(sfile):
            cur_t = int(sess["time_index"])
            for ht in sess.get("hist_query_times", []):
                if int(ht) >= cur_t:
                    raise AssertionError(
                        f"Temporal leakage detected in session {sess['session_id']}: hist_query_time={ht} >= cur_time={cur_t}"
                    )
            for ht in sess.get("hist_clicked_times", []) + sess.get("hist_purchased_times", []):
                if int(ht) >= cur_t:
                    raise AssertionError(
                        f"Temporal leakage detected in session {sess['session_id']}: hist_item_time={ht} >= cur_time={cur_t}"
                    )
            pos_ids = {int(x) for x in sess["clicked_item_ids"]} | {int(x) for x in sess["purchased_item_ids"]}
            if not pos_ids:
                raise AssertionError(f"Session {sess['session_id']} has empty positive targets.")
            missing = pos_ids - corpus_ids
            if missing:
                raise AssertionError(f"Session {sess['session_id']} targets {missing} missing from corpus!")
            for hiid in sess.get("hist_clicked_item_ids", []) + sess.get("hist_purchased_item_ids", []):
                if int(hiid) not in corpus_ids:
                    raise AssertionError(f"History item {hiid} not in corpus!")
            if sp == "train":
                for iid in pos_ids:
                    observed_train_pos[iid] += 1
            if "cold_start_item" in sess.get("slices", []):
                if not any(train_pos_from_corpus.get(iid, 0) == 0 for iid in pos_ids):
                    raise AssertionError(f"Invalid cold_start_item slice tag on session {sess['session_id']}")

    for iid, cnt in train_pos_from_corpus.items():
        if observed_train_pos.get(iid, 0) != cnt:
            raise AssertionError(
                f"Train pos count mismatch on item {iid}: recorded={cnt} vs observed={observed_train_pos.get(iid, 0)}"
            )

    stats_f = bpath / "stats.json"
    if stats_f.is_file():
        st_meta = json.loads(stats_f.read_text(encoding="utf-8"))
        if st_meta.get("tier") == "core5":
            req_k = int(st_meta.get("config", {}).get("core_item_k", 5))
            violating = [iid for iid, cnt in train_pos_from_corpus.items() if cnt < req_k]
            if violating:
                raise AssertionError(
                    f"Core5 invariant violated! {len(violating)} items have train_pos_count < {req_k} (e.g. {violating[:5]})"
                )

    return {"verified": True, "corpus_size": len(corpus_ids)}


def main() -> None:
    p = argparse.ArgumentParser(description="Build chronological KuaiSearch benchmark tiers for MiniOneSearch")
    p.add_argument("--raw_dir", required=True, help="Directory containing raw KuaiSearch tables")
    p.add_argument("--out_dir", required=True, help="Output directory for built benchmark")
    p.add_argument("--download", action="store_true", help="Download raw KuaiSearch tables from HuggingFace first")
    p.add_argument("--tier", choices=["mini", "core5", "scale"], default="mini")
    p.add_argument("--target_corpus_size", type=int, default=0, help="Override target corpus size (default: 100K mini / 0=all core5 / 500K scale)")
    p.add_argument("--max_users", type=int, default=-1, help="Override max users (-1 uses tier default)")
    p.add_argument("--core_item_k", type=int, default=5, help="Minimum train positive sessions per item in core5")
    p.add_argument("--core_user_k", type=int, default=1, help="Minimum train positive sessions per user in core5")
    p.add_argument("--enable_rank_prehistory", action="store_true", help="Opt-in legacy rank_lite prehistory (default: pure recall_lite timestamps)")
    p.add_argument("--skip_rank_prehistory", action="store_true", help="Default pure timestamped recall_lite mode (kept for backward CLI compatibility)")
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()

    include_rank = bool(a.enable_rank_prehistory and not a.skip_rank_prehistory)
    if a.download:
        download_kuaisearch_raw(a.raw_dir, include_rank_lite=include_rank)

    default_corpus = 100_000 if a.tier == "mini" else (0 if a.tier == "core5" else 500_000)
    default_users = 16_000 if a.tier == "mini" else None
    cfg = BuildConfig(
        tier=a.tier,
        target_corpus_size=a.target_corpus_size if a.target_corpus_size > 0 else default_corpus,
        max_users=a.max_users if a.max_users > 0 else default_users,
        core_item_k=a.core_item_k,
        core_user_k=a.core_user_k,
        seed=a.seed,
        use_rank_prehistory=include_rank,
    )
    stats = build_benchmark_from_raw(a.raw_dir, a.out_dir, cfg)
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
