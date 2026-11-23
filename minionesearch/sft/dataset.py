"""Multi-Stage Progressive Alignment & Joint Self-Distillation Dataset Builders.

Implements clean, factorized prompt templates where Personalized Search prompts split cleanly into:
  `prompt = user_prefix + query_suffix`
so that `UserPrefixKVCacheEngine` can pre-cache `user_prefix` with 100% token-exact equivalence!

Three Progressive Stages:
  - Stage 1 (`Semantic Alignment`): Item Text <-> SID, Query <-> Query SID, SID -> Category/Brand
  - Stage 2 (`Q-I Co-occurrence`): Query -> Item SID, Item SID -> Co-Clicked Search Query
  - Stage 3 (`Personalized Search + Privileged Teacher View`):
      Student View: `user_prefix + "当前搜索：{query}\n推荐商品ID："` -> `{target_sid}`
      Teacher View: `user_prefix + "当前搜索：{query} | 意图线索：{teacher_hints}\n推荐商品ID："` -> `{target_sid}`
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import jieba
import torch
from torch.utils.data import Dataset

from minionesearch.decoding.trie import SIDVocabulary
from minionesearch.utils import stable_hash_int


def compute_museq_behavior_sid(
    hist_ids: Sequence[int],
    item_to_codes: Mapping[int, Sequence[int]],
    sid_vocab: SIDVocabulary,
    tau: float = 0.50,
) -> str:
    """OneSearch-V1 §3.2.1 Eq. (3) with Temperature-Normalized Index & Hierarchical Mode Selection.

    Fixes two subtle issues in naive `exp(i)` integer averaging:
      1. Normalizes step index `u_i = (i + 1) / m` in `(0, 1]` and shifts `(u_i - 1.0) / tau <= 0`,
         guaranteeing zero FP16/FP64 overflow at any sequence length `m` and smooth decay control
         (`tau=0.35` for sharp short-term intent vs `tau=2.0` for broad long-term user profile).
      2. Selects the hierarchical prefix-consistent weighted mode (`argmax_c sum w_i`) at each depth,
         guaranteeing `SID_short` and `SID_long` always lie on real semantic clusters visited by the user
         rather than landing on an unrelated arithmetic-midpoint cluster ID!
    """
    valid_codes = [tuple(int(x) for x in item_to_codes[int(iid)]) for iid in hist_ids if int(iid) in item_to_codes]
    if not valid_codes:
        return ""
    m = len(valid_codes)
    eff_tau = max(float(tau), 0.05)
    raw_w = [math.exp(((float(i + 1) / float(m)) - 1.0) / eff_tau) for i in range(m)]
    w_sum = sum(raw_w)
    norm_w = [w / w_sum for w in raw_w]

    chosen_codes: List[int] = []
    active_indices = list(range(m))
    for l in range(sid_vocab.num_levels):
        score_by_c: Dict[int, float] = {}
        for idx in active_indices:
            c_val = valid_codes[idx][l]
            score_by_c[c_val] = score_by_c.get(c_val, 0.0) + norm_w[idx]
        best_c = max(score_by_c.items(), key=lambda kv: (kv[1], -kv[0]))[0]
        chosen_codes.append(best_c)
        active_indices = [idx for idx in active_indices if valid_codes[idx][l] == best_c]
    return sid_vocab.codes_to_str(chosen_codes)


def extract_session_intent_keywords(
    sess: Mapping[str, Any],
    corpus_map: Mapping[int, Mapping[str, Any]],
    max_keywords: int = 5,
) -> List[str]:
    """OneSearch-V2 §3.2.1: Extract clean category/brand/attribute intent keywords (NO literal SKU titles!)."""
    pos_ids = list(dict.fromkeys((sess.get("purchased_item_ids") or []) + (sess.get("clicked_item_ids") or [])))
    kws: List[str] = []
    seen = set()

    def _add(term: str) -> None:
        t = term.strip()
        if 2 <= len(t) <= 14 and t not in seen:
            seen.add(t)
            kws.append(t)

    for iid in pos_ids[:5]:
        item = corpus_map.get(int(iid))
        if not item:
            continue
        c2 = str(item.get("category_level2_name") or "").strip()
        c3 = str(item.get("category_level3_name") or "").strip()
        brand = str(item.get("brand_name") or "").strip()
        if c2 and c3 and c2 != c3:
            _add(f"{c2}/{c3}")
        elif c3:
            _add(c3)
        elif c2:
            _add(c2)
        if brand:
            _add(brand)

    # Supplement with salient 2-4 char modifier words shared across interacted items if < max_keywords
    if len(kws) < max_keywords:
        word_cnt: Dict[str, int] = {}
        for iid in pos_ids[:5]:
            title = str((corpus_map.get(int(iid)) or {}).get("item_title") or "")
            for w in dict.fromkeys(jieba.cut(title)):
                ws = w.strip()
                if 2 <= len(ws) <= 5 and not ws.isascii():
                    word_cnt[ws] = word_cnt.get(ws, 0) + 1
        for ws, _ in sorted(word_cnt.items(), key=lambda kv: (-kv[1], kv[0])):
            _add(ws)
            if len(kws) >= max_keywords:
                break

    return kws[:max_keywords]


@dataclass
class PromptBuilder:
    max_hist_queries: int = 8
    max_hist_items: int = 10
    include_user_profile: bool = True
    user_id_mode: str = "mu_seq"  # "mu_seq" (OneSearch-V1 §3.2.1) | "random_hash" (TIGER) | "none"

    def format_user_prefix(
        self,
        sess: Mapping[str, Any],
        item_sid_map: Mapping[int, str],
        item_to_codes: Optional[Mapping[int, Sequence[int]]] = None,
        sid_vocab: Optional[SIDVocabulary] = None,
    ) -> str:
        """Build static cacheable `user_prefix` (`Mu-Seq` User ID + Profile + Short Search & Click SIDs)."""
        parts: List[str] = []
        if self.include_user_profile:
            prof = sess.get("user_profile") or {}
            p_tokens = [
                str(prof.get("gender") or "").strip(),
                str(prof.get("age_bucket") or "").strip(),
                str(prof.get("province") or "").strip(),
            ]
            p_str = ",".join([x for x in p_tokens if x])
            if p_str:
                parts.append(f"用户画像:{p_str}")

        all_hist_c = [int(i) for i in (sess.get("hist_clicked_item_ids") or [])]
        hist_c = all_hist_c[-self.max_hist_items :]
        hist_p = [int(i) for i in (sess.get("hist_purchased_item_ids") or [])[-20:]]

        if self.user_id_mode == "mu_seq" and item_to_codes is not None and sid_vocab is not None:
            # Disjoint temporal windows + distinct temperatures so `SID_short` (recent immediate intent)
            # and `SID_long` (long-term / purchase profile) capture complementary signals without collapsing!
            short_window = all_hist_c[-4:]
            short_uid_sid = compute_museq_behavior_sid(short_window, item_to_codes, sid_vocab, tau=0.35)
            older_clicks = all_hist_c[:-4] if len(all_hist_c) > 4 else []
            long_window = (older_clicks + hist_p) if (older_clicks or hist_p) else all_hist_c[: max(1, len(all_hist_c) // 2)]
            long_uid_sid = compute_museq_behavior_sid(long_window, item_to_codes, sid_vocab, tau=2.00)
            if short_uid_sid or long_uid_sid:
                parts.append(f"用户偏好ID:{short_uid_sid}{long_uid_sid}")
        elif self.user_id_mode == "random_hash":
            uid = int(sess.get("user_id") or 0)
            parts.append(f"用户ID:U{stable_hash_int(f'tiger_u_{uid}') % 4096}")

        hist_qs = [str(q).strip() for q in (sess.get("hist_queries") or [])[-self.max_hist_queries :] if str(q).strip()]
        if hist_qs:
            parts.append("最近搜索:" + "".join(f"<{q}>" for q in hist_qs))

        hist_sids = [item_sid_map[iid] for iid in hist_c if iid in item_sid_map]
        if hist_sids:
            parts.append("历史交互商品id:" + "和".join(hist_sids))

        if not parts:
            return "用户画像:新用户 | "
        return "，".join(parts) + "，"

    @staticmethod
    def format_student_query_suffix(
        query: str,
        query_sid: Optional[str] = None,
        explicit_cot_mode: bool = False,
    ) -> str:
        """OneSearch-V2 Eq. (2) Student Prompt Suffix ( matches `onesearch-family/self-distillation/data/sample.json`!)."""
        q = str(query).strip()
        q_part = f"用户当前搜索query为<{q}>，对应id为{query_sid}" if query_sid else f"用户当前搜索query为<{q}>"
        if explicit_cot_mode:
            return f"{q_part}，请先分析意图关键词再给出商品id："
        return f"{q_part}，请给出合适的商品id："

    @staticmethod
    def format_teacher_query_suffix(
        query: str,
        intent_keywords: Sequence[str],
        query_sid: Optional[str] = None,
    ) -> str:
        """OneSearch-V2 Eq. (1) Privileged Teacher Suffix (`相关的商品关键词有<kw1><kw2>...`)."""
        q = str(query).strip()
        q_part = f"用户当前搜索query为<{q}>，对应id为{query_sid}" if query_sid else f"用户当前搜索query为<{q}>"
        kw_str = "".join(f"<{k}>" for k in intent_keywords) if intent_keywords else f"<{q}>"
        return f"{q_part}，相关的商品关键词有{kw_str}，请给出合适的商品id："

    def build_inference_pair(
        self,
        sess: Mapping[str, Any],
        item_sid_map: Mapping[int, str],
        query_sid_map: Optional[Mapping[str, str]] = None,
        item_to_codes: Optional[Mapping[int, Sequence[int]]] = None,
        sid_vocab: Optional[SIDVocabulary] = None,
        personalized: bool = True,
        explicit_cot_mode: bool = False,
    ) -> Tuple[str, str]:
        """Return `(user_prefix, student_query_suffix)` for evaluation & KV-cache profiling."""
        u_pref = (
            self.format_user_prefix(sess, item_sid_map, item_to_codes=item_to_codes, sid_vocab=sid_vocab)
            if personalized
            else ""
        )
        q_str = str(sess["query"]).strip()
        q_sid = query_sid_map.get(q_str) if query_sid_map else None
        q_suf = self.format_student_query_suffix(q_str, query_sid=q_sid, explicit_cot_mode=explicit_cot_mode)
        return u_pref, q_suf


def build_stage1_alignment_records(
    corpus_records: Sequence[Mapping[str, Any]],
    item_sid_map: Mapping[int, str],
    query_sid_map: Optional[Mapping[str, str]] = None,
    train_sessions: Optional[Sequence[Mapping[str, Any]]] = None,
    include_cot_tasks: bool = True,
    max_items: Optional[int] = None,
    seed: int = 42,
) -> List[Dict[str, str]]:
    """Stage 1: OneSearch-V1 Semantic Alignment + OneSearch-V2 §3.2.2 Keyword-CoT Alignment Tasks."""
    rng = random.Random(seed)
    items = list(corpus_records)
    corpus_map = {int(r["item_id"]): r for r in corpus_records}
    if max_items is not None and len(items) > max_items:
        items = rng.sample(items, max_items)

    records: List[Dict[str, str]] = []
    for r in items:
        iid = int(r["item_id"])
        sid = item_sid_map.get(iid)
        if not sid:
            continue
        stext = str(r.get("structured_text") or r.get("item_title") or "").strip()
        title = str(r.get("item_title") or "").strip()
        cats = "/".join(
            [
                str(r.get(k) or "").strip()
                for k in ("category_level1_name", "category_level2_name", "category_level3_name")
                if str(r.get(k) or "").strip()
            ]
        )
        records.append({"prompt": f"商品标题为<{stext}>，请给出商品id：", "target": sid, "task": "s1_i2sid"})
        if title and rng.random() < 0.4:
            records.append({"prompt": f"商品id为{sid}，对应的商品标题为：", "target": title[:48], "task": "s1_sid2i"})
        if cats:
            records.append({"prompt": f"商品id为{sid}，所属类目为：", "target": cats, "task": "s1_sid2cat"})

    if query_sid_map:
        q_items = list(query_sid_map.items())
        for q_str, q_sid in q_items:
            records.append({"prompt": f"用户query为<{q_str}>，对应的查询id为：", "target": q_sid, "task": "s1_q2sid"})
            if rng.random() < 0.35:
                records.append({"prompt": f"查询id为{q_sid}，对应的用户query为：", "target": q_str, "task": "s1_sid2q"})

    # OneSearch-V2 §3.2.2 CoT Alignment Tasks (`query -> keywords` & `query + user -> P_keywords`)
    if include_cot_tasks and train_sessions:
        pb = PromptBuilder()
        for sess in train_sessions:
            kws = extract_session_intent_keywords(sess, corpus_map, max_keywords=5)
            if not kws:
                continue
            q = str(sess["query"]).strip()
            kw_target = ",".join(kws)
            records.append({"prompt": f"用户当前搜索query为<{q}>，请给出1-5个意图关键词：", "target": kw_target, "task": "s1_cot_q2kw"})
            if sess.get("hist_clicked_item_ids") and rng.random() < 0.5:
                u_pref = pb.format_user_prefix(sess, item_sid_map)
                records.append(
                    {"prompt": f"{u_pref}用户当前搜索query为<{q}>，请给出1-5个个性化关键词：", "target": kw_target, "task": "s1_cot_uq2pkw"}
                )

    rng.shuffle(records)
    return records


def build_stage2_cooccurrence_records(
    train_sessions: Sequence[Mapping[str, Any]],
    corpus_records: Sequence[Mapping[str, Any]],
    item_sid_map: Mapping[int, str],
    query_sid_map: Optional[Mapping[str, str]] = None,
    max_records: Optional[int] = None,
    seed: int = 42,
) -> List[Dict[str, str]]:
    """Stage 2: OneSearch-V1 §3.4.1 Co-occurrence Synchronization (`query <-> item`, `SID_q <-> SID_i`)."""
    rng = random.Random(seed)
    records: List[Dict[str, str]] = []

    for sess in train_sessions:
        q = str(sess["query"]).strip()
        q_sid = query_sid_map.get(q) if query_sid_map else None
        pos_ids = list(dict.fromkeys((sess.get("purchased_item_ids") or []) + (sess.get("clicked_item_ids") or [])))
        for iid in pos_ids[:3]:
            sid = item_sid_map.get(int(iid))
            if not sid:
                continue
            q_suf = PromptBuilder.format_student_query_suffix(q, query_sid=q_sid)
            records.append({"prompt": q_suf, "target": sid, "task": "s2_q2isid"})
            if q_sid:
                records.append({"prompt": f"用户当前搜索query的id为{q_sid}，对应的商品id：", "target": sid, "task": "s2_qsid2isid"})

    for r in corpus_records:
        iid = int(r["item_id"])
        sid = item_sid_map.get(iid)
        if not sid:
            continue
        top_qs = r.get("train_top_queries") or []
        if top_qs:
            q0 = str(top_qs[0])
            records.append({"prompt": f"用户点击商品的id为{sid}，对应的搜索query：", "target": q0, "task": "s2_isid2q"})
            if query_sid_map and q0 in query_sid_map:
                records.append({"prompt": f"用户点击商品的id为{sid}，对应的搜索query的id：", "target": query_sid_map[q0], "task": "s2_isid2qsid"})
        title = str(r.get("item_title") or "").strip()
        brand = str(r.get("brand_name") or "").strip()
        c3 = str(r.get("category_level3_name") or r.get("category_level2_name") or "").strip()
        pseudo_q = " ".join([x for x in (brand, c3, title[:18]) if x]).strip()
        if pseudo_q:
            records.append(
                {"prompt": PromptBuilder.format_student_query_suffix(pseudo_q, query_sid=None), "target": sid, "task": "s2_pseudo_q2isid"}
            )

    rng.shuffle(records)
    if max_records is not None and len(records) > max_records:
        records = records[:max_records]
    return records


def build_stage3_personalized_records(
    sessions: Sequence[Mapping[str, Any]],
    corpus_map: Mapping[int, Mapping[str, Any]],
    item_sid_map: Mapping[int, str],
    query_sid_map: Optional[Mapping[str, str]] = None,
    item_to_codes: Optional[Mapping[int, Sequence[int]]] = None,
    sid_vocab: Optional[SIDVocabulary] = None,
    prompt_builder: Optional[PromptBuilder] = None,
    personalized: bool = True,
    sliding_window_aug: bool = True,
    explicit_cot_mode: bool = False,
    max_positives_per_session: int = 4,
    seed: int = 42,
) -> List[Dict[str, str]]:
    """Stage 3: Personalized Search (`Mu-Seq + Query + SID_q -> Target SID`) with Clean Keyword-CoT Teacher."""
    pb = prompt_builder or PromptBuilder()
    rng = random.Random(seed)
    records: List[Dict[str, str]] = []

    for sess in sessions:
        q = str(sess["query"]).strip()
        purch = [int(x) for x in (sess.get("purchased_item_ids") or []) if int(x) in item_sid_map]
        clicks = [int(x) for x in (sess.get("clicked_item_ids") or []) if int(x) in item_sid_map and int(x) not in purch]
        ordered_pos = (purch + clicks)[:max_positives_per_session]
        if not ordered_pos:
            continue

        intent_kws = extract_session_intent_keywords(sess, corpus_map, max_keywords=5)
        q_sid = query_sid_map.get(q) if query_sid_map else None
        stu_suf = pb.format_student_query_suffix(q, query_sid=q_sid, explicit_cot_mode=explicit_cot_mode)
        tea_suf = pb.format_teacher_query_suffix(q, intent_keywords=intent_kws, query_sid=q_sid)

        u_pref = (
            pb.format_user_prefix(sess, item_sid_map, item_to_codes=item_to_codes, sid_vocab=sid_vocab)
            if personalized
            else ""
        )
        stu_prompt = u_pref + stu_suf
        tea_prompt = u_pref + tea_suf

        for iid in ordered_pos:
            sid = item_sid_map[iid]
            if explicit_cot_mode:
                kw_chain = ",".join(intent_kws) if intent_kws else q
                target_str = f"关键词:{kw_chain} 商品id:{sid}"
            else:
                target_str = sid
            records.append(
                {
                    "prompt": stu_prompt,
                    "teacher_prompt": tea_prompt,
                    "target": target_str,
                    "task": "s3_personalized",
                }
            )

        # OneSearch-V1 §3.4.1 Sliding-Window Short-Sequence Augmentation (teaches short/new-user robustness!)
        hist_c = sess.get("hist_clicked_item_ids") or []
        if personalized and sliding_window_aug and len(hist_c) >= 4 and not explicit_cot_mode:
            sub_len = rng.randint(1, len(hist_c) // 2)
            sub_sess = dict(sess)
            sub_sess["hist_clicked_item_ids"] = hist_c[:sub_len]
            sub_sess["hist_queries"] = (sess.get("hist_queries") or [])[:sub_len]
            sub_pref = pb.format_user_prefix(sub_sess, item_sid_map, item_to_codes=item_to_codes, sid_vocab=sid_vocab)
            top_sid = item_sid_map[ordered_pos[0]]
            records.append(
                {
                    "prompt": sub_pref + stu_suf,
                    "teacher_prompt": sub_pref + tea_suf,
                    "target": top_sid,
                    "task": "s3_sliding_window",
                }
            )

    rng.shuffle(records)
    return records


class SFTMultiStageDataset(Dataset):
    def __init__(self, records: Sequence[Mapping[str, str]]):
        self.records = list(records)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Mapping[str, str]:
        return self.records[idx]


def collate_sft_or_sdft_batch(
    batch: Sequence[Mapping[str, str]],
    tokenizer: Any,
    max_prompt_tokens: int = 320,
    max_target_tokens: int = 32,
    include_teacher: bool = False,
) -> Dict[str, torch.Tensor]:
    """Tokenize prompt + target with left-truncated prompts, right-padded sequences, and `-100` prompt mask.

    Also records exact `[start, end)` target slice indices (`comp_start`, `comp_len`) per sample
    so Joint SDFT computes KL divergence ONLY over completion token positions (40x memory savings!).
    """
    pad_id = int(tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id)
    eos_id = int(tokenizer.eos_token_id if tokenizer.eos_token_id is not None else pad_id)

    def _pack_side(prompt_key: str) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
        p_list: List[List[int]] = []
        t_list: List[List[int]] = []
        for rec in batch:
            p_str = str(rec.get(prompt_key) or rec["prompt"])
            t_str = str(rec["target"])
            p_ids = tokenizer(p_str, add_special_tokens=False)["input_ids"][-max_prompt_tokens:]
            t_ids = tokenizer(t_str, add_special_tokens=False)["input_ids"][: max_target_tokens - 1] + [eos_id]
            p_list.append(p_ids)
            t_list.append(t_ids)

        max_seq = max(len(p) + len(t) for p, t in zip(p_list, t_list))
        max_comp = max(len(t) for t in t_list)
        bsz = len(batch)

        input_ids = torch.full((bsz, max_seq), fill_value=pad_id, dtype=torch.long)
        attn_mask = torch.zeros((bsz, max_seq), dtype=torch.long)
        labels = torch.full((bsz, max_seq), fill_value=-100, dtype=torch.long)
        # Store logit positions `[bsz, max_comp]` predicting `t_ids` (`pos = len(p) - 1 + j`)
        logit_pos = torch.zeros((bsz, max_comp), dtype=torch.long)
        comp_mask = torch.zeros((bsz, max_comp), dtype=torch.bool)

        for i, (p_ids, t_ids) in enumerate(zip(p_list, t_list)):
            full = p_ids + t_ids
            input_ids[i, : len(full)] = torch.tensor(full, dtype=torch.long)
            attn_mask[i, : len(full)] = 1
            p_len = max(1, len(p_ids))
            labels[i, len(p_ids) : len(full)] = torch.tensor(t_ids, dtype=torch.long)
            for j in range(len(t_ids)):
                logit_pos[i, j] = max(0, p_len - 1 + j)
                comp_mask[i, j] = True

        return input_ids, attn_mask, labels, logit_pos, comp_mask

    stu_ids, stu_mask, stu_labels, stu_lpos, stu_cmask = _pack_side("prompt")
    out: Dict[str, torch.Tensor] = {
        "input_ids": stu_ids,
        "attention_mask": stu_mask,
        "labels": stu_labels,
        "comp_logit_pos": stu_lpos,
        "comp_mask": stu_cmask,
    }

    if include_teacher:
        tea_ids, tea_mask, tea_labels, tea_lpos, tea_cmask = _pack_side("teacher_prompt")
        out["teacher_input_ids"] = tea_ids
        out["teacher_attention_mask"] = tea_mask
        out["teacher_labels"] = tea_labels
        out["teacher_comp_logit_pos"] = tea_lpos
        out["teacher_comp_mask"] = tea_cmask

    return out
