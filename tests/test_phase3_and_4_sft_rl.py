"""Unit tests for Phase 3 (Vocab Init, Joint SDFT, User-Prefix KV-Cache) & Phase 4 (Search-RL & SP-TPMA)."""
from __future__ import annotations

import math
from typing import Dict, List

import pytest
import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

from minionesearch.decoding.fast_beam_bench import NativeFastConstrainedBeamEngine
from minionesearch.decoding.kv_cache_engine import UserPrefixKVCacheEngine, _clone_dynamic_cache
from minionesearch.decoding.trie import GPUPrefixTrie, SIDVocabulary
from minionesearch.evaluation.report_compiler import compile_full_report
from minionesearch.rl.algorithms import (
    compute_listwise_dpo_loss,
    compute_sp_tpma_advantages,
    compute_tpma_original_advantages,
    compute_vanilla_grpo_advantages,
    ppo_clipped_kl_surrogate_loss,
)
from minionesearch.rl.rewards import SearchRewardEngine
from minionesearch.rl.rl_trainer import generate_constrained_group_rollouts
from minionesearch.sft.dataset import (
    PromptBuilder,
    collate_sft_or_sdft_batch,
    compute_museq_behavior_sid,
    extract_session_intent_keywords,
)
from minionesearch.sft.sft_trainer import (
    compute_completion_kl_loss,
    compute_focal_completion_ce_loss,
    compute_rdrop_symmetric_kl,
    gather_completion_logits,
)
from minionesearch.sft.vocab_init import initialize_sid_embeddings_semantic


class TinyCharAndSIDTokenizer:
    """Deterministic tiny tokenizer compatible with HuggingFace `model.generate()` & `SIDVocabulary`."""

    def __init__(self, base_vocab_size: int = 128) -> None:
        self._vocab: Dict[str, int] = {"<pad>": 0, "<eos>": 1}
        for i in range(2, base_vocab_size):
            self._vocab[chr(32 + (i % 90)) + (f"_{i}" if i >= 92 else "")] = i
        self._id2tok: Dict[int, str] = {v: k for k, v in self._vocab.items()}
        self.pad_token_id = 0
        self.eos_token_id = 1
        self.pad_token = "<pad>"
        self.eos_token = "<eos>"
        self._added_tokens: List[str] = []

    def __len__(self) -> int:
        return len(self._vocab)

    def get_vocab(self) -> Dict[str, int]:
        return dict(self._vocab)

    def add_tokens(self, new_tokens: List[str], special_tokens: bool = False) -> int:
        cnt = 0
        for t in new_tokens:
            if t not in self._vocab:
                idx = len(self._vocab)
                self._vocab[t] = idx
                self._id2tok[idx] = t
                self._added_tokens.append(t)
                cnt += 1
        return cnt

    def _encode_str(self, text: str) -> List[int]:
        ids: List[int] = []
        i = 0
        # Match `<x_num>` SID tokens atomically, other chars hash into [2..90]
        while i < len(text):
            if text[i] == "<":
                end = text.find(">", i)
                if end != -1:
                    cand = text[i : end + 1]
                    if cand in self._vocab:
                        ids.append(self._vocab[cand])
                        i = end + 1
                        continue
            ids.append(2 + (ord(text[i]) % 80))
            i += 1
        return ids

    def __call__(
        self,
        texts: str | List[str],
        add_special_tokens: bool = False,
        return_tensors: str | None = None,
    ) -> Dict[str, List[int] | List[List[int]] | torch.Tensor]:
        if isinstance(texts, str):
            ids = self._encode_str(texts)
            if return_tensors == "pt":
                return {"input_ids": torch.tensor([ids], dtype=torch.long)}
            return {"input_ids": ids}
        batch_ids = [self._encode_str(t) for t in texts]
        if return_tensors == "pt":
            max_l = max(len(r) for r in batch_ids)
            padded = [[self.pad_token_id] * (max_l - len(r)) + r for r in batch_ids]
            return {"input_ids": torch.tensor(padded, dtype=torch.long)}
        return {"input_ids": batch_ids}

    def batch_decode(self, batch_ids: torch.Tensor, skip_special_tokens: bool = True) -> List[str]:
        out: List[str] = []
        for row in batch_ids.tolist():
            parts = []
            for tid in row:
                if skip_special_tokens and tid in (0, 1):
                    continue
                parts.append(self._id2tok.get(int(tid), ""))
            out.append("".join(parts))
        return out


def _make_tiny_qwen_and_trie():
    torch.manual_seed(42)
    tok = TinyCharAndSIDTokenizer(base_vocab_size=128)
    cfg = Qwen2Config(
        vocab_size=len(tok),
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        pad_token_id=0,
        eos_token_id=1,
    )
    model = Qwen2ForCausalLM(cfg)
    sid_vocab = SIDVocabulary((6, 6, 6, 6))
    item_to_codes = {
        10: (0, 1, 2, 3),
        20: (0, 1, 4, 5),
        30: (2, 3, 1, 0),
        40: (2, 5, 0, 2),
        50: (4, 0, 3, 1),
        60: (5, 2, 2, 4),
    }
    corpus = [
        {"item_id": iid, "item_title": f"商品{iid}", "category_level1_name": "服饰", "category_level2_id": (iid // 20) + 1, "posterior_ctr": 0.05}
        for iid in item_to_codes
    ]
    level_tids = initialize_sid_embeddings_semantic(model, tok, sid_vocab, corpus, item_to_codes)
    trie = GPUPrefixTrie(sid_vocab, item_to_codes)
    return model, tok, trie, level_tids, corpus


def test_vocab_init_and_user_prefix_kv_cache_exact_match() -> None:
    model, tok, trie, level_tids, _ = _make_tiny_qwen_and_trie()
    assert model.get_input_embeddings().weight.shape[0] == len(tok)
    assert not torch.isnan(model.get_input_embeddings().weight).any()

    engine = UserPrefixKVCacheEngine(model, tok, trie, level_tids)
    user_prefix = "用户画像:F,24-30 | 历史交互:<a_0><b_1><c_2><d_3> | "
    query_suffix = "当前搜索:连衣裙\n推荐商品ID:"

    # 1. Full uncached constrained beam search
    full_ids = tok(user_prefix + query_suffix, return_tensors="pt")["input_ids"]
    preds_uncached = engine.generate_constrained_beams_single(full_ids, num_beams=4, past_key_values=None)

    # 2. User-prefix KV-cached constrained beam search
    pref_ids, kv_cache = engine.precompute_user_prefix_cache(user_prefix)
    suf_ids = tok(query_suffix, return_tensors="pt")["input_ids"]
    cat_ids = torch.cat([pref_ids, suf_ids], dim=1)
    preds_cached = engine.generate_constrained_beams_single(cat_ids, num_beams=4, past_key_values=kv_cache)

    assert len(preds_uncached) == 4
    assert preds_uncached == preds_cached, f"KV-cache mismatch: {preds_uncached} vs {preds_cached}"


def test_joint_sdft_completion_kl_gradient_flow() -> None:
    _, tok, trie, _, _ = _make_tiny_qwen_and_trie()
    sid_10 = trie.item_to_sid_str[10]
    batch = [
        {
            "prompt": "用户画像:M | 当前搜索:跑鞋\n推荐商品ID:",
            "teacher_prompt": "用户画像:M | 当前搜索:跑鞋 | 意图线索:运动户外\n推荐商品ID:",
            "target": sid_10,
        }
    ]
    packed = collate_sft_or_sdft_batch(batch, tok, include_teacher=True)
    assert "teacher_input_ids" in packed

    v_size = len(tok)
    stu_logits = torch.randn((1, packed["input_ids"].size(1), v_size), requires_grad=True)
    tea_logits = torch.randn((1, packed["teacher_input_ids"].size(1), v_size), requires_grad=True)

    stu_c = gather_completion_logits(stu_logits, packed["comp_logit_pos"])
    tea_c = gather_completion_logits(tea_logits, packed["teacher_comp_logit_pos"])

    # Identical logits -> KL must be 0
    zero_kl = compute_completion_kl_loss(stu_c, stu_c.detach(), packed["comp_mask"], temperature=1.5)
    assert math.isclose(float(zero_kl.item()), 0.0, abs_tol=1e-5)

    # Distinct logits -> KL > 0 and stop-grad on teacher inside KL!
    kl = compute_completion_kl_loss(stu_c, tea_c, packed["comp_mask"], temperature=1.5)
    assert float(kl.item()) > 0.0
    kl.backward()
    assert stu_logits.grad is not None and stu_logits.grad.abs().sum().item() > 0.0
    assert tea_logits.grad is None


def test_sp_tpma_fixes_cross_branch_leak_and_std_explosion() -> None:
    # Case A: Ground truth is `(1, 5, 2, 0, 0)`.
    # Rollout 0 has WRONG coarse `<a_7>` but coincidentally `<b_5>`: `(7, 5, 2, 0, 0)`.
    # Rollout 1 has RIGHT coarse `<a_1>` but wrong `<b_3>`: `(1, 3, 0, 0, 0)`.
    gt_sets = [[(1, 5, 2, 0, 0)]]
    codes = torch.tensor([[[7, 5, 2, 0, 0], [1, 3, 0, 0, 0]]], dtype=torch.long)
    rewards = torch.tensor([[0.0, 0.0]], dtype=torch.float32)

    sp_adv, _, _ = compute_sp_tpma_advantages(codes, gt_sets, rewards, w_item=0.0)
    # At depth 1 (`<b_*>`), neither rollout has ancestor-valid `(1, 5)` match, so depth-1 prefix adv MUST be 0!
    assert math.isclose(float(sp_adv[0, 0, 1].item()), 0.0, abs_tol=1e-6)
    assert math.isclose(float(sp_adv[0, 1, 1].item()), 0.0, abs_tol=1e-6)
    # At depth 0 (`<a_*>`), rollout 1 (`<a_1>`) beats rollout 0 (`<a_7>`)!
    assert float(sp_adv[0, 1, 0].item()) > float(sp_adv[0, 0, 0].item())

    # Case B: 15 out of 16 rollouts match `<a_1>` and 1 rollout misses (`<a_4>`).
    g16_rows = [[1, 0, 0, 0, 0]] * 15 + [[4, 0, 0, 0, 0]]
    codes_16 = torch.tensor([g16_rows], dtype=torch.long)
    rew_16 = torch.zeros((1, 16), dtype=torch.float32)

    orig_adv, _ = compute_tpma_original_advantages(codes_16, gt_sets, rew_16, w_item=0.0)
    sp_adv_16, _, _ = compute_sp_tpma_advantages(codes_16, gt_sets, rew_16, w_item=0.0)

    # Original TPMA divides by tiny std (~0.48), exploding outlier penalty to ~ -3.87!
    assert float(orig_adv[0, 15, 0].abs().item()) > 3.5
    # SP-TPMA preserves clean bounded scale (< 1.0)!
    assert float(sp_adv_16[0, 15, 0].abs().item()) < 1.0

    # Also verify Listwise DPO & PPO+KL surrogate
    pol_lp = torch.tensor([[0.0, -2.0, -3.0]], requires_grad=True)
    ref_lp = torch.tensor([[-1.0, -1.0, -1.0]])
    dpo_loss, dpo_m = compute_listwise_dpo_loss(pol_lp, ref_lp, beta=0.2)
    dpo_loss.backward()
    assert dpo_m["dpo/top1_win_rate"] == 1.0

    van_adv, van_g = compute_vanilla_grpo_advantages(torch.tensor([[3.0, 0.0]]), num_levels=4)
    tok_lp = torch.zeros((2, 4), requires_grad=True)
    pg_loss, pg_m = ppo_clipped_kl_surrogate_loss(
        tok_lp, tok_lp.detach(), tok_lp.detach(), van_adv.view(2, 4), van_g.view(2, 4), torch.ones((2, 4))
    )
    pg_loss.backward()
    assert math.isclose(pg_m["rl/kl_div"], 0.0, abs_tol=1e-6)
    assert pg_m["rl/clip_frac"] == 0.0

    # Verify genuine multi-step PPO clipping fires (`clip_frac > 0`) when `cur_lp` shifts away from `old_lp`!
    shifted_lp = (tok_lp.detach() + 0.5).requires_grad_(True)
    pg_loss2, pg_m2 = ppo_clipped_kl_surrogate_loss(
        shifted_lp, tok_lp.detach(), tok_lp.detach(), van_adv.view(2, 4), van_g.view(2, 4), torch.ones((2, 4)),
        use_ppo_clip=True,
    )
    pg_loss2.backward()
    assert pg_m2["rl/clip_frac"] > 0.5


def test_constrained_group_rollouts_and_composite_reward() -> None:
    model, tok, trie, level_tids, corpus = _make_tiny_qwen_and_trie()
    p_ids = tok(["当前搜索:连衣裙", "当前搜索:跑鞋"], return_tensors="pt")["input_ids"]
    p_mask = torch.ones_like(p_ids)

    flat_gen, rollout_codes, rollout_items, n_inj = generate_constrained_group_rollouts(
        raw_policy=model,
        tokenizer=tok,
        trie=trie,
        level_token_ids=level_tids,
        prompt_ids=p_ids,
        prompt_mask=p_mask,
        num_generations=4,
        mode="hybrid",
    )
    assert n_inj == 0
    assert flat_gen.shape == (8, 4)
    assert rollout_codes.shape == (2, 4, 4)
    for group in rollout_items:
        assert len(group) == 4
        assert all(iid in trie.item_to_codes for iid in group)

    rew_engine = SearchRewardEngine(corpus)
    sessions = [
        {"query": "连衣裙", "clicked_item_ids": [10], "purchased_item_ids": [20], "impressed_item_ids": [10, 20, 30]},
        {"query": "跑鞋", "clicked_item_ids": [40], "purchased_item_ids": [], "impressed_item_ids": [40, 50]},
    ]
    rew_t, rew_m = rew_engine.score_rollout_groups(sessions, rollout_items)
    assert rew_t.shape == (2, 4)
    assert "reward/total_mean" in rew_m


def test_report_compiler_renders_all_sections(tmp_path) -> None:
    import json
    from minionesearch.utils import write_jsonl

    bench_dir = tmp_path / "bench"
    runs_dir = tmp_path / "runs"
    out_dir = tmp_path / "report"

    write_jsonl(
        bench_dir / "corpus.jsonl",
        [
            {"item_id": 1, "structured_text": "连衣裙A", "category_level2_id": 10},
            {"item_id": 2, "structured_text": "跑鞋B", "category_level2_id": 20},
        ],
    )
    write_jsonl(
        bench_dir / "test_sessions.jsonl",
        [
            {"session_id": "s1", "query": "连衣裙", "clicked_item_ids": [1], "purchased_item_ids": [1], "slices": ["head_exact"]},
        ],
    )
    runs_dir.mkdir(parents=True, exist_ok=True)
    (runs_dir / "phase1_baselines_test.json").write_text(
        json.dumps(
            {
                "BM25": {
                    "metrics": {"overall": {"Recall@1": 0.5, "Recall@10": 1.0, "NDCG@10": 0.75}, "slice:head_exact": {"NDCG@10": 0.75}},
                    "latency": {"p50_ms": 1.2, "p95_ms": 1.8, "p99_ms": 2.1, "qps": 800.0},
                }
            }
        ),
        encoding="utf-8",
    )
    (runs_dir / "sid_bench_results.json").write_text(
        json.dumps(
            {
                "KHQE-RQ-OPQ": {
                    "mean_codebook_util": 1.0,
                    "mean_norm_perplexity": 0.95,
                    "collision_rate": 0.0,
                    "max_bucket_size": 1,
                    "cat_purity_l1_d1": 0.85,
                    "cat_purity_l2_d2": 0.78,
                    "cat_purity_l3_d3": 0.72,
                    "query_coclick_align_d1": 0.81,
                    "query_coclick_align_d2": 0.64,
                    "cold_start_d2_cat_acc": 0.76,
                }
            }
        ),
        encoding="utf-8",
    )
    gen_sub = runs_dir / "C4_rl_1.7b_sp_tpma_grpo"
    gen_sub.mkdir(parents=True, exist_ok=True)
    (gen_sub / "test_predictions.json").write_text(json.dumps({"s1": [1, 2]}), encoding="utf-8")
    (gen_sub / "sft_eval_summary.json").write_text(
        json.dumps(
            {
                "online_single_query_no_kv_cache": {"p50_ms": 28.0, "p95_ms": 31.0, "p99_ms": 35.0, "qps": 35.7},
                "online_single_query_user_prefix_kv_cache": {"p50_ms": 12.0, "p95_ms": 14.0, "p99_ms": 16.0, "qps": 83.3},
            }
        ),
        encoding="utf-8",
    )

    compiled = compile_full_report(bench_dir, runs_dir, out_dir)
    assert "C4_rl_1.7b_sp_tpma_grpo" in compiled["generative_runs"]
    md_text = (out_dir / "BENCHMARK_REPORT.md").read_text(encoding="utf-8")
    assert "End-to-End Search Benchmark Comparison" in md_text
    assert "User-Prefix KV-Cache" in md_text


def test_onesearch_v1_museq_and_v2_rdrop_focal_fidelity() -> None:
    model, tok, trie, _, _ = _make_tiny_qwen_and_trie()
    # 1. Test OneSearch-V1 §3.2.1 Eq. (3) Mu-Seq Behavior SID:
    #    - Long sequences (len=500) must NEVER overflow!
    #    - Disjoint short vs long histories must yield distinct `SID_short` and `SID_long`!
    long_seq_500 = [10, 30] * 200 + [50, 60] * 50
    beh_sid = compute_museq_behavior_sid(long_seq_500, trie.item_to_codes, trie.sid_vocab, tau=0.35)
    parsed = trie.sid_vocab.str_to_codes(beh_sid)
    assert parsed is not None and len(parsed) == 4

    pb = PromptBuilder(user_id_mode="mu_seq")
    multi_interest_sess = {
        "user_id": 9,
        "query": "跑鞋",
        # Older clicks on item 10/20 (`<a_0>`), recent 4 clicks on item 50 (`<a_4>`), purchased item 30 (`<a_2>`)
        "hist_clicked_item_ids": [10, 20, 10, 20, 50, 50, 50, 50],
        "hist_purchased_item_ids": [30, 30],
        "hist_queries": ["旧词", "新词"],
    }
    u_pref_multi = pb.format_user_prefix(
        multi_interest_sess, trie.item_to_sid_str, item_to_codes=trie.item_to_codes, sid_vocab=trie.sid_vocab
    )
    # Short window (`[50,50,50,50]`) -> `trie.item_to_sid_str[50]`, Long window (`[10,20,10,20,30,30]`) != `50`!
    sid_50 = trie.item_to_sid_str[50]
    assert f"用户偏好ID:{sid_50}" in u_pref_multi
    assert f"用户偏好ID:{sid_50}{sid_50}" not in u_pref_multi

    # 2. Test OneSearch-V2 §3.2.1 Clean Intent Keywords (must NOT copy full SKU title!)
    sku_title = "超长独家货号XYZ9988夏季新款显瘦连衣裙长裙"
    cmap = {
        10: {
            "item_id": 10,
            "item_title": sku_title,
            "brand_name": "茵曼",
            "category_level2_name": "女装",
            "category_level3_name": "连衣裙",
        }
    }
    sess = {
        "user_id": 7,
        "query": "显瘦裙子",
        "clicked_item_ids": [10],
        "purchased_item_ids": [10],
        "hist_clicked_item_ids": [30, 50],
        "hist_purchased_item_ids": [50],
        "hist_queries": ["夏装"],
    }
    kws = extract_session_intent_keywords(sess, cmap, max_keywords=4)
    assert "女装/连衣裙" in kws and "茵曼" in kws
    assert sku_title not in kws

    u_pref, q_suf = pb.build_inference_pair(
        sess,
        trie.item_to_sid_str,
        query_sid_map={"显瘦裙子": "<a_0><b_1><c_2><d_3>"},
        item_to_codes=trie.item_to_codes,
        sid_vocab=trie.sid_vocab,
    )
    assert "用户偏好ID:" in u_pref
    assert "对应id为<a_0><b_1><c_2><d_3>" in q_suf

    # 3. Test REAL Qwen2ForCausalLM fused R-Drop (verifying `rdrop_loss > 0` even when default config had dropout=0.0!)
    batch = [{"prompt": u_pref + q_suf, "teacher_prompt": u_pref + q_suf, "target": trie.item_to_sid_str[10]}]
    packed = collate_sft_or_sdft_batch(batch, tok, include_teacher=True)
    model.train()
    cat_ids = torch.cat([packed["input_ids"], packed["input_ids"]], dim=0)
    cat_mask = torch.cat([packed["attention_mask"], packed["attention_mask"]], dim=0)
    cat_embeds = torch.nn.functional.dropout(model.get_input_embeddings()(cat_ids), p=0.06, training=True)
    cat_out = model(inputs_embeds=cat_embeds, attention_mask=cat_mask, use_cache=False)
    logits_1, logits_2 = cat_out.logits.chunk(2, dim=0)
    c1 = gather_completion_logits(logits_1, packed["comp_logit_pos"])
    c2 = gather_completion_logits(logits_2, packed["comp_logit_pos"])
    rd = compute_rdrop_symmetric_kl(c1, c2, packed["comp_mask"])
    fc = compute_focal_completion_ce_loss(c1, packed["labels"], packed["comp_logit_pos"], packed["comp_mask"], gamma=1.0)
    (rd + fc).backward()
    assert float(rd.item()) > 1e-6, f"R-Drop silently collapsed to zero: {rd.item()}"
    assert float(fc.item()) > 0.0


def test_native_fast_constrained_beam_engine_exact_kv_equivalence() -> None:
    model, tok, trie, level_tids, _ = _make_tiny_qwen_and_trie()
    fast_eng = NativeFastConstrainedBeamEngine(model, tok, trie, level_tids)
    pref = torch.randint(2, 50, (1, 24))
    suf = torch.randint(2, 50, (1, 8))
    full = torch.cat([pref, suf], dim=1)

    uncached = fast_eng.decode_fast_beams(full, num_beams=4)
    kv = _clone_dynamic_cache(fast_eng.precompute_prefix_kv(pref), repeat_beams=1)
    cached = fast_eng.decode_fast_beams(suf, num_beams=4, tiled_prefix_cache=kv, prefix_len=24)

    assert len(uncached) == 4
    assert uncached == cached, f"FastBeam KV mismatch: {uncached} vs {cached}"


def test_rl_trainer_end_to_end_multi_step_ppo_and_provenance(tmp_path, monkeypatch) -> None:
    import json
    import pytest
    import transformers
    from minionesearch.rl.rl_trainer import RLTrainConfig, train_generative_search_rl
    from minionesearch.utils import write_jsonl

    model, tok, trie, _, corpus = _make_tiny_qwen_and_trie()
    tok.save_pretrained = lambda p: None  # type: ignore[attr-defined]
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", lambda *a, **kw: tok)

    sft_ckpt = tmp_path / "sft_run" / "checkpoint"
    sft_ckpt.mkdir(parents=True)
    model.save_pretrained(sft_ckpt)
    (tmp_path / "sft_run" / "prompt_config.json").write_text(
        json.dumps({"user_id_mode": "mu_seq", "use_query_sid": False}), encoding="utf-8"
    )

    sid_dir = tmp_path / "sid"
    sid_dir.mkdir()
    trie.save(sid_dir / "trie.json")
    (sid_dir / "item_sid_map.json").write_text(
        json.dumps({str(k): v for k, v in trie.item_to_sid_str.items()}), encoding="utf-8"
    )

    bench_dir = tmp_path / "bench"
    write_jsonl(bench_dir / "corpus.jsonl", corpus)
    sess_rows = [
        {
            "session_id": f"s{i}",
            "user_id": 1,
            "time_index": 10 + i,
            "query": "连衣裙",
            "clicked_item_ids": [10],
            "purchased_item_ids": [],
            "impressed_item_ids": [10, 20, 30],
            "hist_queries": [],
            "hist_clicked_item_ids": [],
            "hist_purchased_item_ids": [],
            "slices": ["head_exact"],
        }
        for i in range(4)
    ]
    write_jsonl(bench_dir / "train_sessions.jsonl", sess_rows)
    write_jsonl(bench_dir / "val_sessions.jsonl", sess_rows[:2])
    write_jsonl(bench_dir / "test_sessions.jsonl", sess_rows[2:])

    cfg = RLTrainConfig(
        algo="sp_tpma_grpo", num_generations=4, epochs=1, ppo_epochs=2,
        batch_size=2, lr=1e-3, eval_beams=4, seed=42,
    )
    # Missing explicit rel_rm_ckpt path must fail fast!
    with pytest.raises(FileNotFoundError):
        train_generative_search_rl(bench_dir, sid_dir, sft_ckpt, tmp_path / "out_err", cfg, rel_rm_ckpt=tmp_path / "nonexistent_rm")

    res = train_generative_search_rl(bench_dir, sid_dir, sft_ckpt, tmp_path / "out_ok", cfg, rel_rm_ckpt=None)
    assert res["code_version"] == "v3_clean_orthogonal"
    assert res["inherited_prompt_config"]["use_query_sid"] is False
    assert len(res["val_history"]) >= 2
    assert "actual_gt_injection_rate" in res
    assert "mean_ppo_clip_frac" in res
