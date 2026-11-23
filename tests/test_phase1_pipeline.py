"""Unit tests for Phase 1 chronological data pipeline, IR metrics, BM25, and Relevance RM math."""
from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
import torch

from minionesearch.baselines.bm25 import BM25Retriever, tokenize_zh_ecommerce
from minionesearch.baselines.relevance_rm import (
    compute_relevance_eval_metrics,
    expected_relevance_from_logits,
)
from minionesearch.data import BuildConfig, build_benchmark_from_raw, verify_benchmark_integrity
from minionesearch.evaluation import compute_query_ir_metrics, evaluate_predictions
from minionesearch.utils import iter_jsonl, write_jsonl


def test_chronological_builder_and_slices(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    out_dir = tmp_path / "built_mini"

    # Create synthetic items across categories
    items = []
    for iid in range(1, 41):
        items.append(
            {
                "item_id": iid,
                "item_title": f"测试品牌{iid % 4} 运动透气跑步鞋型号X{iid}",
                "brand_name": f"测试品牌{iid % 4}",
                "seller_name": f"旗舰店{iid % 5}",
                "category_level1_id": (iid % 2) + 1,
                "category_level1_name": "运动户外" if (iid % 2 == 0) else "服饰鞋包",
                "category_level2_id": (iid % 4) + 1,
                "category_level2_name": f"二级类目{(iid % 4) + 1}",
                "category_level3_id": (iid % 6) + 1,
                "category_level3_name": f"三级类目{(iid % 6) + 1}",
            }
        )
    write_jsonl(raw_dir / "items_lite" / "train.jsonl", items)
    write_jsonl(
        raw_dir / "users_lite" / "train.jsonl",
        [
            {"user_id": 101, "gender": "M", "age_bucket": "18-23", "fre_province": "广东", "fre_city": "深圳"},
            {"user_id": 102, "gender": "F", "age_bucket": "24-30", "fre_province": "浙江", "fre_city": "杭州"},
        ],
    )

    # Sessions testing:
    # 1) Same-timestamp sessions (s1 & s2 at t=100) MUST NOT see each other!
    # 2) Future session target (item 15 at t=300) injected into rank prehistory at t=100 MUST be stripped!
    # 3) Item 39 never clicked in train -> must trigger `cold_start_item` in test!
    recall_rows = [
        {
            "session_id": "s1",
            "user_id": 101,
            "time_index": 100,
            "query": "测试品牌1 跑步鞋",
            "clicked_item_ids": [1, 5],
            "purchased_item_ids": [1],
            "impressed_item_ids": [1, 2, 3, 4, 5, 6],
            "split": "train",
        },
        {
            "session_id": "s2",
            "user_id": 101,
            "time_index": 100,
            "query": "同时刻第二个查询",
            "clicked_item_ids": [2],
            "purchased_item_ids": [],
            "impressed_item_ids": [2, 7, 8],
            "split": "train",
        },
        {
            "session_id": "s3",
            "user_id": 101,
            "time_index": 200,
            "query": "透气运动鞋夏季新款加长复杂长尾词",
            "clicked_item_ids": [10],
            "purchased_item_ids": [],
            "impressed_item_ids": [10, 11, 12],
            "split": "train",
        },
        {
            "session_id": "s4",
            "user_id": 101,
            "time_index": 300,
            "query": "冷启动新款跑鞋超长复杂属性查询",
            "clicked_item_ids": [15, 39],
            "purchased_item_ids": [39],
            "impressed_item_ids": [15, 39, 20],
            "split": "test",
        },
    ]
    write_jsonl(raw_dir / "recall_lite" / "train.jsonl", recall_rows)

    # Inject deliberate future-leak candidate (item 15 & 39) + legitimate pre-history item (item 25)
    rank_rows = [
        {
            "session_id": "s1",
            "user_id": 101,
            "recently_clicked_item_ids": [25, 15, 39, 1],
            "recently_purchased_item_ids": [39],
        }
    ]
    write_jsonl(raw_dir / "rank_lite" / "train.jsonl", rank_rows)

    rel_rows = [
        {"query": f"q{i}", "item_title": f"t{i}", "brand": "b", "attr_value": "a", "score": i % 4}
        for i in range(30)
    ]
    write_jsonl(raw_dir / "relevance" / "train.jsonl", rel_rows)

    cfg = BuildConfig(tier="mini", target_corpus_size=35, max_users=10, val_ratio=0.0, seed=42, use_rank_prehistory=True)
    stats = build_benchmark_from_raw(raw_dir, out_dir, cfg)
    assert stats["num_corpus_items"] == 35
    assert verify_benchmark_integrity(out_dir)["verified"] is True

    train_sess = {s["session_id"]: s for s in iter_jsonl(out_dir / "train_sessions.jsonl")}
    test_sess = {s["session_id"]: s for s in iter_jsonl(out_dir / "test_sessions.jsonl")}

    # Same-timestamp check: s1 and s2 at t=100 must have ZERO causal queries from t=100
    assert train_sess["s1"]["hist_queries"] == []
    assert train_sess["s2"]["hist_queries"] == []
    # Future leak check: items 1, 15, 39 must have been stripped from s1's rank pre-history, while 25 remains!
    assert 25 in train_sess["s1"]["hist_clicked_item_ids"]
    assert 15 not in train_sess["s1"]["hist_clicked_item_ids"]
    assert 39 not in train_sess["s1"]["hist_clicked_item_ids"]
    assert 1 not in train_sess["s1"]["hist_clicked_item_ids"]

    # At t=200 (s3), both t=100 queries (s1 & s2) ARE strictly in the past and must appear in history
    assert len(train_sess["s3"]["hist_queries"]) == 2
    assert all(ht < 200 for ht in train_sess["s3"]["hist_query_times"])

    # Cold-start slice check on s4 (item 39 had 0 train positives)
    assert "cold_start_item" in test_sess["s4"]["slices"]
    assert "long_tail_complex" in test_sess["s4"]["slices"]


def test_true_iterative_5core_and_pure_causal_default(tmp_path: Path) -> None:
    """Verify default `use_rank_prehistory=False` touches zero prehistory and iterative `core5` guarantees `train_pos_count >= 5`."""
    raw_dir = tmp_path / "raw_c5"
    out_dir = tmp_path / "built_c5"

    # Item 1 & 2 get >= 5 clicks from active users (u1..u6, each >= 3 train sessions).
    # Item 9 gets 5 clicks ONLY from 1-session transient users -> MUST be peeled away by iterative (3,5)-core!
    items = [
        {"item_id": i, "item_title": f"热门商品{i}", "brand_name": "品牌A", "category_level1_id": 1, "category_level2_id": 1}
        for i in [1, 2, 9]
    ]
    write_jsonl(raw_dir / "items_lite" / "train.jsonl", items)
    users = [{"user_id": u} for u in range(1, 20)]
    write_jsonl(raw_dir / "users_lite" / "train.jsonl", users)

    sessions = []
    # 6 active users (u=1..6), each has 3 train sessions clicking items 1 and 2 + 1 test session
    for u in range(1, 7):
        for step_t in (10, 20, 30):
            sessions.append({
                "session_id": f"u{u}_t{step_t}",
                "user_id": u,
                "time_index": step_t,
                "query": f"查询{step_t}",
                "clicked_item_ids": [1 if step_t < 30 else 2],
                "purchased_item_ids": [],
                "impressed_item_ids": [1, 2, 9],
                "split": "train",
            })
        sessions.append({
            "session_id": f"u{u}_test",
            "user_id": u,
            "time_index": 50,
            "query": "测试查询",
            "clicked_item_ids": [1],
            "purchased_item_ids": [],
            "impressed_item_ids": [1, 2],
            "split": "test",
        })
    # 5 transient 1-session users clicking item 9 -> item 9 has raw_train_cnt=5, but 0 after user_k>=3 peeling!
    for u in range(10, 15):
        sessions.append({
            "session_id": f"trans_{u}",
            "user_id": u,
            "time_index": 15,
            "query": "低活查询",
            "clicked_item_ids": [9],
            "purchased_item_ids": [],
            "impressed_item_ids": [9],
            "split": "train",
        })
    write_jsonl(raw_dir / "recall_lite" / "train.jsonl", sessions)

    cfg = BuildConfig(tier="core5", target_corpus_size=0, core_item_k=5, core_user_k=3, val_ratio=0.0, seed=42)
    assert cfg.use_rank_prehistory is False
    st = build_benchmark_from_raw(raw_dir, out_dir, cfg)
    assert st["num_corpus_items"] == 2  # Item 9 peeled away!
    assert st["corpus_train_pos_summary"]["min"] >= 5
    assert st["corpus_train_pos_summary"]["zero_pos_items"] == 0
    assert verify_benchmark_integrity(out_dir)["verified"] is True


def test_ir_metrics_deduplication_and_graded_ndcg() -> None:
    # Ground truth: item 10 clicked, item 20 purchased (higher grade!)
    # Predictions include duplicate 10 at rank 2 -> must NOT double count!
    m = compute_query_ir_metrics(
        ranked_item_ids=[10, 10, 20, 999, 30],
        clicked_ids={10},
        purchased_ids={20},
        ks=(1, 3, 5),
        item_to_cat={10: 1, 20: 2, 30: 1, 999: 3},
        valid_corpus_ids={10, 20, 30},
        item_rel_scores={10: 2.0, 20: 3.0, 999: 0.0, 30: 1.0},
    )
    assert math.isclose(m["Recall@1"], 0.5)
    assert math.isclose(m["Recall@3"], 1.0)
    from minionesearch.evaluation.stats import holm_bonferroni_adjust, user_clustered_paired_bootstrap

    adj = holm_bonferroni_adjust([0.01, 0.04, 0.20])
    assert math.isclose(adj[0], 0.03) and math.isclose(adj[1], 0.08) and math.isclose(adj[2], 0.20)

    mock_sess = [
        {"session_id": f"s{i}", "user_id": i // 2, "clicked_item_ids": [1], "purchased_item_ids": []}
        for i in range(40)
    ]
    pa = {f"s{i}": [1, 2] for i in range(40)}
    pb = {f"s{i}": ([1, 2] if i < 10 else [2, 1]) for i in range(40)}
    b_res = user_clustered_paired_bootstrap(mock_sess, pa, pb, metric_key="NDCG@10", n_boot=1000, seed=7)
    assert b_res["diff"] > 0.20
    assert b_res["ci95_low_cluster"] > 0.0
    assert b_res["p_two_sided_cluster"] < 0.01
    assert math.isclose(m["MRR@10"], 1.0)
    # Graded NDCG@3: item 10 (gain=1) at rank 0, placeholder at rank 1, item 20 (gain=3) at rank 2
    expected_gdcg = 1.0 / math.log2(2.0) + 3.0 / math.log2(4.0)
    ideal_gdcg = 3.0 / math.log2(2.0) + 1.0 / math.log2(3.0)
    assert math.isclose(m["GradedNDCG@3"], expected_gdcg / ideal_gdcg, rel_tol=1e-5)
    # Valid rate over raw top-5 `[10, 10, 20, 999, 30]`: 4/5 valid, 3/5 unique valid
    assert math.isclose(m["ValidRate@10"], 0.8)
    assert math.isclose(m["UniqueValidRate@10"], 0.6)


def test_bm25_hybrid_chinese_and_alnum_retrieval() -> None:
    corpus = [
        {"item_id": 100, "structured_text": "vivo x200 pro 旗舰拍照手机 512G | 品牌:vivo | 类目:手机"},
        {"item_id": 200, "structured_text": "纯棉防尘床边地毯 120*160cm | 品牌:爱家 | 类目:家纺/地毯"},
        {"item_id": 300, "structured_text": "法式复古半身裙秋季高腰显瘦长裙 | 类目:女装/半身裙"},
    ]
    toks = tokenize_zh_ecommerce("vivox200pro 手机")
    assert any("vivox200pro" in t for t in toks)

    bm25 = BM25Retriever()
    bm25.index_corpus(corpus)
    sessions = [
        {"session_id": "q1", "query": "vivo x200pro 手机"},
        {"session_id": "q2", "query": "床边防尘地毯"},
    ]
    preds, lat = bm25.retrieve_sessions(sessions, top_k=3)
    assert preds["q1"][0] == 100
    assert preds["q2"][0] == 200
    assert lat["qps"] > 0.0

    eval_res = evaluate_predictions(
        sessions=[
            {"session_id": "q1", "clicked_item_ids": [100], "purchased_item_ids": [], "slices": ["head_exact"]},
            {"session_id": "q2", "clicked_item_ids": [200], "purchased_item_ids": [200], "slices": ["long_tail_complex"]},
        ],
        predictions=preds,
        ks=(1, 3),
    )
    assert math.isclose(eval_res["overall"]["Recall@1"], 1.0)
    assert math.isclose(eval_res["slice:head_exact"]["NDCG@1"], 1.0)


def test_ordinal_relevance_math() -> None:
    logits = torch.tensor([[10.0, -10.0, -10.0, -10.0], [-10.0, -10.0, -10.0, 10.0]])
    exp_rel = expected_relevance_from_logits(logits)
    assert math.isclose(float(exp_rel[0].item()), 0.0, abs_tol=1e-4)
    assert math.isclose(float(exp_rel[1].item()), 3.0, abs_tol=1e-4)

    metrics = compute_relevance_eval_metrics(
        gold_labels=[0, 1, 2, 3],
        pred_classes=[0, 1, 2, 3],
        pred_expected=[0.05, 1.02, 1.98, 2.95],
    )
    assert metrics["accuracy"] == 1.0
    assert metrics["spearman_rho"] == 1.0
    assert metrics["strict_rel_auc"] == 1.0
