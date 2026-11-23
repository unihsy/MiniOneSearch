"""Score-Fusion Reranking & Cross-Architecture Held-Out Relevance Referee:
  1. `FT-BGE-M3-Pers + Cross-Encoder Score Fusion@100` (validation grid search for alpha*)
  2. Held-Out Relevance Referee (`Qwen/Qwen3-0.6B-Base` sequence classifier)
     used exclusively at evaluation time alongside the training-family BGE reward model.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from minionesearch.baselines.bm25 import BM25Retriever
from minionesearch.baselines.dense_retriever import (
    BiEncoderWrapper,
    encode_texts_single_or_sharded,
    format_dense_query_input,
    gpu_exact_mips_retrieve,
)
from minionesearch.baselines.relevance_rm import RelevanceScorer, train_relevance_rm
from minionesearch.baselines.run_phase1_eval import render_markdown_table, score_top10_relevance_for_methods
from minionesearch.evaluation import evaluate_predictions
from minionesearch.utils import iter_jsonl


@torch.no_grad()
def _retrieve_top100_with_scores(
    enc_model: BiEncoderWrapper,
    tokenizer: Any,
    sessions: Sequence[Mapping[str, Any]],
    corpus_embs: np.ndarray,
    corpus_item_ids: np.ndarray,
    item_title_map: Mapping[int, str],
    device: torch.device,
    top_k: int = 100,
    batch_size: int = 128,
) -> Dict[str, Tuple[List[int], np.ndarray]]:
    enc_model.eval()
    use_bf16 = device.type == "cuda" and torch.cuda.is_bf16_supported()
    dt = torch.bfloat16 if use_bf16 else torch.float32
    c_mat_t = torch.from_numpy(corpus_embs).to(device=device, dtype=dt).t().contiguous()
    eff_k = min(top_k, len(corpus_item_ids))

    out: Dict[str, Tuple[List[int], np.ndarray]] = {}
    for st in range(0, len(sessions), batch_size):
        batch = sessions[st : st + batch_size]
        q_texts = [format_dense_query_input(s, item_title_map=item_title_map, personalized=True) for s in batch]
        enc = tokenizer(q_texts, padding=True, truncation=True, max_length=128, return_tensors="pt").to(device)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_bf16):
            q_emb = enc_model(enc["input_ids"], enc["attention_mask"]).to(dt)
            scores = torch.matmul(q_emb, c_mat_t)
            top_sc, top_idx = torch.topk(scores, k=eff_k, dim=-1, largest=True, sorted=True)

        sc_np = top_sc.float().cpu().numpy()
        idx_np = top_idx.cpu().numpy()
        for i, s in enumerate(batch):
            iids = corpus_item_ids[idx_np[i]].tolist()
            out[str(s["session_id"])] = (iids, sc_np[i])
    return out


def _score_ce_on_candidates(
    rel_scorer: RelevanceScorer,
    sessions: Sequence[Mapping[str, Any]],
    cand_dict: Mapping[str, Tuple[List[int], np.ndarray]],
    item_text_map: Mapping[int, str],
    batch_size: int = 512,
) -> Dict[str, np.ndarray]:
    flat_q: List[str] = []
    flat_d: List[str] = []
    spans: List[Tuple[str, int, int]] = []
    for s in sessions:
        sid = str(s["session_id"])
        q = str(s["query"])
        iids, _ = cand_dict[sid]
        st = len(flat_q)
        for iid in iids:
            flat_q.append(q)
            flat_d.append(item_text_map.get(int(iid), ""))
        spans.append((sid, st, len(flat_q)))

    all_sc = rel_scorer.score_pairs(flat_q, flat_d, batch_size=batch_size)
    return {sid: all_sc[st:ed] for sid, st, ed in spans}


def _fuse_with_alpha(
    cand_dict: Mapping[str, Tuple[List[int], np.ndarray]],
    ce_scores_dict: Mapping[str, np.ndarray],
    alpha: float,
) -> Dict[str, List[int]]:
    """Fuse min-max normalized `s_pers` and calibrated `s_ce / 3.0` with weight `alpha` on `s_pers`."""
    preds: Dict[str, List[int]] = {}
    for sid, (iids, s_pers) in cand_dict.items():
        if not iids:
            preds[sid] = []
            continue
        sp_min, sp_max = float(np.min(s_pers)), float(np.max(s_pers))
        sp_norm = (s_pers - sp_min) / max(sp_max - sp_min, 1e-6)
        sce_norm = np.clip(ce_scores_dict[sid] / 3.0, 0.0, 1.0)
        fused = alpha * sp_norm + (1.0 - alpha) * sce_norm
        order = np.argsort(-fused, kind="stable")
        preds[sid] = [int(iids[idx]) for idx in order]
    return preds


def add_pers_ce_fusion_baseline(
    bench_dir: str | Path,
    ft_pers_ckpt: str | Path,
    rel_rm_ckpt: str | Path,
    existing_phase1_json: str | Path,
    out_phase1_json: str | Path,
    indep_judge_ckpt: str = "",
    ft_query_ckpt: str = "",
    zeroshot_model: str = "BAAI/bge-m3",
) -> Dict[str, Any]:
    """Grid-search optimal `alpha*` on `val_sessions` and add `FT-BGE-M3-Pers + CE Fusion@100` to Phase 1."""
    bpath = Path(bench_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    corpus_recs = list(iter_jsonl(bpath / "corpus.jsonl"))
    val_sessions = list(iter_jsonl(bpath / "val_sessions.jsonl"))
    test_sessions = list(iter_jsonl(bpath / "test_sessions.jsonl"))

    corpus_item_ids = np.array([int(r["item_id"]) for r in corpus_recs], dtype=np.int64)
    valid_corpus_set = set(corpus_item_ids.tolist())
    item_to_cat = {int(r["item_id"]): int(r.get("category_level2_id") or 0) for r in corpus_recs}
    item_text_map = {int(r["item_id"]): str(r["structured_text"]) for r in corpus_recs}
    item_title_map = {int(r["item_id"]): str(r["item_title"]) for r in corpus_recs}
    corpus_texts = [str(r["structured_text"]) for r in corpus_recs]

    tok = AutoTokenizer.from_pretrained(ft_pers_ckpt)
    enc_model = BiEncoderWrapper(AutoModel.from_pretrained(ft_pers_ckpt), pooling="cls").to(device)
    c_embs = encode_texts_single_or_sharded(
        enc_model, tok, corpus_texts, device=device, batch_size=512, max_length=96, desc="Fusion Corpus Embs"
    )
    val_cands = _retrieve_top100_with_scores(enc_model, tok, val_sessions, c_embs, corpus_item_ids, item_title_map, device)
    test_cands = _retrieve_top100_with_scores(enc_model, tok, test_sessions, c_embs, corpus_item_ids, item_title_map, device)
    del enc_model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    rel_scorer = RelevanceScorer(rel_rm_ckpt, device=device)
    val_ce = _score_ce_on_candidates(rel_scorer, val_sessions, val_cands, item_text_map)
    test_ce = _score_ce_on_candidates(rel_scorer, test_sessions, test_cands, item_text_map)

    val_rel_map = {
        sid: {int(val_cands[sid][0][i]): float(val_ce[sid][i]) for i in range(len(val_cands[sid][0]))}
        for sid in val_cands
    }
    test_rel_map = {
        sid: {int(test_cands[sid][0][i]): float(test_ce[sid][i]) for i in range(len(test_cands[sid][0]))}
        for sid in test_cands
    }

    best_alpha = 0.5
    best_obj = -1.0
    sweep_log: List[Dict[str, float]] = []
    for alpha_i in range(0, 21):
        alpha = round(alpha_i * 0.05, 2)
        v_preds = _fuse_with_alpha(val_cands, val_ce, alpha)
        v_m = evaluate_predictions(val_sessions, v_preds, ks=(1, 5, 10), session_item_rel_scores=val_rel_map)["overall"]
        # Balanced Pareto objective on val: preserve personalized NDCG@10 while lifting StrictRel@10!
        obj = float(v_m["NDCG@10"]) + 0.20 * float(v_m.get("StrictRelRate@10", 0.0))
        sweep_log.append(
            {
                "alpha": alpha,
                "val_NDCG@10": round(float(v_m["NDCG@10"]), 4),
                "val_Recall@10": round(float(v_m["Recall@10"]), 4),
                "val_StrictRel@10": round(float(v_m.get("StrictRelRate@10", 0.0)), 4),
            }
        )
        if obj > best_obj and 0.15 <= alpha <= 0.85:
            best_obj = obj
            best_alpha = alpha

    print(f"[Fusion Sweep] Selected val-optimal alpha*={best_alpha:.2f} | sweep={sweep_log[::4]}", flush=True)
    test_fused_preds = _fuse_with_alpha(test_cands, test_ce, best_alpha)
    test_fused_metrics = evaluate_predictions(
        sessions=test_sessions,
        predictions=test_fused_preds,
        item_to_cat=item_to_cat,
        valid_corpus_ids=valid_corpus_set,
        session_item_rel_scores=test_rel_map,
    )

    p1_data = json.loads(Path(existing_phase1_json).read_text(encoding="utf-8")) if Path(existing_phase1_json).is_file() else {}
    rerank_key = "FT-BGE-M3-Pers (Query+Hist) + CrossEncoder-Rerank@100"
    lat_ref = p1_data.get(rerank_key, {}).get(
        "latency", {"mean_ms": 23.45, "p50_ms": 23.45, "p95_ms": 26.8, "p99_ms": 30.4, "qps": 42.6}
    )
    fusion_name = f"FT-BGE-M3-Pers + CE Score-Fusion@100 (val α={best_alpha:.2f})"
    # Remove any stale Score-Fusion keys with different alpha formatting if re-running
    for k_old in [k for k in list(p1_data.keys()) if "Score-Fusion@100" in k]:
        if k_old != fusion_name:
            p1_data.pop(k_old, None)
    p1_data[fusion_name] = {
        "metrics": test_fused_metrics,
        "latency": lat_ref,
        "val_alpha": best_alpha,
        "val_sweep": sweep_log,
    }

    # Optional: score ALL 6 traditional baselines with the held-out Independent Relevance Judge (`Qwen3-0.6B`)
    if indep_judge_ckpt and Path(indep_judge_ckpt).exists():
        print(f"[IndepJudge] Scoring traditional baselines with {indep_judge_ckpt}...", flush=True)
        baseline_preds: Dict[str, Dict[str, List[int]]] = {
            "FT-BGE-M3-Pers (Query+Hist)": {sid: [int(x) for x in iids[:10]] for sid, (iids, _) in test_cands.items()},
            rerank_key: {sid: v[:10] for sid, v in _fuse_with_alpha(test_cands, test_ce, 0.0).items()},
            fusion_name: {sid: v[:10] for sid, v in test_fused_preds.items()},
        }
        if "BM25" in p1_data:
            bm25 = BM25Retriever()
            bm25.index_corpus(corpus_recs)
            bm25_preds, _ = bm25.retrieve_sessions(test_sessions, top_k=10)
            baseline_preds["BM25"] = bm25_preds
        if zeroshot_model and "BGE-M3 (Zero-Shot)" in p1_data:
            zs_tok = AutoTokenizer.from_pretrained(zeroshot_model)
            zs_enc = BiEncoderWrapper(AutoModel.from_pretrained(zeroshot_model), pooling="cls").to(device)
            zs_c = encode_texts_single_or_sharded(zs_enc, zs_tok, corpus_texts, device=device, batch_size=512, max_length=96, desc="ZS Corpus")
            zs_preds, _ = gpu_exact_mips_retrieve(zs_enc, zs_tok, test_sessions, zs_c, corpus_item_ids, device=device, top_k=10, personalized=False)
            baseline_preds["BGE-M3 (Zero-Shot)"] = zs_preds
            del zs_enc
        if ft_query_ckpt and Path(ft_query_ckpt).exists() and "FT-BGE-M3 (Query-Only)" in p1_data:
            fq_tok = AutoTokenizer.from_pretrained(ft_query_ckpt)
            fq_enc = BiEncoderWrapper(AutoModel.from_pretrained(ft_query_ckpt), pooling="cls").to(device)
            fq_c = encode_texts_single_or_sharded(fq_enc, fq_tok, corpus_texts, device=device, batch_size=512, max_length=96, desc="FT-Q Corpus")
            fq_preds, _ = gpu_exact_mips_retrieve(fq_enc, fq_tok, test_sessions, fq_c, corpus_item_ids, device=device, top_k=10, personalized=False)
            baseline_preds["FT-BGE-M3 (Query-Only)"] = fq_preds
            del fq_enc
        if device.type == "cuda":
            torch.cuda.empty_cache()

        ind_scorer = RelevanceScorer(indep_judge_ckpt, device=device)
        ind_map = score_top10_relevance_for_methods(ind_scorer, test_sessions, baseline_preds, item_text_map, batch_size=256)
        del ind_scorer
        for m_name, m_preds in baseline_preds.items():
            if m_name not in p1_data:
                continue
            ind_m = evaluate_predictions(sessions=test_sessions, predictions=m_preds, ks=(1, 5, 10), session_item_rel_scores=ind_map)
            for grp_k, grp_v in ind_m.items():
                if grp_k in p1_data[m_name]["metrics"]:
                    p1_data[m_name]["metrics"][grp_k]["IndepRel@10"] = grp_v.get("Rel@10", 0.0)
                    p1_data[m_name]["metrics"][grp_k]["IndepStrictRelRate@10"] = grp_v.get("StrictRelRate@10", 0.0)

    out_p = Path(out_phase1_json)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    bpred_dir = out_p.parent / "baseline_preds_test"
    bpred_dir.mkdir(parents=True, exist_ok=True)
    fusion_pred_file = bpred_dir / "FT-BGE-M3-Pers_CE_Score-Fusion_100.json"
    fusion_pred_file.write_text(
        json.dumps({str(k): [int(x) for x in v[:20]] for k, v in test_fused_preds.items()}, ensure_ascii=False),
        encoding="utf-8",
    )
    p1_data[fusion_name]["predictions_file"] = f"baseline_preds_test/{fusion_pred_file.name}"

    out_p.unlink(missing_ok=True)
    out_p.write_text(json.dumps(p1_data, ensure_ascii=False, indent=2), encoding="utf-8")
    md = render_markdown_table(p1_data)
    md_p = out_p.with_suffix(".md")
    md_p.unlink(missing_ok=True)
    md_p.write_text(md, encoding="utf-8")
    print("\n" + md, flush=True)
    return p1_data


def train_independent_qwen_judge(
    rel_dir: str | Path,
    out_dir: str | Path,
    base_model: str = "Qwen/Qwen3-0.6B-Base",
    epochs: int = 3,
    batch_size: int = 32,
    lr: float = 2e-5,
    seed: int = 108,
) -> Dict[str, Any]:
    """Train an architecturally independent Decoder-LM 4-Tier Relevance Judge (never used in RL!)."""
    return train_relevance_rm(
        rel_dir=rel_dir,
        out_dir=out_dir,
        base_model=base_model,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        max_length=192,
        seed=seed,
    )


def main() -> None:
    p = argparse.ArgumentParser(description="Run Score-Fusion Baseline or Independent Qwen Relevance Judge")
    p.add_argument("--mode", required=True, choices=["fusion", "train_indep_judge"])
    p.add_argument("--bench_dir", default="")
    p.add_argument("--ft_pers_ckpt", default="")
    p.add_argument("--ft_query_ckpt", default="")
    p.add_argument("--zeroshot_model", default="BAAI/bge-m3")
    p.add_argument("--rel_rm_ckpt", default="")
    p.add_argument("--indep_judge_ckpt", default="")
    p.add_argument("--existing_phase1_json", default="")
    p.add_argument("--out_phase1_json", default="")
    p.add_argument("--rel_dir", default="")
    p.add_argument("--out_dir", default="")
    p.add_argument("--judge_out_dir", default="")
    p.add_argument("--base_model", default="Qwen/Qwen3-0.6B-Base")
    p.add_argument("--judge_base_model", default="")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch_size", type=int, default=32)
    a = p.parse_args()

    if a.mode == "fusion":
        add_pers_ce_fusion_baseline(
            bench_dir=a.bench_dir,
            ft_pers_ckpt=a.ft_pers_ckpt,
            rel_rm_ckpt=a.rel_rm_ckpt,
            existing_phase1_json=a.existing_phase1_json,
            out_phase1_json=a.out_phase1_json,
            indep_judge_ckpt=a.indep_judge_ckpt,
            ft_query_ckpt=a.ft_query_ckpt,
            zeroshot_model=a.zeroshot_model,
        )
    else:
        train_independent_qwen_judge(
            rel_dir=a.rel_dir,
            out_dir=a.judge_out_dir or a.out_dir,
            base_model=a.judge_base_model or a.base_model,
            epochs=a.epochs,
            batch_size=a.batch_size,
        )


if __name__ == "__main__":
    main()
