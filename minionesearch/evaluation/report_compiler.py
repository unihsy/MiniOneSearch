"""Unified Benchmark Report Compiler (Generates All 5 Publication Tables + Rel@10 Scoring).

Reads experiment outputs across Phases 1-4, optionally batch-scores `Rel@1/5/10` with the
trained 4-Tier Relevance Cross-Encoder on any generative `test_predictions.json`, and writes
a complete, self-contained `BENCHMARK_REPORT.md` + `compiled_all_results.json`.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import torch

from minionesearch.baselines.relevance_rm import RelevanceScorer
from minionesearch.baselines.run_phase1_eval import score_top10_relevance_for_methods
from minionesearch.evaluation import evaluate_predictions
from minionesearch.evaluation.stats import run_hypothesis_family
from minionesearch.tokenizer.sid_bench import render_sid_bench_markdown
from minionesearch.utils import iter_jsonl


def _fmt_pct(val: float) -> str:
    return f"{val * 100:.2f}%"


def _build_comparison_md_table(
    title: str,
    entries: Sequence[tuple[str, Mapping[str, Any]]],
    has_indep_judge: bool = False,
) -> str:
    lines = [f"## {title}\n"]
    headers = [
        "Model / Variant",
        "Recall@1",
        "Recall@5",
        "Recall@10",
        "NDCG@10",
        "GradedNDCG@10",
        "MRR@10",
        "Rel@10",
        "StrictRel@10",
    ]
    if has_indep_judge:
        headers.extend(["IndepRel@10", "IndepStrictRel@10"])
    headers.extend(["CatDiv@10", "Valid@10"])

    lines.append("| " + " | ".join(headers) + " |")
    lines.append("|" + "|".join(["---"] * len(headers)) + "|")
    for label, mdict in entries:
        ov = mdict.get("overall", {})
        row = [
            f"**{label}**",
            _fmt_pct(ov.get("Recall@1", 0.0)),
            _fmt_pct(ov.get("Recall@5", 0.0)),
            _fmt_pct(ov.get("Recall@10", 0.0)),
            _fmt_pct(ov.get("NDCG@10", 0.0)),
            _fmt_pct(ov.get("GradedNDCG@10", 0.0)),
            _fmt_pct(ov.get("MRR@10", 0.0)),
            f"{ov['Rel@10']:.3f}" if ("Rel@10" in ov and float(ov.get("Rel@10", 0.0)) > 0.0) else "N/A",
            _fmt_pct(ov.get("StrictRelRate@10", 0.0)) if ("Rel@10" in ov and float(ov.get("Rel@10", 0.0)) > 0.0) else "N/A",
        ]
        if has_indep_judge:
            if "IndepRel@10" in ov and float(ov.get("IndepRel@10", 0.0)) > 0.0:
                row.extend(
                    [
                        f"{ov['IndepRel@10']:.3f}",
                        _fmt_pct(ov.get("IndepStrictRelRate@10", 0.0)),
                    ]
                )
            else:
                row.extend(["N/A", "N/A"])
        row.extend(
            [
                _fmt_pct(ov.get("CatDiv@10", 0.0)),
                _fmt_pct(ov.get("ValidRate@10", 1.0)),
            ]
        )
        lines.append("| " + " | ".join(row) + " |")

    # Slice breakdown
    slices = ["head_exact", "personalized_ambiguous", "long_tail_complex", "cold_start_item"]
    lines.append("\n| Model / Variant | " + " | ".join([f"`{s}` (NDCG@10 / Recall@10)" for s in slices]) + " |")
    lines.append("|---|" + "|".join(["---"] * len(slices)) + "|")
    for label, mdict in entries:
        srow = [f"**{label}**"] + [
            f"{_fmt_pct(mdict.get(f'slice:{s}', {}).get('NDCG@10', 0.0))} / {_fmt_pct(mdict.get(f'slice:{s}', {}).get('Recall@10', 0.0))}"
            for s in slices
        ]
        lines.append("| " + " | ".join(srow) + " |")
    return "\n".join(lines) + "\n"


def compile_full_report(
    bench_dir: str | Path,
    results_root: str | Path,
    out_dir: str | Path,
    rel_rm_ckpt: Optional[str | Path] = None,
    indep_judge_ckpt: Optional[str | Path] = None,
) -> Dict[str, Any]:
    bpath = Path(bench_dir)
    rroot = Path(results_root)
    opath = Path(out_dir)
    opath.mkdir(parents=True, exist_ok=True)

    corpus_recs = list(iter_jsonl(bpath / "corpus.jsonl"))
    test_sessions = list(iter_jsonl(bpath / "test_sessions.jsonl"))
    item_text_map = {int(r["item_id"]): str(r["structured_text"]) for r in corpus_recs}
    item_to_cat = {int(r["item_id"]): int(r.get("category_level2_id") or 0) for r in corpus_recs}
    valid_corpus_ids = {int(r["item_id"]) for r in corpus_recs}

    # Discover all generative experiment directories containing `test_predictions.json`
    gen_runs: Dict[str, Path] = {}
    for pred_f in sorted(rroot.rglob("test_predictions.json")):
        run_tag = pred_f.parent.relative_to(rroot).as_posix()
        gen_runs[run_tag] = pred_f.parent

    gen_preds_map: Dict[str, Dict[str, List[int]]] = {
        tag: json.loads((d / "test_predictions.json").read_text(encoding="utf-8"))
        for tag, d in gen_runs.items()
    }

    session_rel_map: Optional[Dict[str, Dict[int, float]]] = None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if rel_rm_ckpt and Path(rel_rm_ckpt).exists() and gen_preds_map:
        scorer = RelevanceScorer(rel_rm_ckpt, device=device)
        session_rel_map = score_top10_relevance_for_methods(
            scorer, test_sessions, gen_preds_map, item_text_map, batch_size=512
        )
        del scorer
        if device.type == "cuda":
            torch.cuda.empty_cache()

    session_indep_map: Optional[Dict[str, Dict[int, float]]] = None
    has_indep = bool(indep_judge_ckpt and Path(indep_judge_ckpt).exists() and gen_preds_map)
    if has_indep and indep_judge_ckpt is not None:
        indep_scorer = RelevanceScorer(indep_judge_ckpt, device=device)
        session_indep_map = score_top10_relevance_for_methods(
            indep_scorer, test_sessions, gen_preds_map, item_text_map, batch_size=256
        )
        del indep_scorer
        if device.type == "cuda":
            torch.cuda.empty_cache()

    gen_metrics_rescored: Dict[str, Dict[str, Any]] = {}
    for tag, preds in gen_preds_map.items():
        m = evaluate_predictions(
            sessions=test_sessions,
            predictions=preds,
            ks=(1, 5, 10, 20),
            item_to_cat=item_to_cat,
            valid_corpus_ids=valid_corpus_ids,
            session_item_rel_scores=session_rel_map,
        )
        if session_indep_map is not None:
            m_ind = evaluate_predictions(
                sessions=test_sessions,
                predictions=preds,
                ks=(1, 5, 10),
                session_item_rel_scores=session_indep_map,
            )
            for grp_k, grp_v in m_ind.items():
                if grp_k in m:
                    m[grp_k]["IndepRel@10"] = grp_v.get("Rel@10", 0.0)
                    m[grp_k]["IndepStrictRelRate@10"] = grp_v.get("StrictRelRate@10", 0.0)
        summary_f = gen_runs[tag] / "sft_eval_summary.json"
        if not summary_f.is_file():
            summary_f = gen_runs[tag] / "rl_eval_summary.json"
        extra = json.loads(summary_f.read_text(encoding="utf-8")) if summary_f.is_file() else {}
        gen_metrics_rescored[tag] = {"metrics": m, "meta": extra}

    # Load Phase 1 Baselines & Phase 2 SID-Bench if present
    p1_file = rroot / "phase1_baselines_test.json"
    p1_data = json.loads(p1_file.read_text(encoding="utf-8")) if p1_file.is_file() else {}

    # Load per-query baseline predictions if saved (`baseline_preds_test/*.json`) for paired bootstrap!
    baseline_preds_loaded: Dict[str, Dict[str, List[int]]] = {}
    for bname, bval in p1_data.items():
        pf_rel = bval.get("predictions_file")
        if pf_rel:
            pf_cand = rroot / pf_rel
            if pf_cand.is_file():
                baseline_preds_loaded[bname] = json.loads(pf_cand.read_text(encoding="utf-8"))

    # If `indep_judge_ckpt` is provided and any loaded baseline lacks `IndepRel@10`, score it automatically!
    missing_indep_b = {
        bk: bp for bk, bp in baseline_preds_loaded.items()
        if has_indep and "IndepRel@10" not in p1_data[bk].get("metrics", {}).get("overall", {})
    }
    if missing_indep_b and indep_judge_ckpt is not None:
        ind_b_scorer = RelevanceScorer(indep_judge_ckpt, device=device)
        b_ind_map = score_top10_relevance_for_methods(ind_b_scorer, test_sessions, missing_indep_b, item_text_map, batch_size=256)
        del ind_b_scorer
        for bk, bp in missing_indep_b.items():
            m_bi = evaluate_predictions(sessions=test_sessions, predictions=bp, ks=(1, 5, 10), session_item_rel_scores=b_ind_map)
            for grp_k, grp_v in m_bi.items():
                if grp_k in p1_data[bk]["metrics"]:
                    p1_data[bk]["metrics"][grp_k]["IndepRel@10"] = grp_v.get("Rel@10", 0.0)
                    p1_data[bk]["metrics"][grp_k]["IndepStrictRelRate@10"] = grp_v.get("StrictRelRate@10", 0.0)

    sid_file = rroot / "sid_bench_results.json"
    sid_data = json.loads(sid_file.read_text(encoding="utf-8")) if sid_file.is_file() else {}

    sections: List[str] = ["# MiniOneSearch Comprehensive Benchmark Report\n"]

    # Table 1: Main Showdown (Baselines + Key Generative Models)
    t1_entries: List[tuple[str, Mapping[str, Any]]] = []
    for bname, bval in p1_data.items():
        t1_entries.append((bname, bval["metrics"]))
    for tag, gval in gen_metrics_rescored.items():
        t1_entries.append((tag, gval["metrics"]))
    if t1_entries:
        sections.append(_build_comparison_md_table("1. End-to-End Search Benchmark Comparison", t1_entries, has_indep_judge=has_indep))

    # Table 2: SID-Bench
    if sid_data:
        sections.append(render_sid_bench_markdown(sid_data))

    # Table 3: Strictly Separated Batch=1 Online Latency vs Batched Throughput
    lat_lines = ["## 3. Serving Latency Breakdown (`Batch=1` Online vs. Batched Throughput)\n"]
    lat_lines.append("| Engine / Model | Serving Regime | P50 (ms) | P95 (ms) | P99 (ms) | QPS |")
    lat_lines.append("|---|---|---|---|---|---|")
    for bname, bval in p1_data.items():
        l = bval.get("latency", {})
        if "online_b1" in l:
            lb1 = l["online_b1"]
            lat_lines.append(
                f"| **{bname}** | Online Single-Query (`B=1`) | {lb1.get('p50_ms', 0):.2f} | {lb1.get('p95_ms', 0):.2f} | {lb1.get('p99_ms', 0):.2f} | {lb1.get('qps', 0):.1f} |"
            )
        lat_lines.append(
            f"| **{bname}** | Batched Throughput (`B>=64`) | {l.get('p50_ms', 0):.2f} | {l.get('p95_ms', 0):.2f} | {l.get('p99_ms', 0):.2f} | {l.get('qps', 0):.1f} |"
        )
    for tag, gval in gen_metrics_rescored.items():
        meta = gval.get("meta", {})
        blat = meta.get("batched_eval_latency") or meta.get("latency")
        if blat:
            lat_lines.append(
                f"| **{tag}** | Batched Throughput (`Beam@20, B=16`) | {blat.get('p50_ms', 0):.2f} | {blat.get('p95_ms', 0):.2f} | {blat.get('p99_ms', 0):.2f} | {blat.get('qps', 0):.1f} |"
            )
        if "online_single_query_no_kv_cache" in meta:
            l0 = meta["online_single_query_no_kv_cache"]
            l1 = meta["online_single_query_user_prefix_kv_cache"]
            lat_lines.append(
                f"| **{tag}** | Online Single-Query (`HF Beam@10, B=1`, Uncached) | {l0.get('p50_ms', 0):.2f} | {l0.get('p95_ms', 0):.2f} | {l0.get('p99_ms', 0):.2f} | {l0.get('qps', 0):.1f} |"
            )
            lat_lines.append(
                f"| **{tag}** | Online Single-Query (`HF Beam@10, B=1`, +User-Prefix KV-Cache) | {l1.get('p50_ms', 0):.2f} | {l1.get('p95_ms', 0):.2f} | {l1.get('p99_ms', 0):.2f} | {l1.get('qps', 0):.1f} |"
            )
    sections.append("\n".join(lat_lines) + "\n")

    # Table 4: User-Clustered Paired Bootstrap Significance Tests (10,000 Resamples + Holm-Bonferroni)
    paired_stats: List[Dict[str, Any]] = []
    fusion_key = next((k for k in baseline_preds_loaded if "Score-Fusion@100" in k), None)
    comparisons_to_run = []
    if fusion_key and gen_preds_map:
        for gtag, gpreds in gen_preds_map.items():
            comparisons_to_run.append((f"{gtag} vs. {fusion_key}", gpreds, baseline_preds_loaded[fusion_key]))
    if len(gen_preds_map) >= 2:
        gkeys = sorted(gen_preds_map.keys())
        sft_ref = next((k for k in gkeys if "sdft" in k.lower() or "s2" in k.lower()), gkeys[0])
        for gtag in gkeys:
            if gtag != sft_ref:
                comparisons_to_run.append((f"{gtag} vs. {sft_ref}", gen_preds_map[gtag], gen_preds_map[sft_ref]))

    if comparisons_to_run:
        paired_stats = run_hypothesis_family(test_sessions, comparisons_to_run, metric_key="NDCG@10", n_boot=10_000, seed=42)
        st_lines = ["## 4. User-Clustered Paired Bootstrap Significance (`NDCG@10`, 10,000 Resamples)\n"]
        st_lines.append("| Comparison (`A - B`) | Δ NDCG@10 | 95% Cluster CI | Raw p (Cluster) | Holm-Adjusted p | Users / Sessions |")
        st_lines.append("|---|---:|---|---:|---:|---:|")
        for r in paired_stats:
            st_lines.append(
                f"| **{r['comparison']}** | {r['diff'] * 100:+.2f}pt | "
                f"`[{r['ci95_low_cluster'] * 100:+.2f}pt, {r['ci95_high_cluster'] * 100:+.2f}pt]` | "
                f"{r['p_two_sided_cluster']:.4f} | {r['p_holm_cluster']:.4f} | "
                f"{int(r['num_users'])} / {int(r['num_sessions'])} |"
            )
        sections.append("\n".join(st_lines) + "\n")

    full_md = "\n".join(sections)
    (opath / "BENCHMARK_REPORT.md").write_text(full_md, encoding="utf-8")
    compiled = {
        "phase1_baselines": p1_data,
        "phase2_sid_bench": sid_data,
        "generative_runs": gen_metrics_rescored,
        "paired_bootstrap_tests": paired_stats,
    }
    (opath / "compiled_all_results.json").write_text(
        json.dumps(compiled, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("\n" + full_md, flush=True)
    return compiled


def main() -> None:
    p = argparse.ArgumentParser(description="Compile unified MiniOneSearch Benchmark Report")
    p.add_argument("--bench_dir", required=True)
    p.add_argument("--results_root", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--rel_rm_ckpt", default="")
    p.add_argument("--indep_judge_ckpt", default="")
    a = p.parse_args()

    compile_full_report(
        bench_dir=a.bench_dir,
        results_root=a.results_root,
        out_dir=a.out_dir,
        rel_rm_ckpt=a.rel_rm_ckpt or None,
        indep_judge_ckpt=a.indep_judge_ckpt or None,
    )


if __name__ == "__main__":
    main()
