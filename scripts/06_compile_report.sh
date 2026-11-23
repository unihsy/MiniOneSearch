#!/usr/bin/env bash
# Open-Source Reproducible Entrypoint: Phase 5 Unified Benchmark Report Compilation
set -euo pipefail

TIER=${TIER:-mini}
BENCH_DIR=${BENCH_DIR:-./data/kuaisearch_${TIER}}
RESULTS_ROOT=${RESULTS_ROOT:-./outputs/${TIER}/runs}
OUT_DIR=${OUT_DIR:-./outputs/${TIER}/report}
REL_RM_CKPT=${REL_RM_CKPT:-./outputs/${TIER}/phase1/relevance_rm/best_model}
INDEP_JUDGE_CKPT=${INDEP_JUDGE_CKPT:-./outputs/${TIER}/phase1/indep_qwen3_judge/best_model}

mkdir -p "${RESULTS_ROOT}"
cp -f "./outputs/${TIER}/phase1/eval/phase1_baselines_test.json" "${RESULTS_ROOT}/phase1_baselines_test.json" 2>/dev/null || true
cp -ru "./outputs/${TIER}/phase1/eval/baseline_preds_test" "${RESULTS_ROOT}/" 2>/dev/null || true
cp -f "./outputs/${TIER}/sid_zoo/sid_bench_results.json" "${RESULTS_ROOT}/sid_bench_results.json" 2>/dev/null || true

python -m minionesearch.evaluation.report_compiler \
  --bench_dir "${BENCH_DIR}" \
  --results_root "${RESULTS_ROOT}" \
  --out_dir "${OUT_DIR}" \
  --rel_rm_ckpt "${REL_RM_CKPT}" \
  --indep_judge_ckpt "${INDEP_JUDGE_CKPT}"
