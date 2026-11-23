#!/usr/bin/env bash
# Open-Source Reproducible Entrypoint: Phase 1 Data Curation + Relevance RM + Search Baselines
# Supports single-GPU or multi-GPU (`NPROC_PER_NODE=4 bash scripts/01_prepare_and_baselines.sh`)
set -euo pipefail

RAW_DIR=${RAW_DIR:-./data/kuaisearch_raw}
TIER=${TIER:-mini}
BENCH_DIR=${BENCH_DIR:-./data/kuaisearch_${TIER}}
OUT_DIR=${OUT_DIR:-./outputs/${TIER}/phase1}
NPROC=${NPROC_PER_NODE:-$(python -c "import torch; print(max(1, torch.cuda.device_count()))")}

echo "[Phase 1] TIER=${TIER} | GPUs=${NPROC} | RAW_DIR=${RAW_DIR} | BENCH_DIR=${BENCH_DIR} | OUT_DIR=${OUT_DIR}"

# 1. Download & Build Chronological KuaiSearch Benchmark (`core5`, `mini`, or `scale`)
if [ ! -f "${BENCH_DIR}/stats.json" ]; then
  python -m minionesearch.data.kuaisearch_builder \
    --raw_dir "${RAW_DIR}" \
    --out_dir "${BENCH_DIR}" \
    --download \
    --tier "${TIER}" \
    --seed 42
fi

# 2. Train 4-Tier Relevance Cross-Encoder Judge & Held-Out Independent Qwen3 Referee
if [ ! -f "${OUT_DIR}/relevance_rm/relevance_rm_metrics.json" ]; then
  torchrun --nproc_per_node="${NPROC}" --master_port=29511 \
    -m minionesearch.baselines.relevance_rm \
    --rel_dir "${BENCH_DIR}/relevance" \
    --out_dir "${OUT_DIR}/relevance_rm" \
    --base_model "BAAI/bge-reranker-v2-m3" \
    --epochs 3 \
    --batch_size 64 \
    --lr 3e-5
fi

if [ ! -d "${OUT_DIR}/indep_qwen3_judge/best_model" ]; then
  python -m minionesearch.baselines.score_fusion_and_indep_judge \
    --mode train_indep_judge \
    --rel_dir "${BENCH_DIR}/relevance" \
    --judge_out_dir "${OUT_DIR}/indep_qwen3_judge" \
    --judge_base_model "Qwen/Qwen3-0.6B-Base"
fi

# 3. Train Fine-Tuned Query-Only & Personalized BGE-M3 Dual-Encoders
if [ ! -d "${OUT_DIR}/ft_bge_m3_query/encoder_ckpt" ]; then
  torchrun --nproc_per_node="${NPROC}" --master_port=29512 \
    -m minionesearch.baselines.dense_retriever \
    --bench_dir "${BENCH_DIR}" \
    --out_dir "${OUT_DIR}/ft_bge_m3_query" \
    --base_model "BAAI/bge-m3" \
    --personalized 0 \
    --epochs 3 \
    --batch_size 64 \
    --lr 2e-5
fi

if [ ! -d "${OUT_DIR}/ft_bge_m3_pers/encoder_ckpt" ]; then
  torchrun --nproc_per_node="${NPROC}" --master_port=29513 \
    -m minionesearch.baselines.dense_retriever \
    --bench_dir "${BENCH_DIR}" \
    --out_dir "${OUT_DIR}/ft_bge_m3_pers" \
    --base_model "BAAI/bge-m3" \
    --personalized 1 \
    --epochs 3 \
    --batch_size 64 \
    --lr 2e-5
fi

# 4. Run Unified Head-to-Head Evaluation + Validation-Tuned Score-Fusion@100
python -m minionesearch.baselines.run_phase1_eval \
  --bench_dir "${BENCH_DIR}" \
  --out_dir "${OUT_DIR}/eval" \
  --zeroshot_model "BAAI/bge-m3" \
  --ft_query_ckpt "${OUT_DIR}/ft_bge_m3_query/encoder_ckpt" \
  --ft_pers_ckpt "${OUT_DIR}/ft_bge_m3_pers/encoder_ckpt" \
  --rel_rm_ckpt "${OUT_DIR}/relevance_rm/best_model" \
  --split test \
  --save_zeroshot_corpus_emb "${BENCH_DIR}/corpus_bge_m3.npy"

python -m minionesearch.baselines.score_fusion_and_indep_judge \
  --mode fusion \
  --bench_dir "${BENCH_DIR}" \
  --ft_query_ckpt "${OUT_DIR}/ft_bge_m3_query/encoder_ckpt" \
  --ft_pers_ckpt "${OUT_DIR}/ft_bge_m3_pers/encoder_ckpt" \
  --rel_rm_ckpt "${OUT_DIR}/relevance_rm/best_model" \
  --indep_judge_ckpt "${OUT_DIR}/indep_qwen3_judge/best_model" \
  --existing_phase1_json "${OUT_DIR}/eval/phase1_baselines_test.json" \
  --out_phase1_json "${OUT_DIR}/eval/phase1_baselines_test.json"
