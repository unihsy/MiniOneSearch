#!/usr/bin/env bash
# Open-Source Reproducible Entrypoint: Phase 2 Search-SID Tokenizer Zoo & SID-Bench
set -euo pipefail

TIER=${TIER:-mini}
BENCH_DIR=${BENCH_DIR:-./data/kuaisearch_${TIER}}
EMB_PATH=${EMB_PATH:-${BENCH_DIR}/corpus_bge_m3.npy}
ALIGNED_DIR=${ALIGNED_DIR:-./outputs/${TIER}/aligned_5loss}
OUT_DIR=${OUT_DIR:-./outputs/${TIER}/sid_zoo}
if [ -z "${CODEBOOKS:-}" ]; then
  case "${TIER}" in
    scale) CODEBOOKS="512,512,256,256,256" ;;
    *)     CODEBOOKS="512,256,256,256,256" ;;
  esac
fi
NPROC=${NPROC_PER_NODE:-$(python -c "import torch; print(max(1, torch.cuda.device_count()))")}

if [ ! -d "${ALIGNED_DIR}/encoder_ckpt" ]; then
  torchrun --nproc_per_node="${NPROC}" --master_port=29521 \
    -m minionesearch.tokenizer.aligned_encoder \
    --bench_dir "${BENCH_DIR}" \
    --out_dir "${ALIGNED_DIR}" \
    --base_model "BAAI/bge-m3" \
    --epochs 3 \
    --batch_size 48 \
    --lr 2e-5
fi

python -m minionesearch.tokenizer.run_sid_zoo \
  --bench_dir "${BENCH_DIR}" \
  --raw_emb_path "${EMB_PATH}" \
  --encoder_ckpt "${ALIGNED_DIR}/encoder_ckpt" \
  --out_dir "${OUT_DIR}" \
  --codebook_sizes "${CODEBOOKS}" \
  --rqvae_epochs 45 \
  --kmeans_iters 30 \
  --seed 42
