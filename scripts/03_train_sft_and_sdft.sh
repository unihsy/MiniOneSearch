#!/usr/bin/env bash
# Open-Source Reproducible Entrypoint: Phase 3 Progressive SFT + Joint SDFT
set -euo pipefail

TIER=${TIER:-mini}
BENCH_DIR=${BENCH_DIR:-./data/kuaisearch_${TIER}}
SID_DIR=${SID_DIR:-./outputs/${TIER}/sid_zoo/khqe_rq_opq_dedup}
OUT_DIR=${OUT_DIR:-./outputs/${TIER}/runs/sft_qwen3_1.7b_sdft}
BASE_MODEL=${BASE_MODEL:-Qwen/Qwen3-1.7B-Base}
NPROC=${NPROC_PER_NODE:-$(python -c "import torch; print(max(1, torch.cuda.device_count()))")}

torchrun --nproc_per_node="${NPROC}" --master_port=29531 \
  -m minionesearch.sft.sft_trainer \
  --bench_dir "${BENCH_DIR}" \
  --sid_dir "${SID_DIR}" \
  --out_dir "${OUT_DIR}" \
  --base_model "${BASE_MODEL}" \
  --use_progressive_stages 1 \
  --use_joint_sdft 1 \
  --personalized 1 \
  --user_id_mode "mu_seq" \
  --use_query_sid 0 \
  --stage12_epochs 2 \
  --stage3_epochs 3 \
  --batch_size 32 \
  --lr 1.5e-4 \
  --eval_beams 20 \
  --seed 42
