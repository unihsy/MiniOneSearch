#!/usr/bin/env bash
# Open-Source Reproducible Entrypoint: Parameter Scaling Law (Qwen3 0.6B -> 1.7B -> 4B -> 8B)
set -euo pipefail

TIER=${TIER:-mini}
BENCH_DIR=${BENCH_DIR:-./data/kuaisearch_${TIER}}
SID_DIR=${SID_DIR:-./outputs/${TIER}/sid_zoo/khqe_rq_opq_dedup}
ROOT_OUT=${ROOT_OUT:-./outputs/${TIER}/runs/scaling_law}
NPROC=${NPROC_PER_NODE:-$(python -c "import torch; print(max(1, torch.cuda.device_count()))")}

for SPEC in "0.6B:Qwen/Qwen3-0.6B-Base:32:1:2.0e-4" \
            "1.7B:Qwen/Qwen3-1.7B-Base:32:1:1.5e-4" \
            "4B:Qwen/Qwen3-4B-Base:16:2:1.0e-4" \
            "8B:Qwen/Qwen3-8B-Base:8:4:8.0e-5"; do
  IFS=":" read -r TAG MODEL BS ACC LR <<< "${SPEC}"
  echo "[ScalingLaw] Running ${TAG} (${MODEL}) on TIER=${TIER}..."
  torchrun --nproc_per_node="${NPROC}" --master_port=29551 \
    -m minionesearch.sft.sft_trainer \
    --bench_dir "${BENCH_DIR}" \
    --sid_dir "${SID_DIR}" \
    --out_dir "${ROOT_OUT}/qwen3_${TAG}" \
    --base_model "${MODEL}" \
    --use_progressive_stages 1 \
    --use_joint_sdft 1 \
    --personalized 1 \
    --user_id_mode "mu_seq" \
    --use_query_sid 0 \
    --stage12_epochs 2 \
    --stage3_epochs 3 \
    --batch_size "${BS}" \
    --grad_accum_steps "${ACC}" \
    --lr "${LR}" \
    --eval_beams 20 \
    --seed 42
done
