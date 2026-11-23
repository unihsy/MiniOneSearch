#!/usr/bin/env bash
# Open-Source Reproducible Entrypoint: Phase 4 Search-RL Post-Training (SP-TPMA-GRPO)
set -euo pipefail

TIER=${TIER:-mini}
BENCH_DIR=${BENCH_DIR:-./data/kuaisearch_${TIER}}
SID_DIR=${SID_DIR:-./outputs/${TIER}/sid_zoo/khqe_rq_opq_dedup}
SFT_CKPT=${SFT_CKPT:-./outputs/${TIER}/runs/sft_qwen3_1.7b_sdft/checkpoint}
REL_RM_CKPT=${REL_RM_CKPT:-./outputs/${TIER}/phase1/relevance_rm/best_model}
ALGO=${ALGO:-sp_tpma_grpo}
INJECT_GT=${INJECT_GT:-0}
SFT_ANCHOR=${SFT_ANCHOR:-0.00}
OUT_DIR=${OUT_DIR:-./outputs/${TIER}/runs/rl_${ALGO}}
NPROC=${NPROC_PER_NODE:-$(python -c "import torch; print(max(1, torch.cuda.device_count()))")}

torchrun --nproc_per_node="${NPROC}" --master_port=29541 \
  -m minionesearch.rl.rl_trainer \
  --bench_dir "${BENCH_DIR}" \
  --sid_dir "${SID_DIR}" \
  --sft_ckpt_dir "${SFT_CKPT}" \
  --out_dir "${OUT_DIR}" \
  --rel_rm_ckpt "${REL_RM_CKPT}" \
  --algo "${ALGO}" \
  --num_generations 12 \
  --rollout_mode hybrid \
  --max_rl_sessions 8000 \
  --epochs 1 \
  --ppo_epochs 2 \
  --use_ppo_clip 1 \
  --inject_gt_anchor "${INJECT_GT}" \
  --w_item 0.50 \
  --batch_size 8 \
  --lr 5.0e-6 \
  --kl_beta 0.03 \
  --sft_anchor_weight "${SFT_ANCHOR}" \
  --eval_beams 20 \
  --seed 42
