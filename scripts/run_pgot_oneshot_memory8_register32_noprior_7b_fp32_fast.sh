#!/usr/bin/env bash
# Same O8/R32 baseline, global batch 32; faster four-GPU FP32 execution only.
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export NUM_GPUS=4 PER_DEVICE_TRAIN_BATCH_SIZE=4
export GLOBAL_BATCH_SIZE=32 GRADIENT_ACCUMULATION_STEPS=2
export FSDP_CONFIG="${PROJECT_ROOT}/configs/fsdp_7b_full_shard_post.json"
export TRAIN_ENTRYPOINT="${PROJECT_ROOT}/train.py" PGOT_SKIP_FINAL_SAVE=0
export OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/checkpoints/pgot_oneshot_memory8_register32_noprior_7b_fp32_fast}"
export EVAL_ROOT="${EVAL_ROOT:-${PROJECT_ROOT}/outputs/eval_pgot_oneshot_memory8_register32_noprior_7b_fp32_fast}"
export WANDB_NAME="${WANDB_NAME:-pgot_oneshot_memory8_register32_noprior_7b_fp32_fast}"
if [[ -s "${EVAL_ROOT}/train.log" ]]; then
    echo "Training log already exists; choose a fresh OUTPUT_DIR and EVAL_ROOT." >&2
    exit 2
fi
exec bash "${PROJECT_ROOT}/scripts/run_pgot_oneshot_memory8_register32_noprior_7b_fp32.sh"
