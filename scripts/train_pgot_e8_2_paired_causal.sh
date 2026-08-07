#!/usr/bin/env bash
# E8.2: fine-tune an E8.1 checkpoint with paired causal reconstruction.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export MODEL_PATH="${MODEL_PATH:-${PROJECT_ROOT}/checkpoints/pgot_e8_1_clean/checkpoint-10000}"
export OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/checkpoints/pgot_e8_2_paired_causal}"
export WANDB_NAME="${WANDB_NAME:-pgot_e8_2_paired_causal}"
export E8_CAUSAL_ENABLE=True
export E8_CAUSAL_BATCH_PROBABILITY="${E8_CAUSAL_BATCH_PROBABILITY:-0.25}"
export E8_CAUSAL_RAMP_STEPS="${E8_CAUSAL_RAMP_STEPS:-1000}"
export MAX_STEPS="${MAX_STEPS:-5000}"

# The active paired branch concatenates full/object-ablated/register-only DiT
# batches.  Lower the micro-batch and recover the same global batch by gradient
# accumulation on the two GPUs.
export NUM_GPUS="${NUM_GPUS:-2}"
export PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-2}"
export GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-32}"

exec bash "${PROJECT_ROOT}/scripts/train_pgot_e8_visual_memory.sh"
