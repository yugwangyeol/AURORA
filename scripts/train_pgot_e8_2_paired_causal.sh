#!/usr/bin/env bash
# E8.2: fine-tune an E8.1 checkpoint with paired causal reconstruction.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# E8.2 is a causal fine-tune of E8.1-A: Writer uses GT ownership and the
# Reader is supervised by the detached predicted Writer map. Keep those
# choices explicit so the base launcher cannot silently switch variants.
export MODEL_PATH="${MODEL_PATH:-${PROJECT_ROOT}/checkpoints/pgot_e8_ablation_a_writer_gt_reader_writer/checkpoint-10000}"
export OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/checkpoints/pgot_e8_2_paired_causal}"
export WANDB_NAME="${WANDB_NAME:-pgot_e8_2_paired_causal}"
export E8_OWNER_WEIGHT="${E8_OWNER_WEIGHT:-1.0}"
export E8_READER_SUPERVISION_MODE="${E8_READER_SUPERVISION_MODE:-writer}"
export E8_READER_OBJECT_WEIGHT="${E8_READER_OBJECT_WEIGHT:-0.5}"
export E8_READER_BACKGROUND_WEIGHT="${E8_READER_BACKGROUND_WEIGHT:-0.25}"
export E8_CAUSAL_ENABLE=True
export E8_CAUSAL_BATCH_PROBABILITY="${E8_CAUSAL_BATCH_PROBABILITY:-0.25}"
export E8_CAUSAL_RAMP_STEPS="${E8_CAUSAL_RAMP_STEPS:-1000}"
export E8_CAUSAL_MARGIN="${E8_CAUSAL_MARGIN:-0.05}"
export E8_REGISTER_MARGIN="${E8_REGISTER_MARGIN:-0.05}"
export E8_NEED_WEIGHT="${E8_NEED_WEIGHT:-0.1}"
export E8_LOCAL_WEIGHT="${E8_LOCAL_WEIGHT:-0.05}"
export E8_REGISTER_BG_WEIGHT="${E8_REGISTER_BG_WEIGHT:-0.1}"
export E8_REGISTER_FG_WEIGHT="${E8_REGISTER_FG_WEIGHT:-0.1}"
export MAX_STEPS="${MAX_STEPS:-5000}"

# The active paired branch concatenates full/object-ablated/register-only DiT
# batches.  Lower the micro-batch and recover the same global batch by gradient
# accumulation on the two GPUs.
export NUM_GPUS="${NUM_GPUS:-2}"
export PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-2}"
export GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-32}"

exec bash "${PROJECT_ROOT}/scripts/train_pgot_e8_visual_memory.sh"
