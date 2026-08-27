#!/usr/bin/env bash
# E10-R / E10-A: E8.2 routing with raw source-SigLIP visual values.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export MODEL_PATH="${MODEL_PATH:-${PROJECT_ROOT}/checkpoints/pgot_e8_2_paired_causal/checkpoint-10000}"
export OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/checkpoints/pgot_e10_r_raw_value}"
export WANDB_NAME="${WANDB_NAME:-pgot_e10_r_raw_value}"

# Preserve the successful E8.1-A/E8.2 routing recipe.  E10-R changes only
# the Writer value source and uses reconstruction training without causality.
export E8_UPDATE_MODE=separate_memory
export E10_RAW_VALUE_ENABLE=True
export E8_OWNER_WEIGHT="${E8_OWNER_WEIGHT:-1.0}"
export E8_READER_SUPERVISION_MODE=writer
export E8_READER_OBJECT_WEIGHT="${E8_READER_OBJECT_WEIGHT:-0.5}"
export E8_READER_BACKGROUND_WEIGHT="${E8_READER_BACKGROUND_WEIGHT:-0.25}"
export E8_CAUSAL_ENABLE=False
export E8_NEED_WEIGHT=0.0
export E8_LOCAL_WEIGHT=0.0
export E8_REGISTER_BG_WEIGHT=0.0
export E8_REGISTER_FG_WEIGHT=0.0

export NUM_GPUS="${NUM_GPUS:-2}"
export PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-6}"
export GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-24}"
export MAX_STEPS="${MAX_STEPS:-10000}"

exec bash "${PROJECT_ROOT}/scripts/train_pgot_e8_visual_memory.sh"
