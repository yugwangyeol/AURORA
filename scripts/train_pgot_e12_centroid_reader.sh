#!/usr/bin/env bash
# E12: E11 Dual-M4 with Writer-predicted visual-memory centroids in Reader keys.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export MODEL_PATH="${MODEL_PATH:-${PROJECT_ROOT}/checkpoints/pgot_e11_dual_m4/checkpoint-10000}"
export OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/checkpoints/pgot_e12_centroid_reader}"
export WANDB_NAME="${WANDB_NAME:-pgot_e12_centroid_reader}"

# Preserve E11 exactly.  E12 changes only the Reader key by adding the
# dynamically predicted centroid of each existing visual memory.
export E8_UPDATE_MODE=separate_memory
export E10_RAW_VALUE_ENABLE=True
export E11_DUAL_M4_ENABLE=True
export E11_MEMORIES_PER_OWNER=4
export E12_CENTROID_READER_ENABLE=True
export E12_CENTROID_GATE_INIT="${E12_CENTROID_GATE_INIT:-0.0}"
export E8_N_REGISTER=4
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
