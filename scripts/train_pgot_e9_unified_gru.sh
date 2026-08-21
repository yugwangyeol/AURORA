#!/usr/bin/env bash
# E9-A: update OVT/register hidden states directly with Slot-style attention
# and an explicit FP32 GRU at layers 21, 24, and 27.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export MODEL_PATH="${MODEL_PATH:-${PROJECT_ROOT}/checkpoints/pgot_e8_2_paired_causal/checkpoint-10000}"
export OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/checkpoints/pgot_e9_unified_gru}"
export WANDB_NAME="${WANDB_NAME:-pgot_e9_unified_gru}"

export E8_UPDATE_MODE=unified_gru
export E8_LAYERS="${E8_LAYERS:-21,24,27}"
export E9_UPDATE_DIM="${E9_UPDATE_DIM:-512}"
export E9_MLP_RATIO="${E9_MLP_RATIO:-2.0}"

# Keep E8.1-A ownership/Reader supervision and E8.2 paired-causal weights fixed
# so the controlled variable is the separate-memory -> unified-OVT updater.
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

export NUM_GPUS="${NUM_GPUS:-2}"
export PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-2}"
export GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-32}"
export MAX_STEPS="${MAX_STEPS:-10000}"

exec bash "${PROJECT_ROOT}/scripts/train_pgot_e8_visual_memory.sh"
