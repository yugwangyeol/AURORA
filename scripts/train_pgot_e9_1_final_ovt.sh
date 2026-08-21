#!/usr/bin/env bash
# E9.1: final unified OVT/register states are the reconstruction bottleneck.
# Slot-style FP32 GRU writes occur at layers 20, 23, and 26; layer 27 then
# produces the exact final tokens used as both Reader keys and values.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export MODEL_PATH="${MODEL_PATH:-${PROJECT_ROOT}/checkpoints/pgot_e9_unified_gru/checkpoint-10000}"
export OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/checkpoints/pgot_e9_1_final_ovt}"
export WANDB_NAME="${WANDB_NAME:-pgot_e9_1_final_ovt}"

export E8_UPDATE_MODE=final_ovt
export E8_LAYERS="${E8_LAYERS:-20,23,26}"
export E9_UPDATE_DIM="${E9_UPDATE_DIM:-512}"
export E9_MLP_RATIO="${E9_MLP_RATIO:-2.0}"

# E8.1-A routing supervision: GT masks supervise Writer ownership; Reader
# routing is distilled from the detached final Writer prediction, not GT.
export E8_OWNER_WEIGHT="${E8_OWNER_WEIGHT:-1.0}"
export E8_READER_SUPERVISION_MODE="${E8_READER_SUPERVISION_MODE:-writer}"
export E8_READER_OBJECT_WEIGHT="${E8_READER_OBJECT_WEIGHT:-0.5}"
export E8_READER_BACKGROUND_WEIGHT="${E8_READER_BACKGROUND_WEIGHT:-0.25}"

# Keep E8.2 paired causal training, but rebalance the register terms that
# dominated E8.2/E9.  Object need/local remain the principal causal signals.
export E8_CAUSAL_ENABLE=True
export E8_CAUSAL_BATCH_PROBABILITY="${E8_CAUSAL_BATCH_PROBABILITY:-0.25}"
export E8_CAUSAL_RAMP_STEPS="${E8_CAUSAL_RAMP_STEPS:-1000}"
export E8_CAUSAL_MARGIN="${E8_CAUSAL_MARGIN:-0.05}"
export E8_REGISTER_MARGIN="${E8_REGISTER_MARGIN:-0.05}"
export E8_NEED_WEIGHT="${E8_NEED_WEIGHT:-0.1}"
export E8_LOCAL_WEIGHT="${E8_LOCAL_WEIGHT:-0.05}"
export E8_REGISTER_BG_WEIGHT="${E8_REGISTER_BG_WEIGHT:-0.005}"
export E8_REGISTER_FG_WEIGHT="${E8_REGISTER_FG_WEIGHT:-0.01}"

# Two GPUs, FP32, effective global batch 24 (= 2 GPUs x 6 x accumulation 2).
export NUM_GPUS="${NUM_GPUS:-2}"
export PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-6}"
export GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-24}"
export MAX_STEPS="${MAX_STEPS:-10000}"

exec bash "${PROJECT_ROOT}/scripts/train_pgot_e8_visual_memory.sh"
