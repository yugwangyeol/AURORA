#!/usr/bin/env bash
# E11 Capacity/DiT16 + L_local and L_reg-bg only.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export MODEL_PATH="${MODEL_PATH:-${PROJECT_ROOT}/checkpoints/pgot_e11_capacity_dit16/checkpoint-10000}"
export OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/checkpoints/pgot_e11_causal_only}"
export WANDB_NAME="${WANDB_NAME:-pgot_e11_causal_only}"

export E8_UPDATE_MODE=separate_memory
export E10_RAW_VALUE_ENABLE=True
export E11_DUAL_M4_ENABLE=True
export E11_MEMORIES_PER_OWNER=4
export E11_OBJECT_MEMORIES_PER_OWNER=8
export E11_REGISTER_MEMORIES_PER_OWNER=16
export E11_QUERY_SEPARATION_ENABLE=False
export E12_CENTROID_READER_ENABLE=False
export E8_N_REGISTER=4
export E8_READER_LAYERS=1
export E8_OWNER_WEIGHT="${E8_OWNER_WEIGHT:-1.0}"
export E8_READER_SUPERVISION_MODE=writer
export E8_READER_OBJECT_WEIGHT="${E8_READER_OBJECT_WEIGHT:-0.5}"
export E8_READER_BACKGROUND_WEIGHT="${E8_READER_BACKGROUND_WEIGHT:-0.25}"

# Keep only the two causal terms agreed for this comparison.
export E8_CAUSAL_ENABLE=True
export E8_NEED_WEIGHT=0.0
export E8_LOCAL_WEIGHT="${E8_LOCAL_WEIGHT:-0.05}"
export E8_REGISTER_BG_WEIGHT="${E8_REGISTER_BG_WEIGHT:-0.1}"
export E8_REGISTER_FG_WEIGHT=0.0
export E8_CAUSAL_BATCH_PROBABILITY="${E8_CAUSAL_BATCH_PROBABILITY:-0.25}"
export E8_CAUSAL_RAMP_STEPS="${E8_CAUSAL_RAMP_STEPS:-1000}"

export PGOT_DIT_OVT_XATTN_ENABLE="${PGOT_DIT_OVT_XATTN_ENABLE:-False}"
export PGOT_DIT_SOFT_ROUTING_ENABLE="${PGOT_DIT_SOFT_ROUTING_ENABLE:-False}"
export DIT_UNFREEZE_LAST_N=16
export NUM_GPUS="${NUM_GPUS:-2}"
export PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-6}"
export GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-24}"
export MAX_STEPS="${MAX_STEPS:-5000}"
# Three sequential FP32 models do not fit with Adam states on the current
# volume.  Preserve loadable model checkpoints while omitting resume-only
# optimizer/scheduler/RNG state for these 5k pilot runs.
export SAVE_ONLY_MODEL="${SAVE_ONLY_MODEL:-True}"
export SAVE_STEPS="${SAVE_STEPS:-1000}"
export SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-1}"

exec bash "${PROJECT_ROOT}/scripts/train_pgot_e8_visual_memory.sh"
