#!/usr/bin/env bash
# Final (semantic + E11 ID) write; soft owner routing then memory attention.
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export MEMORY_KEY_MODE="${MEMORY_KEY_MODE:-content}"
case "${MEMORY_KEY_MODE}" in content|id) ;; *) echo "MEMORY_KEY_MODE must be content or id" >&2; exit 2;; esac
export MODEL_PATH="${MODEL_PATH:-${PROJECT_ROOT}/checkpoints/pgot_e11_capacity_dit16/checkpoint-10000}"
export OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/checkpoints/pgot_oneshot_memory4_${MEMORY_KEY_MODE}}"
export WANDB_NAME="${WANDB_NAME:-pgot_oneshot_memory4_${MEMORY_KEY_MODE}}"
export ONE_SHOT_READER_ENABLE=True
export ONE_SHOT_READOUT_MODE="memory_${MEMORY_KEY_MODE}"
export ONE_SHOT_DETACH_OWNER_ROUTING=True
export E8_UPDATE_MODE=separate_memory
export E10_RAW_VALUE_ENABLE=True
export E11_DUAL_M4_ENABLE=False
export E11_MEMORIES_PER_OWNER="${E11_MEMORIES_PER_OWNER:-16}"
export E11_OBJECT_MEMORIES_PER_OWNER="${E11_OBJECT_MEMORIES_PER_OWNER:-4}"
export E11_REGISTER_MEMORIES_PER_OWNER="${E11_REGISTER_MEMORIES_PER_OWNER:-16}"
export E11_QUERY_SEPARATION_ENABLE=False
export E12_CENTROID_READER_ENABLE=False
export E8_N_REGISTER="${E8_N_REGISTER:-4}"
export E8_READER_LAYERS=1
export E8_OWNER_WEIGHT=1.0
export E8_READER_SUPERVISION_MODE=writer
export E8_READER_OBJECT_WEIGHT=0.5
export E8_READER_BACKGROUND_WEIGHT=0.25
export E8_CAUSAL_ENABLE=False
export PGOT_DIT_OVT_XATTN_ENABLE=False
export PGOT_DIT_SOFT_ROUTING_ENABLE=False
export DIT_UNFREEZE_LAST_N=16
export NUM_GPUS="${NUM_GPUS:-2}"
export PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-2}"
export GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-24}"
export MAX_STEPS="${MAX_STEPS:-5000}"
export SAVE_ONLY_MODEL="${SAVE_ONLY_MODEL:-True}"
export SAVE_STEPS="${SAVE_STEPS:-1000}"
export SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-1}"
exec bash "${PROJECT_ROOT}/scripts/train_pgot_e8_visual_memory.sh"
