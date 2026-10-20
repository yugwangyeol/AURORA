#!/usr/bin/env bash
# O4/R32 object-memory ablation initialized from the same PGOT checkpoint as O8/O16.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export MODEL_PATH="${MODEL_PATH:-${PROJECT_ROOT}/checkpoints/pgot_e11_capacity_dit16/checkpoint-10000}"
export OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/checkpoints/pgot_oneshot_memory4_register32_noprior}"
export WANDB_NAME="${WANDB_NAME:-pgot_oneshot_memory4_register32_noprior}"

# Keep the O8/R32 and O16/R32 training recipe fixed; change only object memory.
export MEMORY_KEY_MODE=content
export ONE_SHOT_SCRATCH_INIT=False
export ONE_SHOT_DIRECT_RAE_QUERY=False
export ONE_SHOT_RAE_QUERY_ADAPTER_ENABLE=False
export ONE_SHOT_MEMORY_CONTRASTIVE_ENABLE=False
export ONE_SHOT_MEMORY_VALUE_SOURCE=siglip
export PGOT_CONTRASTIVE_TARGET_WEIGHT=0.0
export E11_MEMORIES_PER_OWNER=32
export E11_OBJECT_MEMORIES_PER_OWNER=4
export E11_REGISTER_MEMORIES_PER_OWNER=32
export E8_N_REGISTER=4
export ONE_SHOT_WRITER_SOFTMAX_AXIS=patch
export ONE_SHOT_WRITER_OWNER_PRIOR=False

export NUM_GPUS=2
export PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-2}"
export GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-24}"
export MAX_STEPS="${MAX_STEPS:-5000}"
export MASTER_PORT="${MASTER_PORT:-29714}"
export SAVE_ONLY_MODEL="${SAVE_ONLY_MODEL:-True}"
export SAVE_STEPS="${SAVE_STEPS:-1000}"
export SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-1}"

exec bash "${PROJECT_ROOT}/scripts/train_pgot_oneshot_memory4.sh"
