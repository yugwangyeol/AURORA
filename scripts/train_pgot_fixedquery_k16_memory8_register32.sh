#!/usr/bin/env bash
# Matched final-stage structural baseline for DeOR O8/R32.
# Same parent checkpoint, losses, Writer/Reader/DiT, batch, and 5k steps as
# Full DeOR; only variable AR object owners become K=16 learned queries.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export MODEL_PATH="${MODEL_PATH:-${PROJECT_ROOT}/checkpoints/pgot_e11_capacity_dit16/checkpoint-10000}"
export OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/checkpoints/pgot_fixedquery_k16_memory8_register32}"
export WANDB_NAME="${WANDB_NAME:-pgot_fixedquery_k16_memory8_register32}"
export FIXED_QUERY_OWNER_ENABLE=True
export FIXED_QUERY_OWNER_COUNT=16
export MEMORY_KEY_MODE=content
export E11_MEMORIES_PER_OWNER=32
export E11_OBJECT_MEMORIES_PER_OWNER=8
export E11_REGISTER_MEMORIES_PER_OWNER=32
export E8_N_REGISTER=4
export ONE_SHOT_WRITER_SOFTMAX_AXIS=patch
export ONE_SHOT_WRITER_OWNER_PRIOR=False
export E8_READER_SUPERVISION_MODE=writer
export NUM_GPUS="${NUM_GPUS:-2}"
export PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-2}"
export GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-24}"
export MAX_STEPS="${MAX_STEPS:-5000}"

exec bash "${PROJECT_ROOT}/scripts/train_pgot_oneshot_memory4.sh"
