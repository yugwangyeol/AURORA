#!/usr/bin/env bash
# Capacity experiment: 8 memories/object and 32 memories/register, without owner log-prior.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export MODEL_PATH="${MODEL_PATH:-${PROJECT_ROOT}/checkpoints/pgot_e11_capacity_dit16/checkpoint-10000}"
export OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/checkpoints/pgot_oneshot_memory8_register32_noprior}"
export WANDB_NAME="${WANDB_NAME:-pgot_oneshot_memory8_register32_noprior}"
export MEMORY_KEY_MODE=content
export E11_MEMORIES_PER_OWNER=32
export E11_OBJECT_MEMORIES_PER_OWNER=8
export E11_REGISTER_MEMORIES_PER_OWNER=32
export E8_N_REGISTER=4
export ONE_SHOT_WRITER_SOFTMAX_AXIS=patch
export ONE_SHOT_WRITER_OWNER_PRIOR=False

exec bash "${PROJECT_ROOT}/scripts/train_pgot_oneshot_memory4.sh"
