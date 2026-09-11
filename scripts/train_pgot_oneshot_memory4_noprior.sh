#!/usr/bin/env bash
# Memory4-content ablation: semantic+ID patch attention without log ownership prior.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export MODEL_PATH="${MODEL_PATH:-${PROJECT_ROOT}/checkpoints/pgot_e11_capacity_dit16/checkpoint-10000}"
export OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/checkpoints/pgot_oneshot_memory4_noprior}"
export WANDB_NAME="${WANDB_NAME:-pgot_oneshot_memory4_noprior}"
export MEMORY_KEY_MODE=content
export ONE_SHOT_WRITER_SOFTMAX_AXIS=patch
export ONE_SHOT_WRITER_OWNER_PRIOR=False

exec bash "${PROJECT_ROOT}/scripts/train_pgot_oneshot_memory4.sh"
