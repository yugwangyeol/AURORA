#!/usr/bin/env bash
# No-prior Memory4 ablation with raw Scale-RAE queries sent directly to Reader.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export MODEL_PATH="${MODEL_PATH:-${PROJECT_ROOT}/checkpoints/pgot_e11_capacity_dit16/checkpoint-10000}"
export OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/checkpoints/pgot_oneshot_memory4_noprior_directrae}"
export WANDB_NAME="${WANDB_NAME:-pgot_oneshot_memory4_noprior_directrae}"
export ONE_SHOT_DIRECT_RAE_QUERY=True

exec bash "${PROJECT_ROOT}/scripts/train_pgot_oneshot_memory4_noprior.sh"
