#!/usr/bin/env bash
# E11 Capacity + Query Separation: semantic-only owner, memory+ID inner route.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export MODEL_PATH="${MODEL_PATH:-${PROJECT_ROOT}/checkpoints/pgot_e11_dual_m4/checkpoint-10000}"
export OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/checkpoints/pgot_e11_capacity_query_separation}"
export WANDB_NAME="${WANDB_NAME:-pgot_e11_capacity_query_separation}"
export E11_QUERY_SEPARATION_ENABLE=True

exec bash "${PROJECT_ROOT}/scripts/train_pgot_e11_capacity.sh"
