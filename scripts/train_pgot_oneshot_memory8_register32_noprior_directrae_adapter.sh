#!/usr/bin/env bash
# Object-8/Register-32 no-prior memory with direct RAE queries and a small
# identity-initialized RMSNorm residual MLP before the Reader.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export MODEL_PATH="${MODEL_PATH:-${PROJECT_ROOT}/checkpoints/pgot_e11_capacity_dit16/checkpoint-10000}"
export OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/checkpoints/pgot_oneshot_memory8_register32_noprior_directrae_adapter}"
export WANDB_NAME="${WANDB_NAME:-pgot_oneshot_memory8_register32_noprior_directrae_adapter}"
export ONE_SHOT_DIRECT_RAE_QUERY=True
export ONE_SHOT_RAE_QUERY_ADAPTER_ENABLE=True
export ONE_SHOT_RAE_QUERY_ADAPTER_BOTTLENECK="${ONE_SHOT_RAE_QUERY_ADAPTER_BOTTLENECK:-384}"

exec bash "${PROJECT_ROOT}/scripts/train_pgot_oneshot_memory8_register32_noprior.sh"
