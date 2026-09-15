#!/usr/bin/env bash
# Autoregressive evaluation for the direct-RAE adapter + CODA objective.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export MODEL_PATH="${MODEL_PATH:-${PROJECT_ROOT}/checkpoints/pgot_oneshot_memory8_register32_noprior_directrae_adapter_contrastive/checkpoint-${MAX_STEPS:-5000}}"
export OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/outputs/eval_pgot_oneshot_memory8_register32_noprior_directrae_adapter_contrastive/ar}"

exec bash "${PROJECT_ROOT}/scripts/eval_pgot_oneshot_memory8_register32_noprior_directrae_adapter_ar.sh" "$@"
