#!/usr/bin/env bash
# Autoregressive evaluation for Object-8/Register-32 DINO memory values.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export MODEL_PATH="${MODEL_PATH:-${PROJECT_ROOT}/checkpoints/pgot_oneshot_memory8_register32_noprior_dinovalue/checkpoint-${MAX_STEPS:-5000}}"
export OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/outputs/eval_pgot_oneshot_memory8_register32_noprior_dinovalue/ar}"

exec bash "${PROJECT_ROOT}/scripts/eval_pgot_oneshot_memory8_register32_noprior_ar.sh" "$@"
