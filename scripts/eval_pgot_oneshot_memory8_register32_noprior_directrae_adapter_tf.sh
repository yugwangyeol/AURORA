#!/usr/bin/env bash
# Teacher-forced full evaluation for the Object-8/Register-32 direct-query adapter.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export MODEL_PATH="${MODEL_PATH:-${PROJECT_ROOT}/checkpoints/pgot_oneshot_memory8_register32_noprior_directrae_adapter/checkpoint-${MAX_STEPS:-5000}}"
export OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/outputs/eval_pgot_oneshot_memory8_register32_noprior_directrae_adapter/tf}"
export COMPUTE_RFID="${COMPUTE_RFID:-True}"
export COMPUTE_KID="${COMPUTE_KID:-True}"

exec bash "${PROJECT_ROOT}/scripts/eval_pgot_oneshot_memory4.sh" \
    --caption_mode teacher_forced "$@"
