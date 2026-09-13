#!/usr/bin/env bash
# Teacher-forced full evaluation for no-prior Memory4 + direct RAE Reader query.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export MODEL_PATH="${MODEL_PATH:-${PROJECT_ROOT}/checkpoints/pgot_oneshot_memory4_noprior_directrae/checkpoint-${MAX_STEPS:-5000}}"
export OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/outputs/eval_pgot_oneshot_memory4_noprior_directrae/tf}"
export COMPUTE_RFID="${COMPUTE_RFID:-True}"
export COMPUTE_KID="${COMPUTE_KID:-True}"

exec bash "${PROJECT_ROOT}/scripts/eval_pgot_oneshot_memory4.sh" \
    --caption_mode teacher_forced "$@"
