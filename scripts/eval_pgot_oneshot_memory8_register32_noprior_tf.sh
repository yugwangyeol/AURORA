#!/usr/bin/env bash
# Full teacher-forced evaluation with rFID and KID.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export MEMORY_KEY_MODE=content
export MODEL_PATH="${MODEL_PATH:-${PROJECT_ROOT}/checkpoints/pgot_oneshot_memory8_register32_noprior/checkpoint-${MAX_STEPS:-5000}}"
export OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/outputs/eval_pgot_oneshot_memory8_register32_noprior/tf}"
export COMPUTE_RFID="${COMPUTE_RFID:-True}"
export COMPUTE_KID="${COMPUTE_KID:-True}"

exec bash "${PROJECT_ROOT}/scripts/eval_pgot_oneshot_memory4.sh" --caption_mode teacher_forced "$@"
