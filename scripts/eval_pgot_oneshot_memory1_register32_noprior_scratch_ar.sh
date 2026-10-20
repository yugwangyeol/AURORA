#!/usr/bin/env bash
# Full COCO autoregressive evaluation with reconstruction rFID/KID and count metrics.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export MEMORY_KEY_MODE=content
export MODEL_PATH="${MODEL_PATH:-${PROJECT_ROOT}/checkpoints/pgot_oneshot_memory1_register32_noprior_scratch/checkpoint-${MAX_STEPS:-5000}}"
export OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/outputs/eval_pgot_oneshot_memory1_register32_noprior_scratch/ar}"
export COMPUTE_RFID="${COMPUTE_RFID:-True}"
export COMPUTE_KID="${COMPUTE_KID:-True}"
export COMPUTE_CLASS_METRICS="${COMPUTE_CLASS_METRICS:-True}"
export BATCH_SIZE="${AR_BATCH_SIZE:-4}"

exec bash "${PROJECT_ROOT}/scripts/eval_pgot_oneshot_memory4.sh" \
    --caption_mode autoregressive \
    --ar_max_new_tokens "${AR_MAX_NEW_TOKENS:-512}" "$@"
