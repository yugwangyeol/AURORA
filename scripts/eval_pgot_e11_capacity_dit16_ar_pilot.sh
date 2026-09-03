#!/usr/bin/env bash
# 100-image autoregressive-caption pilot for E11 Capacity + DiT-16.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export MODEL_PATH="${MODEL_PATH:-${PROJECT_ROOT}/checkpoints/pgot_e11_capacity_dit16/checkpoint-10000}"
export OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/outputs/eval_pgot_e11_capacity_dit16_ar_pilot100}"
export MAX_SAMPLES="${MAX_SAMPLES:-100}"
export BATCH_SIZE="${BATCH_SIZE:-1}"
export NUM_WORKERS="${NUM_WORKERS:-2}"
export DTYPE=fp32

PROJECT_EVAL_ARGS=(
    --caption_mode autoregressive
    --ar_max_new_tokens "${AR_MAX_NEW_TOKENS:-512}"
)

# Keep every other setting identical to the existing E11 Capacity + DiT-16
# CODA-512 evaluation.  The only intentional change is generated captions.
exec bash "${PROJECT_ROOT}/scripts/eval_pgot_e8_visual_memory.sh" "${PROJECT_EVAL_ARGS[@]}"
