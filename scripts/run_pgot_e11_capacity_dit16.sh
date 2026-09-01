#!/usr/bin/env bash
# Train E11 Capacity + DiT-16, then evaluate the final checkpoint.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MAX_STEPS="${MAX_STEPS:-7000}"
TRAIN_OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/checkpoints/pgot_e11_capacity_dit16}"
EVAL_OUTPUT_DIR="${EVAL_OUTPUT_DIR:-${PROJECT_ROOT}/outputs/eval_pgot_e11_capacity_dit16}"

OUTPUT_DIR="${TRAIN_OUTPUT_DIR}" MAX_STEPS="${MAX_STEPS}" \
    bash "${PROJECT_ROOT}/scripts/train_pgot_e11_capacity_dit16.sh" \
    2>&1 | tee "${PROJECT_ROOT}/outputs/train_pgot_e11_capacity_dit16.log"

MODEL_PATH="${TRAIN_OUTPUT_DIR}/checkpoint-${MAX_STEPS}" \
OUTPUT_DIR="${EVAL_OUTPUT_DIR}" \
    bash "${PROJECT_ROOT}/scripts/eval_pgot_e11_capacity_dit16.sh" \
    2>&1 | tee "${PROJECT_ROOT}/outputs/eval_pgot_e11_capacity_dit16.log"
