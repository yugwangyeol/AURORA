#!/usr/bin/env bash
# Two-GPU training followed by simultaneous full TF/AR evaluation.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export MAX_STEPS="${MAX_STEPS:-5000}"
export OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/checkpoints/pgot_oneshot_ownergrad_feature}"
export EVAL_ROOT="${EVAL_ROOT:-${PROJECT_ROOT}/outputs/eval_pgot_oneshot_ownergrad_feature}"
mkdir -p "${EVAL_ROOT}"
if [[ -e "${OUTPUT_DIR}/config.json" || -n "$(find "${OUTPUT_DIR}" -maxdepth 1 -name 'checkpoint-*' -print -quit 2>/dev/null || true)" ]]; then
    echo "Training output already contains a model: ${OUTPUT_DIR}; choose a fresh OUTPUT_DIR" >&2
    exit 2
fi
bash "${PROJECT_ROOT}/scripts/train_pgot_oneshot_ownergrad_feature.sh" 2>&1 | tee "${EVAL_ROOT}/train.log"
export MODEL_PATH="${OUTPUT_DIR}/checkpoint-${MAX_STEPS}"
if [[ ! -f "${MODEL_PATH}/config.json" ]]; then
    export MODEL_PATH="${OUTPUT_DIR}"
fi
exec bash "${PROJECT_ROOT}/scripts/eval_pgot_oneshot_ownergrad_feature_parallel.sh"
