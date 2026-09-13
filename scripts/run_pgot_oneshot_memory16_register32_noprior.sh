#!/usr/bin/env bash
# Two-GPU training, then TF evaluation, then AR evaluation.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export MAX_STEPS="${MAX_STEPS:-5000}"
export OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/checkpoints/pgot_oneshot_memory16_register32_noprior}"
EVAL_ROOT="${EVAL_ROOT:-${PROJECT_ROOT}/outputs/eval_pgot_oneshot_memory16_register32_noprior}"
IFS=, read -r -a EVAL_DEVICES <<< "${CUDA_VISIBLE_DEVICES}"
if (( ${#EVAL_DEVICES[@]} != 2 )); then
    echo "Set CUDA_VISIBLE_DEVICES to exactly two GPUs" >&2
    exit 2
fi

if [[ -e "${OUTPUT_DIR}/config.json" || -n "$(find "${OUTPUT_DIR}" -maxdepth 1 -name 'checkpoint-*' -print -quit 2>/dev/null || true)" ]]; then
    echo "Training output already contains a model: ${OUTPUT_DIR}; choose a fresh OUTPUT_DIR" >&2
    exit 2
fi

mkdir -p "${EVAL_ROOT}"
bash "${PROJECT_ROOT}/scripts/train_pgot_oneshot_memory16_register32_noprior.sh" \
    2>&1 | tee "${EVAL_ROOT}/train.log"

export MODEL_PATH="${OUTPUT_DIR}/checkpoint-${MAX_STEPS}"
if [[ ! -f "${MODEL_PATH}/config.json" ]]; then export MODEL_PATH="${OUTPUT_DIR}"; fi

CUDA_VISIBLE_DEVICES="${EVAL_DEVICES[0]}" OUTPUT_DIR="${EVAL_ROOT}/tf" \
    bash "${PROJECT_ROOT}/scripts/eval_pgot_oneshot_memory16_register32_noprior_tf.sh" \
    2>&1 | tee "${EVAL_ROOT}/tf.log"
test -f "${EVAL_ROOT}/tf/summary.json"

CUDA_VISIBLE_DEVICES="${EVAL_DEVICES[1]}" OUTPUT_DIR="${EVAL_ROOT}/ar" \
    bash "${PROJECT_ROOT}/scripts/eval_pgot_oneshot_memory16_register32_noprior_ar.sh" \
    2>&1 | tee "${EVAL_ROOT}/ar.log"
test -f "${EVAL_ROOT}/ar/summary.json"

echo "Training -> TF eval -> AR eval complete: ${EVAL_ROOT}"
