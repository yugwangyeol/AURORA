#!/usr/bin/env bash
# Two-GPU training, then single-GPU teacher-forced and autoregressive evaluation.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export MAX_STEPS="${MAX_STEPS:-5000}"
TRAIN_OUTPUT_DIR="${TRAIN_OUTPUT_DIR:-${PROJECT_ROOT}/checkpoints/pgot_oneshot_memory8_register32_noprior_contrastive}"
EVAL_ROOT="${EVAL_ROOT:-${PROJECT_ROOT}/outputs/eval_pgot_oneshot_memory8_register32_noprior_contrastive}"

IFS=, read -r -a DEVICES <<< "${CUDA_VISIBLE_DEVICES}"
if (( ${#DEVICES[@]} != 2 )); then
    echo "Set CUDA_VISIBLE_DEVICES to exactly two GPUs" >&2
    exit 2
fi
if [[ -e "${TRAIN_OUTPUT_DIR}/config.json" || -n "$(find "${TRAIN_OUTPUT_DIR}" -maxdepth 1 -name 'checkpoint-*' -print -quit 2>/dev/null || true)" ]]; then
    echo "Training output already contains a model: ${TRAIN_OUTPUT_DIR}; choose a fresh TRAIN_OUTPUT_DIR" >&2
    exit 2
fi
mkdir -p "${EVAL_ROOT}"

OUTPUT_DIR="${TRAIN_OUTPUT_DIR}" \
    bash "${PROJECT_ROOT}/scripts/train_pgot_oneshot_memory8_register32_noprior_contrastive.sh" \
    2>&1 | tee "${EVAL_ROOT}/train.log"

MODEL_PATH="${TRAIN_OUTPUT_DIR}/checkpoint-${MAX_STEPS}"
if [[ ! -f "${MODEL_PATH}/config.json" ]]; then MODEL_PATH="${TRAIN_OUTPUT_DIR}"; fi

CUDA_VISIBLE_DEVICES="${DEVICES[0]}" MODEL_PATH="${MODEL_PATH}" \
OUTPUT_DIR="${EVAL_ROOT}/tf" \
    bash "${PROJECT_ROOT}/scripts/eval_pgot_oneshot_memory8_register32_noprior_contrastive_tf.sh" \
    2>&1 | tee "${EVAL_ROOT}/tf.log"

CUDA_VISIBLE_DEVICES="${DEVICES[0]}" MODEL_PATH="${MODEL_PATH}" \
OUTPUT_DIR="${EVAL_ROOT}/ar" \
    bash "${PROJECT_ROOT}/scripts/eval_pgot_oneshot_memory8_register32_noprior_contrastive_ar.sh" \
    2>&1 | tee "${EVAL_ROOT}/ar.log"

test -f "${EVAL_ROOT}/tf/summary.json"
test -f "${EVAL_ROOT}/ar/summary.json"
echo "Training -> TF eval -> AR eval complete: ${EVAL_ROOT}"
