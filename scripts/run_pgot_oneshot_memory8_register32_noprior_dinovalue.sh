#!/usr/bin/env bash
# Two-GPU training, then single-GPU teacher-forced and autoregressive evaluation.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export MAX_STEPS="${MAX_STEPS:-5000}"
TRAIN_OUTPUT_DIR="${TRAIN_OUTPUT_DIR:-${PROJECT_ROOT}/checkpoints/pgot_oneshot_memory8_register32_noprior_dinovalue}"
EVAL_ROOT="${EVAL_ROOT:-${PROJECT_ROOT}/outputs/eval_pgot_oneshot_memory8_register32_noprior_dinovalue}"

IFS=, read -r -a DEVICES <<< "${CUDA_VISIBLE_DEVICES}"
if (( ${#DEVICES[@]} != 2 )); then
    echo "Set CUDA_VISIBLE_DEVICES to exactly two GPUs" >&2
    exit 2
fi
mkdir -p "${EVAL_ROOT}"

MODEL_PATH="${TRAIN_OUTPUT_DIR}/checkpoint-${MAX_STEPS}"
if [[ -f "${MODEL_PATH}/config.json" ]]; then
    echo "[PGOT/run] completed checkpoint exists; skipping training: ${MODEL_PATH}"
elif [[ -f "${TRAIN_OUTPUT_DIR}/config.json" ]]; then
    MODEL_PATH="${TRAIN_OUTPUT_DIR}"
    echo "[PGOT/run] completed root model exists; skipping training: ${MODEL_PATH}"
elif [[ -n "$(find "${TRAIN_OUTPUT_DIR}" -maxdepth 1 -type d -name 'checkpoint-*' -print -quit 2>/dev/null || true)" ]]; then
    echo "Incomplete training checkpoints found in ${TRAIN_OUTPUT_DIR}." >&2
    echo "Resume training explicitly or choose a fresh TRAIN_OUTPUT_DIR." >&2
    exit 2
else
    echo "[PGOT/run] stage 1/3: training (${MAX_STEPS} steps)"
    OUTPUT_DIR="${TRAIN_OUTPUT_DIR}" \
        bash "${PROJECT_ROOT}/scripts/train_pgot_oneshot_memory8_register32_noprior_dinovalue.sh" \
        2>&1 | tee "${EVAL_ROOT}/train.log"
    MODEL_PATH="${TRAIN_OUTPUT_DIR}/checkpoint-${MAX_STEPS}"
    if [[ ! -f "${MODEL_PATH}/config.json" ]]; then MODEL_PATH="${TRAIN_OUTPUT_DIR}"; fi
    test -f "${MODEL_PATH}/config.json" || {
        echo "Training ended without a loadable model: ${MODEL_PATH}" >&2
        exit 1
    }
fi

echo "[PGOT/run] stage 2/3: teacher-forced evaluation"
CUDA_VISIBLE_DEVICES="${DEVICES[0]}" MODEL_PATH="${MODEL_PATH}" \
OUTPUT_DIR="${EVAL_ROOT}/tf" \
    bash "${PROJECT_ROOT}/scripts/eval_pgot_oneshot_memory8_register32_noprior_dinovalue_tf.sh" \
    2>&1 | tee "${EVAL_ROOT}/tf.log"
test -s "${EVAL_ROOT}/tf/summary.json" || {
    echo "Teacher-forced evaluation did not produce summary.json" >&2
    exit 1
}

echo "[PGOT/run] stage 3/3: autoregressive evaluation"
CUDA_VISIBLE_DEVICES="${DEVICES[0]}" MODEL_PATH="${MODEL_PATH}" \
OUTPUT_DIR="${EVAL_ROOT}/ar" \
    bash "${PROJECT_ROOT}/scripts/eval_pgot_oneshot_memory8_register32_noprior_dinovalue_ar.sh" \
    2>&1 | tee "${EVAL_ROOT}/ar.log"

test -s "${EVAL_ROOT}/ar/summary.json" || {
    echo "Autoregressive evaluation did not produce summary.json" >&2
    exit 1
}
echo "Training -> TF eval -> AR eval complete: ${EVAL_ROOT}"
