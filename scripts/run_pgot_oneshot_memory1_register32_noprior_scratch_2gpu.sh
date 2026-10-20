#!/usr/bin/env bash
# Train O1/R32 from vanilla Scale-RAE on two GPUs, then evaluate on one GPU each.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export MAX_STEPS="${MAX_STEPS:-5000}"
export OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/checkpoints/pgot_oneshot_memory1_register32_noprior_scratch}"
export EVAL_ROOT="${EVAL_ROOT:-${PROJECT_ROOT}/outputs/eval_pgot_oneshot_memory1_register32_noprior_scratch}"
BASE_MODEL_PATH="${MODEL_PATH:-/home/jovyan/data/Scale-RAE-Qwen1.5B_DiT2.4B}"

IFS=, read -r -a DEVICES <<< "${CUDA_VISIBLE_DEVICES}"
if (( ${#DEVICES[@]} != 2 )); then
    echo "Set CUDA_VISIBLE_DEVICES to exactly two GPU indices (for example 0,1)." >&2
    exit 2
fi
if [[ -e "${OUTPUT_DIR}/config.json" || \
      -n "$(find "${OUTPUT_DIR}" -maxdepth 1 -name 'checkpoint-*' -print -quit 2>/dev/null || true)" ]]; then
    echo "Training output already contains a model: ${OUTPUT_DIR}" >&2
    echo "Choose a fresh OUTPUT_DIR; this script intentionally does not resume." >&2
    exit 2
fi

mkdir -p "${EVAL_ROOT}/logs"
MODEL_PATH="${BASE_MODEL_PATH}" \
bash "${PROJECT_ROOT}/scripts/train_pgot_oneshot_memory1_register32_noprior_scratch.sh" \
    2>&1 | tee "${EVAL_ROOT}/logs/train.log"

TRAINED_MODEL_PATH="${OUTPUT_DIR}/checkpoint-${MAX_STEPS}"
if [[ ! -f "${TRAINED_MODEL_PATH}/config.json" ]]; then
    TRAINED_MODEL_PATH="${OUTPUT_DIR}"
fi
test -f "${TRAINED_MODEL_PATH}/config.json" || {
    echo "Training finished without a loadable config.json under ${OUTPUT_DIR}" >&2
    exit 1
}

MODEL_PATH="${TRAINED_MODEL_PATH}" \
bash "${PROJECT_ROOT}/scripts/eval_pgot_oneshot_memory1_register32_noprior_scratch_parallel_2gpu.sh"
