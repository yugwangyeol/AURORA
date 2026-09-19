#!/usr/bin/env bash
# Original initialization -> w/o Reader-loss training -> AR (no TF).
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2,3}"
IFS=, read -r -a DEVICES <<< "${CUDA_VISIBLE_DEVICES}"
if (( ${#DEVICES[@]} != 2 )) || [[ "${DEVICES[0]}" == "${DEVICES[1]}" ]]; then
    echo 'Set exactly two distinct GPUs, e.g. CUDA_VISIBLE_DEVICES=2,3.' >&2; exit 2
fi
export MAX_STEPS="${MAX_STEPS:-5000}"
export OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/checkpoints/pgot_oneshot_memory8_register32_noprior_woreader}"
export EVAL_ROOT="${EVAL_ROOT:-${PROJECT_ROOT}/outputs/eval_pgot_oneshot_memory8_register32_noprior_woreader}"
if [[ -e "${OUTPUT_DIR}/config.json" || -n "$(find "${OUTPUT_DIR}" -maxdepth 1 -name 'checkpoint-*' -print -quit 2>/dev/null || true)" || -s "${EVAL_ROOT}/train.log" ]]; then
    echo 'Existing training output; use a fresh OUTPUT_DIR/EVAL_ROOT. For completed training use the eval-only script.' >&2; exit 2
fi
mkdir -p "${EVAL_ROOT}"
bash "${PROJECT_ROOT}/scripts/train_pgot_oneshot_memory8_register32_noprior_woreader.sh" \
    2>&1 | tee "${EVAL_ROOT}/train.log"
export MODEL_PATH="${OUTPUT_DIR}/checkpoint-${MAX_STEPS}"
if [[ ! -f "${MODEL_PATH}/config.json" ]]; then export MODEL_PATH="${OUTPUT_DIR}"; fi
exec bash "${PROJECT_ROOT}/scripts/eval_pgot_oneshot_memory8_register32_noprior_woreader.sh"
