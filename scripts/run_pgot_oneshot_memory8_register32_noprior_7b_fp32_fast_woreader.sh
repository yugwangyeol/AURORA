#!/usr/bin/env bash
# Four-GPU training, then parallel COCO AR / VOC AR / compositional evaluation.
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export MAX_STEPS="${MAX_STEPS:-5000}"
export OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/checkpoints/pgot_oneshot_memory8_register32_noprior_7b_fp32_fast_woreader}"
export EVAL_ROOT="${EVAL_ROOT:-${PROJECT_ROOT}/outputs/eval_pgot_oneshot_memory8_register32_noprior_7b_fp32_fast_woreader}"
if [[ -e "${OUTPUT_DIR}/config.json" || \
      -n "$(find "${OUTPUT_DIR}" -maxdepth 1 -name 'checkpoint-*' -print -quit 2>/dev/null || true)" || \
      -s "${EVAL_ROOT}/train.log" ]]; then
    echo 'Existing output detected; use fresh OUTPUT_DIR and EVAL_ROOT.' >&2; exit 2
fi
mkdir -p "${EVAL_ROOT}"
bash "${PROJECT_ROOT}/scripts/train_pgot_oneshot_memory8_register32_noprior_7b_fp32_fast_woreader.sh" \
    2>&1 | tee "${EVAL_ROOT}/train.log"
export MODEL_PATH="${OUTPUT_DIR}/checkpoint-${MAX_STEPS}"
if [[ ! -f "${MODEL_PATH}/config.json" ]]; then
    echo "Training finished without expected checkpoint: ${MODEL_PATH}" >&2; exit 1
fi
AUDIT_PYTHON="${AUDIT_PYTHON:-/home/jovyan/.conda/envs/scale_rae/bin/python}"
PYTHONNOUSERSITE=1 PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}" \
LD_LIBRARY_PATH="$(dirname "${AUDIT_PYTHON}")/../lib:${LD_LIBRARY_PATH:-}" \
"${AUDIT_PYTHON}" -m pgot.eval.verify_woreader_config \
    --model_path "${MODEL_PATH}" --model_size 7b
exec bash "${PROJECT_ROOT}/scripts/eval_pgot_oneshot_memory8_register32_noprior_7b_fp32_fast_woreader_parallel.sh"
