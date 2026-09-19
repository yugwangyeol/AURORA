#!/usr/bin/env bash
# Evaluate the trained w/o LM-loss checkpoint: AR only, on GPU1 by default.
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
IFS=, read -r -a DEVICES <<< "${CUDA_VISIBLE_DEVICES}"
if (( ${#DEVICES[@]} != 2 )) || [[ "${DEVICES[0]}" == "${DEVICES[1]}" ]]; then
    echo 'Set exactly two distinct GPUs, e.g. CUDA_VISIBLE_DEVICES=0,1.' >&2; exit 2
fi
export MODEL_PATH="${MODEL_PATH:-${PROJECT_ROOT}/checkpoints/pgot_oneshot_memory8_register32_noprior_wolm/checkpoint-${MAX_STEPS:-5000}}"
EVAL_ROOT="${EVAL_ROOT:-${PROJECT_ROOT}/outputs/eval_pgot_oneshot_memory8_register32_noprior_wolm}"
AUDIT_PYTHON="${AUDIT_PYTHON:-/home/jovyan/.conda/envs/scale_rae/bin/python}"
PYTHONNOUSERSITE=1 PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}" \
LD_LIBRARY_PATH="$(dirname "${AUDIT_PYTHON}")/../lib:${LD_LIBRARY_PATH:-}" \
"${AUDIT_PYTHON}" -m pgot.eval.verify_woreader_config --ablation wolm --model_path "${MODEL_PATH}"
mkdir -p "${EVAL_ROOT}"
export DTYPE=fp32
CUDA_VISIBLE_DEVICES="${DEVICES[1]}" OUTPUT_DIR="${EVAL_ROOT}/ar" \
    bash "${PROJECT_ROOT}/scripts/eval_pgot_oneshot_memory8_register32_noprior_ar.sh" \
    2>&1 | tee -a "${EVAL_ROOT}/ar.log"
echo "Completed AR: ${EVAL_ROOT}/ar"
