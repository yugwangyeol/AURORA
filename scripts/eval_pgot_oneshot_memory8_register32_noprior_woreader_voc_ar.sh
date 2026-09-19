#!/usr/bin/env bash
# VOC zero-shot AR evaluation for the original 1.5B O8/R32 checkpoint trained
# without Reader supervision losses. The Reader module itself remains enabled.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-/home/jovyan/.conda/envs/scale_rae/bin/python}"
MODEL_PATH="${MODEL_PATH:-${PROJECT_ROOT}/checkpoints/pgot_oneshot_memory8_register32_noprior_woreader/checkpoint-5000}"

if [[ "${SMOKE:-0}" == 1 ]]; then
    default_output="${PROJECT_ROOT}/outputs/_smoke_pgot_woreader_voc_ar"
else
    default_output="${PROJECT_ROOT}/outputs/segeval_pgot_oneshot_memory8_register32_noprior_woreader"
fi

export PYTHONNOUSERSITE=1
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="$(dirname "${PYTHON}")/../lib:${LD_LIBRARY_PATH:-}"
"${PYTHON}" -m pgot.eval.verify_woreader_config --model_path "${MODEL_PATH}"

export MODEL_PATH
export OUTPUT_ROOT="${OUTPUT_ROOT:-${default_output}}"
exec bash "${PROJECT_ROOT}/scripts/eval_pgot_memory8_full_coco_voc_ar.sh" voc
