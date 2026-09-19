#!/usr/bin/env bash
# RefCOCO language-region alignment for the original 1.5B O8/R32 checkpoint
# trained without Reader supervision losses. Runs the same fixed protocol and
# CTRL-O comparison as eval_deor_ctrlo_refcoco.sh in a separate output root.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PGOT_PYTHON:-/home/jovyan/.conda/envs/scale_rae/bin/python}"
MODEL_PATH="${MODEL_PATH:-${PROJECT_ROOT}/checkpoints/pgot_oneshot_memory8_register32_noprior_woreader/checkpoint-5000}"

if [[ "${SMOKE:-0}" == 1 ]]; then
    default_output="${PROJECT_ROOT}/outputs/smoke_deor_woreader_ctrlo_refcoco_direct"
else
    default_output="${PROJECT_ROOT}/outputs/eval_deor_woreader_ctrlo_refcoco_direct"
fi

export PYTHONNOUSERSITE=1
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="$(dirname "${PYTHON}")/../lib:${LD_LIBRARY_PATH:-}"
"${PYTHON}" -m pgot.eval.verify_woreader_config --model_path "${MODEL_PATH}"

export MODEL_PATH
export OUTPUT_ROOT="${OUTPUT_ROOT:-${default_output}}"
exec bash "${PROJECT_ROOT}/scripts/eval_deor_ctrlo_refcoco.sh"
