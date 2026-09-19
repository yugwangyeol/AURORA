#!/usr/bin/env bash
# Actual PGOT inference only; no external image editor, pixel paste, or GT masks.
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"
export CUDA_VISIBLE_DEVICES="${GPU:?Set GPU to a free device or MIG UUID}"
export PYTHONNOUSERSITE=1
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="/home/jovyan/.conda/envs/scale_rae/lib:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS=3
export MKL_NUM_THREADS=3
PYTHON_BIN="${PYTHON_BIN:-/home/jovyan/.conda/envs/scale_rae/bin/python}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/outputs/figure_model_editing_memory8_register32_noprior}"
mkdir -p "${OUTPUT_ROOT}"
args=(--output_dir "${OUTPUT_ROOT}" --candidates "${CANDIDATES:-16}" --examples "${EXAMPLES:-8}"
      --diffusion_inference_steps "${STEPS:-10}" --seed "${SEED:-1234}")
if [[ -n "${LIMIT:-}" ]]; then args+=(--limit "${LIMIT}"); fi
if [[ ! -f "${OUTPUT_ROOT}/tasks.json" ]]; then
    "${PYTHON_BIN}" -m pgot.eval.figure_model_editing --stage prepare "${args[@]}"
fi
"${PYTHON_BIN}" -u -m pgot.eval.figure_model_editing --stage render "${args[@]}" 2>&1 | tee "${OUTPUT_ROOT}/render.log"
"${PYTHON_BIN}" -m pgot.eval.figure_model_editing --stage select "${args[@]}" \
    --selection_subdir "${SELECTION_SUBDIR:-final}" 2>&1 | tee "${OUTPUT_ROOT}/selection.log"
