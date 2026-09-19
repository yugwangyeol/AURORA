#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"
export PYTHONNOUSERSITE=1
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="/home/jovyan/.conda/envs/scale_rae/lib:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS=3
export CUDA_VISIBLE_DEVICES="${GPU:?Set GPU to a free MIG UUID or CUDA device}"
PYTHON_BIN="/home/jovyan/.venvs/ctrlo_eval/bin/python"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/outputs/figure_ours_ctrlo_category}"
mkdir -p "${OUTPUT_ROOT}"
args=(--output_dir "${OUTPUT_ROOT}" --examples "${EXAMPLES:-10}")
if [[ -n "${LIMIT:-}" ]]; then args+=(--limit "${LIMIT}"); fi
"${PYTHON_BIN}" -m pgot.eval.figure_ctrlo_examples --stage infer "${args[@]}" 2>&1 | tee "${OUTPUT_ROOT}/inference.log"
select_args=()
if [[ -n "${SELECTION_IDS:-}" ]]; then
    select_args+=(--selection_ids "${SELECTION_IDS}" --selection_subdir curated)
fi
"${PYTHON_BIN}" -m pgot.eval.figure_ctrlo_examples --stage select "${args[@]}" "${select_args[@]}" 2>&1 | tee "${OUTPUT_ROOT}/selection.log"
