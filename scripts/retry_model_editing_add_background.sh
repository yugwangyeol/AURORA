#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"
export CUDA_VISIBLE_DEVICES="${GPU:?Choose a free device or MIG UUID}"
export PYTHONNOUSERSITE=1
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="/home/jovyan/.conda/envs/scale_rae/lib:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS=3
export MKL_NUM_THREADS=3
PYTHON_BIN="${PYTHON_BIN:-/home/jovyan/.conda/envs/scale_rae/bin/python}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/outputs/figure_model_editing_retry_add_background}"
args=(--output_dir "${OUTPUT_ROOT}" --diffusion_inference_steps "${STEPS:-20}" --seed "${SEED:-1234}")
if [[ ! -f "${OUTPUT_ROOT}/tasks.json" ]]; then
    "${PYTHON_BIN}" -m pgot.eval.retry_model_editing --stage prepare "${args[@]}"
fi
"${PYTHON_BIN}" -u -m pgot.eval.retry_model_editing --stage render "${args[@]}" 2>&1 | tee "${OUTPUT_ROOT}/render.log"
for variant in native predicted_guided; do
    "${PYTHON_BIN}" -m pgot.eval.retry_model_editing --stage select "${args[@]}" \
        --task_filter "_${variant}_" --selection_subdir "${variant}" --examples 8
done
