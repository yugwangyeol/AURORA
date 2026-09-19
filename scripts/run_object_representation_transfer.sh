#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export CUDA_VISIBLE_DEVICES="${GPU:-0}"
export PYTHONNOUSERSITE=1
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="/home/jovyan/.conda/envs/scale_rae/lib:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS=3 MKL_NUM_THREADS=3
PYTHON_BIN="${PYTHON_BIN:-/home/jovyan/.conda/envs/scale_rae/bin/python}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/outputs/object_representation_transfer_memory8_register32}"
mkdir -p "${OUTPUT_DIR}/logs"
args=(--output_dir "${OUTPUT_DIR}" --pairs "${PAIRS:-${PROJECT_ROOT}/configs/object_representation_transfer_pairs.jsonl}"
      --model_path "${MODEL_PATH:-${PROJECT_ROOT}/checkpoints/pgot_oneshot_memory8_register32_noprior/checkpoint-5000}"
      --seed "${SEED:-1234}" --diffusion_inference_steps "${STEPS:-10}"
      --guidance_scale "${GUIDANCE_SCALE:-1}" --limit "${LIMIT:-0}")
"${PYTHON_BIN}" -m pgot.eval.run_object_representation_transfer --stage prepare "${args[@]}" 2>&1 | tee -a "${OUTPUT_DIR}/logs/prepare.log"
"${PYTHON_BIN}" -u -m pgot.eval.run_object_representation_transfer --stage render "${args[@]}" 2>&1 | tee -a "${OUTPUT_DIR}/logs/render.log"
"${PYTHON_BIN}" -m pgot.eval.run_object_representation_transfer --stage audit "${args[@]}" 2>&1 | tee -a "${OUTPUT_DIR}/logs/audit.log"
