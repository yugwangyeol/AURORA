#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export MODEL_PATH="${MODEL_PATH:-${PROJECT_ROOT}/checkpoints/pgot_oneshot_memory8_register32_noprior_7b_fp32_fast_woreader/checkpoint-${MAX_STEPS:-5000}}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/outputs/eval_pgot_oneshot_memory8_register32_noprior_7b_fp32_fast_woreader}"
AUDIT_PYTHON="${AUDIT_PYTHON:-/home/jovyan/.conda/envs/scale_rae/bin/python}"
PYTHONNOUSERSITE=1 PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}" \
LD_LIBRARY_PATH="$(dirname "${AUDIT_PYTHON}")/../lib:${LD_LIBRARY_PATH:-}" \
"${AUDIT_PYTHON}" -m pgot.eval.verify_woreader_config \
    --model_path "${MODEL_PATH}" --model_size 7b
export GPU="${GPU:-${CUDA_VISIBLE_DEVICES:-0}}" DTYPE=fp32 BATCH_SIZE="${BATCH_SIZE:-1}"
exec bash "${PROJECT_ROOT}/scripts/eval_pgot_memory8_full_coco_voc_ar.sh" voc
