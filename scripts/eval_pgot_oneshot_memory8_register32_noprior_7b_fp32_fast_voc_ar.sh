#!/usr/bin/env bash
# VOC validation: image-only AR segmentation for the 7B FP32 fast checkpoint.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export MODEL_PATH="${MODEL_PATH:-${PROJECT_ROOT}/checkpoints/pgot_oneshot_memory8_register32_noprior_7b_fp32_fast/checkpoint-${MAX_STEPS:-5000}}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/outputs/eval_pgot_oneshot_memory8_register32_noprior_7b_fp32_fast}"
export GPU="${GPU:-${CUDA_VISIBLE_DEVICES:-0}}"
export DTYPE=fp32
export BATCH_SIZE="${BATCH_SIZE:-${AR_BATCH_SIZE:-1}}"

exec bash "${PROJECT_ROOT}/scripts/eval_pgot_memory8_full_coco_voc_ar.sh" voc
