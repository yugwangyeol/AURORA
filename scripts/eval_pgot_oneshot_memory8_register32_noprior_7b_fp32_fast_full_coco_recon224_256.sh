#!/usr/bin/env bash
# One 7B COCO AR pass: full 5000 segmentation + native-224 and upscaled-256 reconstruction metrics.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GPU="${GPU:?Set GPU to one physical CUDA index}"
[[ "${GPU}" != *,* ]] || { echo "GPU must name exactly one device" >&2; exit 2; }
export CUDA_VISIBLE_DEVICES="${GPU}"
export MODEL_PATH="${MODEL_PATH:-${PROJECT_ROOT}/checkpoints/pgot_oneshot_memory8_register32_noprior_7b_fp32_fast/checkpoint-5000}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/outputs/eval_pgot_oneshot_memory8_register32_noprior_7b_fp32_fast/coco_full_ar_recon224_256}"
AUDIT_PYTHON="${AUDIT_PYTHON:-/home/jovyan/.conda/envs/scale_rae/bin/python}"

PYTHONNOUSERSITE=1 PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}" \
LD_LIBRARY_PATH="$(dirname "${AUDIT_PYTHON}")/../lib:${LD_LIBRARY_PATH:-}" \
"${AUDIT_PYTHON}" -m pgot.eval.verify_oneshot_o8r32_config \
    --model_path "${MODEL_PATH}" --model_size 7b --require_scratch_init

export DTYPE=fp32
export BATCH_SIZE="${BATCH_SIZE:-1}"
export COMPUTE_RFID=True
export COMPUTE_KID=True
export RECON_METRIC_RESOLUTION=0
export ALSO_RECON_METRIC_RESOLUTION=256
export DIFFUSION_INFERENCE_STEPS="${DIFFUSION_INFERENCE_STEPS:-10}"

exec bash "${PROJECT_ROOT}/scripts/eval_pgot_memory8_full_coco_voc_ar.sh" coco
