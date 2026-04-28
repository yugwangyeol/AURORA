#!/bin/bash
# =============================================================================
# 2-GPU eval wrapper for captionslot_firstslot_headprior stage1 checkpoint.
# Computes rFID, SSIM, PSNR, MSE (slot-attention eval disabled for speed).
#
# Usage:
#   CHECKPOINT=checkpoint-40000 bash scripts/run_eval_headprior_stage1.sh
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-/home/jovyan/AURORA/checkpoints/captionslot_firstslot_headprior_s1.0_stage1}"
export CHECKPOINT="${CHECKPOINT:-checkpoint-40000}"
export MODEL_PATH="${MODEL_PATH:-${CHECKPOINT_ROOT}/${CHECKPOINT}}"

export IMAGE_DIR="${IMAGE_DIR:-/home/jovyan/data/coco/val2017}"
export CAPTIONS_JSONL="${CAPTIONS_JSONL:-/home/jovyan/Scale-RAE/outputs/caption_to_image_recon_concise/captions.jsonl}"
export OUTPUT_DIR="${OUTPUT_DIR:-/home/jovyan/AURORA/outputs/eval_headprior_stage1_${CHECKPOINT}}"

export GPU0="${GPU0:-0}"
export GPU1="${GPU1:-1}"

export DTYPE="${DTYPE:-fp32}"
export PER_DEVICE_EVAL_BATCH_SIZE="${PER_DEVICE_EVAL_BATCH_SIZE:-8}"
export DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-8}"
export MAX_CAPTION_TOKENS="${MAX_CAPTION_TOKENS:-64}"
export MAX_SAMPLES="${MAX_SAMPLES:-}"
export GUIDANCE_LEVEL="${GUIDANCE_LEVEL:-1.0}"
export DIFFUSION_STEPS="${DIFFUSION_STEPS:-20}"
export TORCH_COMPILE="${TORCH_COMPILE:-0}"

# Only the 4 recon metrics are requested → disable slot-attention eval & attn overlays.
export EVAL_SLOT_ATTENTION="${EVAL_SLOT_ATTENTION:-0}"
export SAVE_ATTN_MAPS="${SAVE_ATTN_MAPS:-0}"
export REPORT_LOSSES="${REPORT_LOSSES:-0}"
export SAVE_IMAGES="${SAVE_IMAGES:-1}"
export SAVE_LIMIT="${SAVE_LIMIT:-32}"
export SAVE_FIXED_FIRST_N="${SAVE_FIXED_FIRST_N:-32}"

if [ ! -d "${MODEL_PATH}" ]; then
    echo "[ERROR] Checkpoint directory not found: ${MODEL_PATH}" >&2
    exit 1
fi

mkdir -p "${OUTPUT_DIR}"

echo "===== Headprior Stage1 Eval (2-GPU) ====="
echo "Model:        ${MODEL_PATH}"
echo "Image dir:    ${IMAGE_DIR}"
echo "Captions:     ${CAPTIONS_JSONL}"
echo "Output:       ${OUTPUT_DIR}"
echo "GPUs:         ${GPU0}, ${GPU1}"
echo "Per-GPU BS:   ${PER_DEVICE_EVAL_BATCH_SIZE}"
echo "Metrics:      rFID, SSIM, PSNR, MSE"
echo "=========================================="

exec bash "${SCRIPT_DIR}/run_eval_captionslot_checkpoint_parallel.sh"
