#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export EVAL_MODE="${EVAL_MODE:-full}"
export RUN_DIR="${RUN_DIR:-/home/jovyan/AURORA/checkpoints/aurora_refcoco_multislot24_pair2_reg64_xattn_zeroinit_stage1_fp32}"
export CAPTION_OUTPUT_DIR="${CAPTION_OUTPUT_DIR:-/home/jovyan/AURORA/outputs/stagea_object_captions_val2017_refexp}"
export IMAGE_DIR="${IMAGE_DIR:-/home/jovyan/data/coco/val2017}"

export GPU0="${GPU0:-0}"
export GPU1="${GPU1:-1}"
export DTYPE="${DTYPE:-fp32}"
export PER_DEVICE_EVAL_BATCH_SIZE="${PER_DEVICE_EVAL_BATCH_SIZE:-16}"
export DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-4}"
export MAX_CAPTION_TOKENS="${MAX_CAPTION_TOKENS:-192}"
export GUIDANCE_LEVEL="${GUIDANCE_LEVEL:-1.0}"
export DIFFUSION_STEPS="${DIFFUSION_STEPS:-20}"
export TORCH_COMPILE="${TORCH_COMPILE:-1}"
export SEGMENTATION_BG_THRESHOLD="${SEGMENTATION_BG_THRESHOLD:-0.5}"
export SEGMENTATION_BG_THRESHOLDS="${SEGMENTATION_BG_THRESHOLDS:-0.2,0.3}"
export ATTN_THRESHOLD="${ATTN_THRESHOLD:-0.5}"
export ATTN_THRESHOLDS="${ATTN_THRESHOLDS:-0.2,0.3}"
export MAX_SAMPLES="${MAX_SAMPLES:-}"

# Fast metric-focused defaults: no image dumps unless explicitly enabled.
export SAVE_IMAGES="${SAVE_IMAGES:-0}"
export SAVE_LIMIT="${SAVE_LIMIT:-0}"
export SAVE_FIXED_FIRST_N="${SAVE_FIXED_FIRST_N:-0}"
export REPORT_LOSSES="${REPORT_LOSSES:-0}"
export EVAL_SLOT_ATTENTION="${EVAL_SLOT_ATTENTION:-1}"
export SAVE_ATTN_MAPS="${SAVE_ATTN_MAPS:-0}"
export SAVE_ATTN_LIMIT="${SAVE_ATTN_LIMIT:-0}"

exec env \
  SEGMENTATION_BG_THRESHOLD="${SEGMENTATION_BG_THRESHOLD}" \
  SEGMENTATION_BG_THRESHOLDS="${SEGMENTATION_BG_THRESHOLDS}" \
  ATTN_THRESHOLD="${ATTN_THRESHOLD}" \
  ATTN_THRESHOLDS="${ATTN_THRESHOLDS}" \
  MAX_SAMPLES="${MAX_SAMPLES}" \
  bash "${SCRIPT_DIR}/run_eval_captionslot_generated_object_captions.sh"
