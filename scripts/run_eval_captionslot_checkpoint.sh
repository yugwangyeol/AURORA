#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export MODEL_PATH="${MODEL_PATH:-/home/jovyan/AURORA/checkpoints/captionslot_firstslot_noprior_recon_stage1_fp32/checkpoint-18000}"
export IMAGE_DIR="${IMAGE_DIR:-/home/jovyan/data/coco/val2017}"
export CAPTIONS_JSONL="${CAPTIONS_JSONL:-/home/jovyan/Scale-RAE/outputs/caption_to_image_recon_concise/captions.jsonl}"
export OUTPUT_DIR="${OUTPUT_DIR:-/home/jovyan/AURORA/outputs/captionslot_eval_checkpoint}"
export DTYPE="${DTYPE:-fp32}"
export DEVICE="${DEVICE:-cuda}"
export PER_DEVICE_EVAL_BATCH_SIZE="${PER_DEVICE_EVAL_BATCH_SIZE:-4}"
export DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-4}"
export MAX_CAPTION_TOKENS="${MAX_CAPTION_TOKENS:-64}"
export MAX_SAMPLES="${MAX_SAMPLES:-}"
export NUM_SHARDS="${NUM_SHARDS:-1}"
export SHARD_INDEX="${SHARD_INDEX:-0}"
export GUIDANCE_LEVEL="${GUIDANCE_LEVEL:-1.0}"
export SAVE_IMAGES="${SAVE_IMAGES:-1}"
export SAVE_LIMIT="${SAVE_LIMIT:-50}"
export SAVE_FIXED_FIRST_N="${SAVE_FIXED_FIRST_N:-50}"
export REPORT_LOSSES="${REPORT_LOSSES:-1}"
export COCO_INSTANCES="${COCO_INSTANCES:-/home/jovyan/data/coco/annotations/instances_val2017.json}"
export EVAL_SLOT_ATTENTION="${EVAL_SLOT_ATTENTION:-1}"
export ATTN_THRESHOLD="${ATTN_THRESHOLD:-0.5}"
export ATTN_THRESHOLDS="${ATTN_THRESHOLDS:-}"
export SEGMENTATION_BG_THRESHOLD="${SEGMENTATION_BG_THRESHOLD:-0.5}"
export SEGMENTATION_BG_THRESHOLDS="${SEGMENTATION_BG_THRESHOLDS:-}"
export SAVE_ATTN_MAPS="${SAVE_ATTN_MAPS:-1}"
export SAVE_ATTN_LIMIT="${SAVE_ATTN_LIMIT:-100}"
export DIFFUSION_STEPS="${DIFFUSION_STEPS:-20}"
export TORCH_COMPILE="${TORCH_COMPILE:-1}"

PYTHONNOUSERSITE="${PYTHONNOUSERSITE:-1}" \
/home/jovyan/.conda/envs/scale_rae/bin/python \
  "${SCRIPT_DIR}/eval_captionslot_checkpoint.py" \
  --model-path "${MODEL_PATH}" \
  --image-dir "${IMAGE_DIR}" \
  --captions-jsonl "${CAPTIONS_JSONL}" \
  --output-dir "${OUTPUT_DIR}" \
  --device "${DEVICE}" \
  --dtype "${DTYPE}" \
  --per-device-eval-batch-size "${PER_DEVICE_EVAL_BATCH_SIZE}" \
  --dataloader-num-workers "${DATALOADER_NUM_WORKERS}" \
  --max-caption-tokens "${MAX_CAPTION_TOKENS}" \
  --num-shards "${NUM_SHARDS}" \
  --shard-index "${SHARD_INDEX}" \
  --guidance-level "${GUIDANCE_LEVEL}" \
  --save-images "${SAVE_IMAGES}" \
  --save-limit "${SAVE_LIMIT}" \
  --save-fixed-first-n "${SAVE_FIXED_FIRST_N}" \
  --report-losses "${REPORT_LOSSES}" \
  --coco-instances "${COCO_INSTANCES}" \
  --eval-slot-attention "${EVAL_SLOT_ATTENTION}" \
  --attn-threshold "${ATTN_THRESHOLD}" \
  --attn-thresholds "${ATTN_THRESHOLDS}" \
  --segmentation-bg-threshold "${SEGMENTATION_BG_THRESHOLD}" \
  --segmentation-bg-thresholds "${SEGMENTATION_BG_THRESHOLDS}" \
  --save-attn-maps "${SAVE_ATTN_MAPS}" \
  --save-attn-limit "${SAVE_ATTN_LIMIT}" \
  --diffusion-steps "${DIFFUSION_STEPS}" \
  --torch-compile "${TORCH_COMPILE}" \
  ${MAX_SAMPLES:+--max-samples "${MAX_SAMPLES}"}
