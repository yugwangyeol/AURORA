#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

MODEL_PATH="${MODEL_PATH:-/home/jovyan/AURORA/checkpoints/aurora_refcoco_singlephase_slot16_lora_stage1_fp32/best-checkpoint}"
OUTPUT_DIR="${OUTPUT_DIR:-/home/jovyan/AURORA/outputs/slot16_lora_best_eval}"
CAPTIONS_JSONL="${CAPTIONS_JSONL:-/home/jovyan/AURORA/outputs/stagea_object_captions_val2017_refexp/predictions.jsonl}"
IMAGE_DIR="${IMAGE_DIR:-/home/jovyan/data/coco/val2017}"
CUDA_DEVICE="${CUDA_DEVICE:-1}"
NUM_SHARDS="${NUM_SHARDS:-1}"
SHARD_INDEX="${SHARD_INDEX:-0}"
GUIDANCE_LEVEL="${GUIDANCE_LEVEL:-1.0}"
DIFFUSION_STEPS="${DIFFUSION_STEPS:-20}"
SLOT_MERGE_MODE="${SLOT_MERGE_MODE:-mean}"

PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}" \
/home/jovyan/.conda/envs/scale_rae/bin/python \
  "${SCRIPT_DIR}/eval_captionslot_checkpoint.py" \
  --model-path "${MODEL_PATH}" \
  --image-dir "${IMAGE_DIR}" \
  --captions-jsonl "${CAPTIONS_JSONL}" \
  --output-dir "${OUTPUT_DIR}" \
  --device cuda \
  --dtype fp32 \
  --per-device-eval-batch-size 8 \
  --dataloader-num-workers 4 \
  --max-caption-tokens 192 \
  --guidance-level "${GUIDANCE_LEVEL}" \
  --diffusion-steps "${DIFFUSION_STEPS}" \
  --eval-slot-attention 1 \
  --attn-threshold 0.5 \
  --attn-thresholds "0.2,0.3" \
  --segmentation-bg-threshold 0.5 \
  --segmentation-bg-thresholds "0.05,0.1,0.15,0.2,0.3" \
  --save-images 1 \
  --save-limit 100 \
  --save-fixed-first-n 50 \
  --save-attn-maps 1 \
  --save-attn-limit 100 \
  --report-losses 1 \
  --torch-compile 0 \
  --num-shards "${NUM_SHARDS}" \
  --shard-index "${SHARD_INDEX}" \
  --slot-merge-mode "${SLOT_MERGE_MODE}" \
  ${MAX_SAMPLES:+--max-samples "${MAX_SAMPLES}"}
