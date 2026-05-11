#!/usr/bin/env bash
# Eval with all tricks enabled:
#   --gt-min-area-pct 0.005  : filter GT instances <0.5% of image (impossible to segment at 16x16 patches)
#   --raw-slot-argmax 0      : still use per-object MEAN merge (set to 1 to try pure-slot argmax)
#   --attn-temperature 0.5   : sharpen attention maps (halve temperature)
#   --segmentation-bg-thresholds "0.05,0.1,0.15,0.2,0.3"  : sweep for best MBO/mIoU
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

MODEL_PATH="${MODEL_PATH:-/home/jovyan/AURORA/checkpoints/aurora_refcoco_singlephase_slot16_lora_stage1_fp32/best-checkpoint}"
OUTPUT_DIR="${OUTPUT_DIR:-/home/jovyan/AURORA/outputs/slot16_lora_tricks_eval}"
CAPTIONS_JSONL="${CAPTIONS_JSONL:-/home/jovyan/AURORA/outputs/stagea_object_captions_val2017_refexp/predictions.jsonl}"
IMAGE_DIR="${IMAGE_DIR:-/home/jovyan/data/coco/val2017}"
CUDA_DEVICE="${CUDA_DEVICE:-1}"

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
  --guidance-level 1.0 \
  --diffusion-steps 20 \
  --eval-slot-attention 1 \
  --attn-threshold 0.5 \
  --attn-thresholds "0.2,0.3" \
  --attn-temperature 0.5 \
  --segmentation-bg-threshold 0.5 \
  --segmentation-bg-thresholds "0.05,0.1,0.15,0.2,0.3" \
  --gt-min-area-pct 0.005 \
  --raw-slot-argmax 0 \
  --save-images 1 \
  --save-limit "${SAVE_LIMIT:-100}" \
  --save-fixed-first-n 50 \
  --save-attn-maps 1 \
  --save-attn-limit "${SAVE_ATTN_LIMIT:-100}" \
  --report-losses 1 \
  --torch-compile 0 \
  ${MAX_SAMPLES:+--max-samples "${MAX_SAMPLES}"}
