#!/usr/bin/env bash
# Full-style CaptionSlot eval for low-lr checkpoints using v10_full_val2017 captions.
# The v10 jsonl stores absolute image paths from another home directory, so this
# wrapper rewrites image paths to IMAGE_DIR by basename before launching eval.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

VARIANT="${VARIANT:-continue}"  # continue | attnfix
case "${VARIANT}" in
  continue)
    MODEL_PATH_DEFAULT="/home/jovyan/AURORA/checkpoints/aurora_refcoco_singlephase_maxpool_lora_continue_lowlr_fp32/best-checkpoint"
    OUTPUT_NAME_DEFAULT="aurora_refcoco_singlephase_maxpool_lora_continue_lowlr_fp32_v10_full_val2017_fullstyle_eval"
    ;;
  attnfix)
    MODEL_PATH_DEFAULT="/home/jovyan/AURORA/checkpoints/aurora_refcoco_singlephase_maxpool_lora_attnfix_lowlr_fp32/best-checkpoint"
    OUTPUT_NAME_DEFAULT="aurora_refcoco_singlephase_maxpool_lora_attnfix_lowlr_fp32_v10_full_val2017_fullstyle_eval"
    ;;
  *)
    echo "[run_eval_lowlr_v10_full_val2017] unknown VARIANT=${VARIANT}; use continue or attnfix" >&2
    exit 1
    ;;
esac

MODEL_PATH="${MODEL_PATH:-${MODEL_PATH_DEFAULT}}"
IMAGE_DIR="${IMAGE_DIR:-/home/jovyan/data/coco/val2017}"
CAPTION_OUTPUT_DIR="${CAPTION_OUTPUT_DIR:-/home/jovyan/AURORA/outputs/v10_full_val2017}"
SOURCE_CAPTIONS_JSONL="${SOURCE_CAPTIONS_JSONL:-${CAPTION_OUTPUT_DIR}/predictions.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-/home/jovyan/AURORA/outputs/${OUTPUT_NAME_DEFAULT}}"
NORMALIZED_CAPTIONS_JSONL="${NORMALIZED_CAPTIONS_JSONL:-${OUTPUT_DIR}/captions_v10_full_val2017_localpaths.jsonl}"

CUDA_DEVICE="${CUDA_DEVICE:-0}"
DTYPE="${DTYPE:-fp32}"
PER_DEVICE_EVAL_BATCH_SIZE="${PER_DEVICE_EVAL_BATCH_SIZE:-8}"
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-4}"
MAX_CAPTION_TOKENS="${MAX_CAPTION_TOKENS:-192}"
GUIDANCE_LEVEL="${GUIDANCE_LEVEL:-1.0}"
DIFFUSION_STEPS="${DIFFUSION_STEPS:-20}"
SLOT_MERGE_MODE="${SLOT_MERGE_MODE:-mean}"
ATTN_THRESHOLD="${ATTN_THRESHOLD:-0.5}"
ATTN_THRESHOLDS="${ATTN_THRESHOLDS:-0.2,0.3}"
ATTN_TEMPERATURE="${ATTN_TEMPERATURE:-1.0}"
SEGMENTATION_BG_THRESHOLD="${SEGMENTATION_BG_THRESHOLD:-0.5}"
SEGMENTATION_BG_THRESHOLDS="${SEGMENTATION_BG_THRESHOLDS:-0.05,0.1,0.15,0.2,0.3}"
GT_MIN_AREA_PCT="${GT_MIN_AREA_PCT:-0.0}"
SAVE_IMAGES="${SAVE_IMAGES:-1}"
SAVE_LIMIT="${SAVE_LIMIT:-100}"
SAVE_FIXED_FIRST_N="${SAVE_FIXED_FIRST_N:-50}"
SAVE_ATTN_MAPS="${SAVE_ATTN_MAPS:-1}"
SAVE_ATTN_LIMIT="${SAVE_ATTN_LIMIT:-100}"
REPORT_LOSSES="${REPORT_LOSSES:-1}"
TORCH_COMPILE="${TORCH_COMPILE:-0}"

mkdir -p "${OUTPUT_DIR}"

PYTHONNOUSERSITE="${PYTHONNOUSERSITE:-1}" \
/home/jovyan/.conda/envs/scale_rae/bin/python - <<'PY' \
  "${SOURCE_CAPTIONS_JSONL}" "${NORMALIZED_CAPTIONS_JSONL}" "${IMAGE_DIR}"
import json
import os
import sys
from pathlib import Path

src = Path(sys.argv[1])
dst = Path(sys.argv[2])
image_dir = Path(sys.argv[3])
dst.parent.mkdir(parents=True, exist_ok=True)

count = 0
missing = 0
with src.open("r", encoding="utf-8") as fin, dst.open("w", encoding="utf-8") as fout:
    for line in fin:
        line = line.strip()
        if not line:
            continue
        item = json.loads(line)
        image_value = item.get("image") or item.get("file_name")
        if image_value is not None:
            local_path = image_dir / os.path.basename(str(image_value))
            item["image"] = str(local_path)
            item["file_name"] = local_path.name
            if not local_path.exists():
                missing += 1
        fout.write(json.dumps(item, ensure_ascii=False) + "\n")
        count += 1

print(f"[run_eval_lowlr_v10_full_val2017] wrote {count} records to {dst}")
if missing:
    raise SystemExit(f"{missing} rewritten image paths do not exist under {image_dir}")
PY

PYTHONNOUSERSITE="${PYTHONNOUSERSITE:-1}" CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}" \
/home/jovyan/.conda/envs/scale_rae/bin/python \
  "${SCRIPT_DIR}/eval_captionslot_checkpoint.py" \
  --model-path "${MODEL_PATH}" \
  --image-dir "${IMAGE_DIR}" \
  --captions-jsonl "${NORMALIZED_CAPTIONS_JSONL}" \
  --output-dir "${OUTPUT_DIR}" \
  --device cuda \
  --dtype "${DTYPE}" \
  --per-device-eval-batch-size "${PER_DEVICE_EVAL_BATCH_SIZE}" \
  --dataloader-num-workers "${DATALOADER_NUM_WORKERS}" \
  --max-caption-tokens "${MAX_CAPTION_TOKENS}" \
  --guidance-level "${GUIDANCE_LEVEL}" \
  --diffusion-steps "${DIFFUSION_STEPS}" \
  --eval-slot-attention 1 \
  --attn-threshold "${ATTN_THRESHOLD}" \
  --attn-thresholds "${ATTN_THRESHOLDS}" \
  --attn-temperature "${ATTN_TEMPERATURE}" \
  --segmentation-bg-threshold "${SEGMENTATION_BG_THRESHOLD}" \
  --segmentation-bg-thresholds "${SEGMENTATION_BG_THRESHOLDS}" \
  --gt-min-area-pct "${GT_MIN_AREA_PCT}" \
  --save-images "${SAVE_IMAGES}" \
  --save-limit "${SAVE_LIMIT}" \
  --save-fixed-first-n "${SAVE_FIXED_FIRST_N}" \
  --save-attn-maps "${SAVE_ATTN_MAPS}" \
  --save-attn-limit "${SAVE_ATTN_LIMIT}" \
  --report-losses "${REPORT_LOSSES}" \
  --torch-compile "${TORCH_COMPILE}" \
  --slot-merge-mode "${SLOT_MERGE_MODE}" \
  ${MAX_SAMPLES:+--max-samples "${MAX_SAMPLES}"}
