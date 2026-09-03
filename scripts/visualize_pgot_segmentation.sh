#!/usr/bin/env bash
# Visualize how PGOT partitions an image: GT vs predicted object segmentation
# overlays, plus per-object attention maps.
#
# Usage:
#   MODEL_PATH=... SAMPLES="0 500 1020" bash scripts/visualize_pgot_segmentation.sh
#
# One model load per sample (~4 min each), so keep SAMPLES short.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_PATH="${MODEL_PATH:-${PROJECT_ROOT}/checkpoints/pgot_e11_capacity_dit16/checkpoint-10000}"
MODEL_LABEL="${MODEL_LABEL:-$(basename "$(dirname "${MODEL_PATH}")")}"
VAL_JSONL="${VAL_JSONL:-${PROJECT_ROOT}/data/pgot_pix2cap_generated_val5k.jsonl}"
COCO_MASK_CACHE="${COCO_MASK_CACHE:-${PROJECT_ROOT}/data/coco_inst_mask_cache_coda512}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/outputs/segvis_${MODEL_LABEL}}"
SAMPLES="${SAMPLES:-0 500 1020 1407}"
PYTHON="${PYTHON:-/home/jovyan/.conda/envs/scale_rae/bin/python}"

for path in "${MODEL_PATH}" "${VAL_JSONL}" "${COCO_MASK_CACHE}/meta.json"; do
    test -e "${path}" || { echo "Missing required path: ${path}" >&2; exit 1; }
done
export CUDA_VISIBLE_DEVICES="${GPU:-0}"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export PYTHONNOUSERSITE=1
export LD_LIBRARY_PATH="$(dirname "${PYTHON}")/../lib:${LD_LIBRARY_PATH:-}"

mkdir -p "${OUTPUT_ROOT}"
for idx in ${SAMPLES}; do
    echo "=== sample ${idx} ==="
    "${PYTHON}" -m pgot.eval.visualize_ovt_overlays \
        --model "${MODEL_LABEL}|${MODEL_PATH}|${READOUT:-ovt_owner}|${MERGE:-mean}" \
        --sample_index "${idx}" \
        --val_jsonl "${VAL_JSONL}" \
        --output_dir "${OUTPUT_ROOT}/sample${idx}" \
        --gt_source coco_instance --coco_mask_cache "${COCO_MASK_CACHE}" \
        --image_preprocess_mode coda_center_crop --coda_crop_size 512 \
        --grid_size 32 --eval_size 512 \
        --max_caption_tokens 1024 --n_ovt_per_object 1 --max_objects 50 \
        --dtype "${DTYPE:-fp32}"
done
echo "Wrote: ${OUTPUT_ROOT}"
