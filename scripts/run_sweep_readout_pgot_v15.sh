#!/bin/bash
# PGOT v15 post-hoc readout sweep. No retraining; caches OVT logits once, then
# sweeps threshold/readout calibration for fARI and top-k mBO/mIoU.
set -euo pipefail

MODEL_PATH="${MODEL_PATH:-/home/jovyan/PGOT/checkpoints/pgot_main_v15_bce_bottleneck}"
VAL_JSONL="${VAL_JSONL:-/home/jovyan/PGOT/data/pgot_val.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-/home/jovyan/PGOT/outputs/sweep_readout_pgot_v15_$(basename "${MODEL_PATH}")}"
BATCH_SIZE="${BATCH_SIZE:-4}"
NUM_WORKERS="${NUM_WORKERS:-4}"
MAX_SAMPLES="${MAX_SAMPLES:-}"
GT_SOURCE="${GT_SOURCE:-coco_instance}"
COCO_MASK_CACHE="${COCO_MASK_CACHE:-/home/jovyan/PGOT/data/coco_inst_mask_cache_coda256}"
GRID_SIZE="${GRID_SIZE:-32}"
EVAL_SIZE="${EVAL_SIZE:-256}"
IMAGE_PREPROCESS_MODE="${IMAGE_PREPROCESS_MODE:-coda_center_crop}"
CODA_CROP_SIZE="${CODA_CROP_SIZE:-512}"
DTYPE="${DTYPE:-fp32}"

BG_THRESHOLDS="${BG_THRESHOLDS:-0.0,0.01,0.03,0.05,0.07,0.10,0.15,0.20}"
TEMPS="${TEMPS:-0.3,0.5,0.7,1.0,1.3}"
MERGES="${MERGES:-mean,max}"
COMPETITIONS="${COMPETITIONS:-sigmoid,softmax}"
USE_BGS="${USE_BGS:-1,0}"
TOPK_FULL_METRIC="${TOPK_FULL_METRIC:-10}"
PARETO="${PARETO:-0}"

mkdir -p "${OUTPUT_DIR}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

SCALE_RAE_ENV="${HOME}/.conda/envs/scale_rae"
PYTHON="${SCALE_RAE_ENV}/bin/python"
CONDA_LIB="$(dirname "${PYTHON}")/../lib"
export LD_LIBRARY_PATH="${CONDA_LIB}:${LD_LIBRARY_PATH:-}"
export PYTHONNOUSERSITE=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

EXTRA_ARGS=()
if [ -n "${MAX_SAMPLES}" ]; then
    EXTRA_ARGS+=(--max_samples "${MAX_SAMPLES}")
fi
if [ "${PARETO}" = "1" ]; then
    EXTRA_ARGS+=(--pareto)
fi

echo "===== PGOT v15 readout sweep ====="
echo "Ckpt:     ${MODEL_PATH}"
echo "Out:      ${OUTPUT_DIR}"
echo "GT:       ${GT_SOURCE}"
echo "Image:    ${IMAGE_PREPROCESS_MODE}"
echo "Grid:     ${GRID_SIZE}"
echo "BG:       ${BG_THRESHOLDS}"
echo "Temps:    ${TEMPS}"
echo "Merges:   ${MERGES}"
echo "Comp:     ${COMPETITIONS}"
echo "Use BG:   ${USE_BGS}"
echo "=================================="

"${PYTHON}" -m pgot.eval.sweep_readout \
    --model_path "${MODEL_PATH}" \
    --val_jsonl "${VAL_JSONL}" \
    --output_dir "${OUTPUT_DIR}" \
    --batch_size "${BATCH_SIZE}" \
    --num_workers "${NUM_WORKERS}" \
    --grid_size "${GRID_SIZE}" \
    --eval_size "${EVAL_SIZE}" \
    --max_caption_tokens 2048 \
    --n_ovt_per_object 2 \
    --max_objects 50 \
    --gt_source "${GT_SOURCE}" \
    --coco_mask_cache "${COCO_MASK_CACHE}" \
    --image_preprocess_mode "${IMAGE_PREPROCESS_MODE}" \
    --coda_crop_size "${CODA_CROP_SIZE}" \
    --dtype "${DTYPE}" \
    --bg_thresholds "${BG_THRESHOLDS}" \
    --temps "${TEMPS}" \
    --merges "${MERGES}" \
    --competitions "${COMPETITIONS}" \
    --use_bgs "${USE_BGS}" \
    --topk_full_metric "${TOPK_FULL_METRIC}" \
    "${EXTRA_ARGS[@]}" \
    2>&1 | tee "${OUTPUT_DIR}/sweep.log"
