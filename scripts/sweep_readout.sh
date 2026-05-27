#!/bin/bash
# =============================================================================
# PGOT — Post-hoc readout sweep (NO retraining).
# Single forward over the eval set, then sweep readout configs to maximize fARI.
#
# Usage:
#   bash scripts/sweep_readout.sh                       # 500-sample diagnostic
#   MAX_SAMPLES=2338 bash scripts/sweep_readout.sh      # full set
# =============================================================================
set -e

MODEL_PATH="${MODEL_PATH:-/home/jovyan/PGOT/checkpoints/pgot_main_v3/checkpoint-14000}"
VAL_JSONL="${VAL_JSONL:-/home/jovyan/PGOT/data/pgot_val.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-/home/jovyan/PGOT/outputs/sweep_readout_$(basename ${MODEL_PATH})}"
MAX_SAMPLES="${MAX_SAMPLES:-500}"
BATCH_SIZE="${BATCH_SIZE:-4}"
GT_SOURCE="${GT_SOURCE:-coco_instance}"
GRID_SIZE="${GRID_SIZE:-32}"

# Sweep grids (override-able)
BG_THRESHOLDS="${BG_THRESHOLDS:-0.0,0.01,0.03,0.05,0.10,0.15,0.20}"
TEMPS="${TEMPS:-0.3,0.5,0.7,1.0}"
MERGES="${MERGES:-mean,max}"
COMPETITIONS="${COMPETITIONS:-sigmoid,softmax}"
USE_BGS="${USE_BGS:-1,0}"

mkdir -p "${OUTPUT_DIR}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH}"

SCALE_RAE_ENV="${HOME}/.conda/envs/scale_rae"
PYTHON="${SCALE_RAE_ENV}/bin/python"
CONDA_LIB="$(dirname "${PYTHON}")/../lib"
export LD_LIBRARY_PATH="${CONDA_LIB}:${LD_LIBRARY_PATH}"
export PYTHONNOUSERSITE=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

echo "===== PGOT Post-hoc Readout Sweep ====="
echo "Ckpt:        ${MODEL_PATH}"
echo "GT source:   ${GT_SOURCE}"
echo "Max samples: ${MAX_SAMPLES}"
echo "bg_thr:      ${BG_THRESHOLDS}"
echo "temps:       ${TEMPS}"
echo "merges:      ${MERGES}"
echo "comps:       ${COMPETITIONS}"
echo "use_bgs:     ${USE_BGS}"
echo "Output:      ${OUTPUT_DIR}"
echo "======================================="

"${PYTHON}" "${PROJECT_ROOT}/pgot/eval/sweep_readout.py" \
    --model_path "${MODEL_PATH}" \
    --val_jsonl "${VAL_JSONL}" \
    --output_dir "${OUTPUT_DIR}" \
    --batch_size "${BATCH_SIZE}" \
    --num_workers 4 \
    --max_samples "${MAX_SAMPLES}" \
    --grid_size "${GRID_SIZE}" \
    --max_caption_tokens 2048 \
    --n_ovt_per_object 2 \
    --max_objects 50 \
    --gt_source "${GT_SOURCE}" \
    --dtype fp32 \
    --bg_thresholds "${BG_THRESHOLDS}" \
    --temps "${TEMPS}" \
    --merges "${MERGES}" \
    --competitions "${COMPETITIONS}" \
    --use_bgs "${USE_BGS}" \
    --topk_full_metric 5 \
    ${PARETO:+--pareto} \
    2>&1 | tee "${OUTPUT_DIR}/sweep.log"
