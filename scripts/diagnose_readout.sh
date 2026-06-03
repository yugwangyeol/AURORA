#!/bin/bash
# =============================================================================
# PGOT Step-0 readout diagnostics.
#
# Usage:
#   bash scripts/diagnose_readout.sh
#   MAX_SAMPLES=2338 bash scripts/diagnose_readout.sh
# =============================================================================
set -e

MODEL_PATH="${MODEL_PATH:-/home/jovyan/PGOT/checkpoints/pgot_main_v3/checkpoint-14000}"
VAL_JSONL="${VAL_JSONL:-/home/jovyan/PGOT/data/pgot_val.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-/home/jovyan/PGOT/outputs/diagnose_readout_$(basename "${MODEL_PATH}")}"
MAX_SAMPLES="${MAX_SAMPLES:-500}"
BATCH_SIZE="${BATCH_SIZE:-4}"
GRID_SIZE="${GRID_SIZE:-32}"
MERGE="${MERGE:-max}"
TEMP="${TEMP:-1.0}"
BG_THRESHOLDS="${BG_THRESHOLDS:-0.05,0.10,0.15,0.20,0.30}"

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

echo "===== PGOT Step-0 Readout Diagnostics ====="
echo "Ckpt:        ${MODEL_PATH}"
echo "Max samples: ${MAX_SAMPLES}"
echo "merge/temp:  ${MERGE} / ${TEMP}"
echo "bg_thr:      ${BG_THRESHOLDS}"
echo "Output:      ${OUTPUT_DIR}"
echo "==========================================="

"${PYTHON}" "${PROJECT_ROOT}/pgot/eval/diagnose_readout.py" \
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
    --dtype fp32 \
    --merge "${MERGE}" \
    --temp "${TEMP}" \
    --bg_thresholds "${BG_THRESHOLDS}" \
    2>&1 | tee "${OUTPUT_DIR}/diagnose.log"
