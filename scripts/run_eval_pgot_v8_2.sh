#!/bin/bash
# =============================================================================
# PGOT v8.2 evaluation.
# Defaults to the strict training-matched spatial readout used by V8.2.
# =============================================================================
set -euo pipefail

MODEL_PATH="${MODEL_PATH:-/home/jovyan/PGOT/checkpoints/pgot_main_v8_2_spatial_outside_log}"
VAL_JSONL="${VAL_JSONL:-/home/jovyan/PGOT/data/pgot_val.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-/home/jovyan/PGOT/outputs/eval_pgot_v8_2_$(basename "${MODEL_PATH}")}"
BATCH_SIZE="${BATCH_SIZE:-4}"
MAX_SAMPLES="${MAX_SAMPLES:-}"
COMPUTE_RFID="${COMPUTE_RFID:-0}"
GT_SOURCE="${GT_SOURCE:-pix2cap_panoptic}"
READOUT="${READOUT:-spatial_trainmatch}"
SPATIAL_TEMPERATURE="${SPATIAL_TEMPERATURE:-1.0}"
EVAL_MERGE="${EVAL_MERGE:-mean}"
GRID_SIZE="${GRID_SIZE:-32}"
EVAL_SIZE="${EVAL_SIZE:-224}"
COCO_MASK_CACHE="${COCO_MASK_CACHE:-/home/jovyan/PGOT/data/coco_inst_mask_cache_coda256}"
DIFF_INFER_STEPS="${DIFF_INFER_STEPS:-25}"
GUIDANCE_SCALE="${GUIDANCE_SCALE:-2.5}"

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

echo "===== PGOT v8.2 Eval ====="
echo "Ckpt:    ${MODEL_PATH}"
echo "Val:     ${VAL_JSONL}"
echo "Out:     ${OUTPUT_DIR}"
echo "Batch:   ${BATCH_SIZE}"
echo "Grid:    ${GRID_SIZE}"
echo "GT:      ${GT_SOURCE}"
echo "Readout: ${READOUT}"
echo "Merge:   ${EVAL_MERGE}"
if [ "${READOUT}" = "spatial_trainmatch" ]; then
    echo "Match:   patch-axis softmax + checkpoint temp + mean merge"
fi
echo "rFID:    ${COMPUTE_RFID}"
echo "=========================="

EXTRA_ARGS=()
if [ -n "${MAX_SAMPLES}" ]; then
    EXTRA_ARGS+=(--max_samples "${MAX_SAMPLES}")
fi
if [ "${COMPUTE_RFID}" = "1" ]; then
    EXTRA_ARGS+=(--compute_rfid)
fi

"${PYTHON}" "${PROJECT_ROOT}/pgot/eval/run_eval.py" \
    --model_path "${MODEL_PATH}" \
    --val_jsonl "${VAL_JSONL}" \
    --output_dir "${OUTPUT_DIR}" \
    --batch_size "${BATCH_SIZE}" \
    --num_workers 4 \
    --grid_size "${GRID_SIZE}" \
    --eval_size "${EVAL_SIZE}" \
    --max_caption_tokens 2048 \
    --n_ovt_per_object 2 \
    --max_objects 50 \
    --bg_threshold 0.05 \
    --dtype fp32 \
    --gt_source "${GT_SOURCE}" \
    --readout "${READOUT}" \
    --eval_merge "${EVAL_MERGE}" \
    --spatial_temperature "${SPATIAL_TEMPERATURE}" \
    --coco_mask_cache "${COCO_MASK_CACHE}" \
    --diffusion_inference_steps "${DIFF_INFER_STEPS}" \
    --guidance_scale "${GUIDANCE_SCALE}" \
    "${EXTRA_ARGS[@]}" \
    2>&1 | tee "${OUTPUT_DIR}/eval.log"
