#!/bin/bash
# =============================================================================
# PGOT v11 evaluation.
# Default is the train-matched main readout: threshold + mean merge.
# =============================================================================
set -euo pipefail

MODEL_PATH="${MODEL_PATH:-/home/jovyan/PGOT/checkpoints/pgot_main_v11_bal_bce_ce03_mean}"
VAL_JSONL="${VAL_JSONL:-/home/jovyan/PGOT/data/pgot_val.jsonl}"
READOUT="${READOUT:-threshold}"
OUTPUT_DIR="${OUTPUT_DIR:-/home/jovyan/PGOT/outputs/eval_pgot_v11_${READOUT}_$(basename "${MODEL_PATH}")}"
BATCH_SIZE="${BATCH_SIZE:-4}"
MAX_SAMPLES="${MAX_SAMPLES:-}"
COMPUTE_RFID="${COMPUTE_RFID:-0}"
GT_SOURCE="${GT_SOURCE:-coco_instance}"
EVAL_MERGE="${EVAL_MERGE:-mean}"
GRID_SIZE="${GRID_SIZE:-32}"
EVAL_SIZE="${EVAL_SIZE:-224}"
COCO_MASK_CACHE="${COCO_MASK_CACHE:-/home/jovyan/PGOT/data/coco_inst_mask_cache_coda256}"
IMAGE_PREPROCESS_MODE="${IMAGE_PREPROCESS_MODE:-default}"
CODA_CROP_SIZE="${CODA_CROP_SIZE:-512}"
BG_THRESHOLD="${BG_THRESHOLD:-0.05}"
DIFF_INFER_STEPS="${DIFF_INFER_STEPS:-25}"
GUIDANCE_SCALE="${GUIDANCE_SCALE:-2.5}"

if [ -d "${MODEL_PATH}" ] && [ ! -f "${MODEL_PATH}/config.json" ]; then
    LATEST_CHECKPOINT="$(find "${MODEL_PATH}" -maxdepth 1 -type d -name 'checkpoint-*' | sort -V | tail -n 1)"
    if [ -z "${LATEST_CHECKPOINT}" ]; then
        echo "ERROR: no config.json or checkpoint-* found under ${MODEL_PATH}" >&2
        exit 1
    fi
    echo "Resolved latest checkpoint: ${LATEST_CHECKPOINT}"
    MODEL_PATH="${LATEST_CHECKPOINT}"
fi

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

echo "===== PGOT v11 Eval ====="
echo "Ckpt:    ${MODEL_PATH}"
echo "Val:     ${VAL_JSONL}"
echo "Out:     ${OUTPUT_DIR}"
echo "Batch:   ${BATCH_SIZE}"
echo "GT:      ${GT_SOURCE}"
echo "Image:   ${IMAGE_PREPROCESS_MODE}"
echo "Readout: ${READOUT}"
echo "Merge:   ${EVAL_MERGE}"
echo "rFID:    ${COMPUTE_RFID}"
echo "========================="

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
    --bg_threshold "${BG_THRESHOLD}" \
    --dtype fp32 \
    --gt_source "${GT_SOURCE}" \
    --image_preprocess_mode "${IMAGE_PREPROCESS_MODE}" \
    --coda_crop_size "${CODA_CROP_SIZE}" \
    --readout "${READOUT}" \
    --eval_merge "${EVAL_MERGE}" \
    --coco_mask_cache "${COCO_MASK_CACHE}" \
    --diffusion_inference_steps "${DIFF_INFER_STEPS}" \
    --guidance_scale "${GUIDANCE_SCALE}" \
    "${EXTRA_ARGS[@]}" \
    2>&1 | tee "${OUTPUT_DIR}/eval.log"
