#!/bin/bash
# =============================================================================
# AURORA v2 Stage 2 training wrapper
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export MODEL_PATH="${MODEL_PATH:-/home/jovyan/AURORA/checkpoints/aurora_v2_stage1}"
export OUTPUT_DIR="${OUTPUT_DIR:-/home/jovyan/AURORA/checkpoints/aurora_v2_stage2}"
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-aurora_v2_stage2}"

export NUM_GPUS="${NUM_GPUS:-1}"
export MAX_STEPS="${MAX_STEPS:-10000}"
export PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-1}"
export PER_DEVICE_EVAL_BATCH_SIZE="${PER_DEVICE_EVAL_BATCH_SIZE:-1}"
export GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-1}"
export LOGGING_STEPS="${LOGGING_STEPS:-10}"
export EVAL_STEPS="${EVAL_STEPS:-100}"
export SAVE_STEPS="${SAVE_STEPS:-500}"
export DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-0}"
export REPORT_TO="${REPORT_TO:-wandb}"
export BF16="${BF16:-True}"

export AURORA_TRAINING_STAGE=2
export AURORA_INCLUDE_INPAINTING=True
export AURORA_TRAIN_DIFFUSION_CONDITION="${AURORA_TRAIN_DIFFUSION_CONDITION:-False}"
export AURORA_TRAIN_LATENT_QUERIES="${AURORA_TRAIN_LATENT_QUERIES:-True}"
export AURORA_LATENT_QUERY_LR="${AURORA_LATENT_QUERY_LR:-1e-6}"
export AURORA_EVAL_LOG_RECONSTRUCTIONS="${AURORA_EVAL_LOG_RECONSTRUCTIONS:-True}"
export AURORA_EVAL_LOG_ATTENTION_OVERLAYS="${AURORA_EVAL_LOG_ATTENTION_OVERLAYS:-True}"

export RECON_IMAGE_FOLDER="${RECON_IMAGE_FOLDER:-/home/jovyan/data/coco/train2017}"
export COCO_ANNOTATION="${COCO_ANNOTATION:-/home/jovyan/data/coco/annotations/instances_train2017.json}"
export INPAINT_DATA_PATH="${INPAINT_DATA_PATH:-/home/jovyan/processed_coco/training_data_v4_patch/training_manifest_patch_v4_0_to_None.json}"
export INPAINT_IMAGE_FOLDER="${INPAINT_IMAGE_FOLDER:-/home/jovyan/processed_coco/training_data_v4_patch}"

if [ ! -e "${MODEL_PATH}" ]; then
    echo "Stage 2 expects a stage 1 checkpoint/model path."
    echo "Set MODEL_PATH=/path/to/aurora_v2_stage1 (current: ${MODEL_PATH})"
    exit 1
fi

echo "===== AURORA Stage 2 ====="
echo "Checkpoint source: ${MODEL_PATH}"
echo "Output dir:        ${OUTPUT_DIR}"
echo "Max steps:         ${MAX_STEPS}"
echo "Inpaint manifest:  ${INPAINT_DATA_PATH}"
echo "Inpaint images:    ${INPAINT_IMAGE_FOLDER}"
echo "=========================="

exec bash "${SCRIPT_DIR}/train_aurora.sh"
