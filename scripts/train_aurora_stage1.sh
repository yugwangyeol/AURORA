#!/bin/bash
# =============================================================================
# AURORA v2 Stage 1 training wrapper
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export MODEL_PATH="${MODEL_PATH:-/home/jovyan/data/Scale-RAE-Qwen1.5B_DiT2.4B}"
export OUTPUT_DIR="${OUTPUT_DIR:-/home/jovyan/AURORA/checkpoints/aurora_v2_stage1}"
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-aurora_v2_stage1}"

export NUM_GPUS="${NUM_GPUS:-2}"
export MAX_STEPS="${MAX_STEPS:--1}"
export NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-100}"
export PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-32}"
export PER_DEVICE_EVAL_BATCH_SIZE="${PER_DEVICE_EVAL_BATCH_SIZE:-8}"
export GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-4}"
export LOGGING_STEPS="${LOGGING_STEPS:-10}"
export EVAL_STEPS="${EVAL_STEPS:-1000}"
export SAVE_STEPS="${SAVE_STEPS:-1000}"
export DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-8}"
export REPORT_TO="${REPORT_TO:-wandb}"
export BF16="${BF16:-True}"
export LEARNING_RATE="${LEARNING_RATE:-5e-5}"
export DIFF_HEAD_LR="${DIFF_HEAD_LR:-5e-6}"
export AURORA_LATENT_QUERY_LR="${AURORA_LATENT_QUERY_LR:-1e-6}"

export AURORA_TRAINING_STAGE=1
export AURORA_INCLUDE_INPAINTING=False
export AURORA_MASK_LOSS_WEIGHT="${AURORA_MASK_LOSS_WEIGHT:-0.5}"
export AURORA_DIVERSITY_LOSS_WEIGHT="${AURORA_DIVERSITY_LOSS_WEIGHT:-0.02}"
export AURORA_TRAIN_DIFFUSION_CONDITION="${AURORA_TRAIN_DIFFUSION_CONDITION:-True}"
export AURORA_TRAIN_LATENT_QUERIES="${AURORA_TRAIN_LATENT_QUERIES:-False}"
export AURORA_CONDITION_GATE_INIT="${AURORA_CONDITION_GATE_INIT:-0.1}"
export AURORA_ATTENTION_USE_LAYERNORM="${AURORA_ATTENTION_USE_LAYERNORM:-True}"
export AURORA_ATTENTION_TEMPERATURE="${AURORA_ATTENTION_TEMPERATURE:-1.0}"
export AURORA_EVAL_NUM_IMAGES="${AURORA_EVAL_NUM_IMAGES:-32}"
export AURORA_EVAL_LOG_IMAGE_COUNT="${AURORA_EVAL_LOG_IMAGE_COUNT:-8}"
export AURORA_EVAL_VISUAL_BATCH_SIZE="${AURORA_EVAL_VISUAL_BATCH_SIZE:-2}"
export AURORA_EVAL_LOG_RECONSTRUCTIONS="${AURORA_EVAL_LOG_RECONSTRUCTIONS:-True}"
export AURORA_EVAL_LOG_ATTENTION_OVERLAYS="${AURORA_EVAL_LOG_ATTENTION_OVERLAYS:-False}"
export AURORA_EVAL_ATTENTION_OVERLAY_COUNT="${AURORA_EVAL_ATTENTION_OVERLAY_COUNT:-8}"

export RECON_IMAGE_FOLDER="${RECON_IMAGE_FOLDER:-/home/jovyan/data/coco/train2017}"
export COCO_ANNOTATION="${COCO_ANNOTATION:-/home/jovyan/data/coco/annotations/instances_train2017.json}"
export DIFFUSION_NORM_STATS_PATH="${DIFFUSION_NORM_STATS_PATH:-/home/jovyan/data/siglip2_bn_stats.pt}"

echo "===== AURORA Stage 1 ====="
echo "Checkpoint source: ${MODEL_PATH}"
echo "Output dir:        ${OUTPUT_DIR}"
echo "GPUs:              ${NUM_GPUS}"
echo "Per-device batch:  ${PER_DEVICE_TRAIN_BATCH_SIZE}"
echo "Grad accum:        ${GRADIENT_ACCUMULATION_STEPS}"
echo "Effective batch:   $((NUM_GPUS * PER_DEVICE_TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS))"
echo "Learning rate:     ${LEARNING_RATE}"
echo "Diff head lr:      ${DIFF_HEAD_LR}"
echo "Latent query lr:   ${AURORA_LATENT_QUERY_LR}"
echo "Mask weight:       ${AURORA_MASK_LOSS_WEIGHT}"
echo "Diversity weight:  ${AURORA_DIVERSITY_LOSS_WEIGHT}"
echo "Diff cond train:   ${AURORA_TRAIN_DIFFUSION_CONDITION}"
echo "Latent query train:${AURORA_TRAIN_LATENT_QUERIES}"
echo "Condition gate:    ${AURORA_CONDITION_GATE_INIT}"
echo "Epochs:            ${NUM_TRAIN_EPOCHS}"
echo "Max steps:         ${MAX_STEPS}"
echo "Reconstruction:    ${RECON_IMAGE_FOLDER}"
echo "COCO annotation:   ${COCO_ANNOTATION}"
echo "Norm stats:        ${DIFFUSION_NORM_STATS_PATH}"
echo "=========================="

exec bash "${SCRIPT_DIR}/train_aurora.sh"
