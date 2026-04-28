#!/bin/bash
# =============================================================================
# CaptionSlot Stage 1 wrapper: first-slot + one-register + head-prior bias
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export MODEL_PATH="${MODEL_PATH:-/home/jovyan/data/Scale-RAE-Qwen1.5B_DiT2.4B}"
export OUTPUT_DIR="${OUTPUT_DIR:-/home/jovyan/AURORA/checkpoints/captionslot_firstslot_headprior_s1.0_stage1}"
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-captionslot_firstslot_headprior_s1.0_stage1}"

export NUM_GPUS="${NUM_GPUS:-2}"
export MAX_STEPS="${MAX_STEPS:-20000}"
export NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-1}"
export PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-8}"
export PER_DEVICE_EVAL_BATCH_SIZE="${PER_DEVICE_EVAL_BATCH_SIZE:-4}"
export GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-2}"
export LOGGING_STEPS="${LOGGING_STEPS:-10}"
export EVAL_STEPS="${EVAL_STEPS:-1000}"
export SAVE_STEPS="${SAVE_STEPS:-1000}"
export DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-8}"
export REPORT_TO="${REPORT_TO:-wandb}"
export BF16="${BF16:-False}"
export LEARNING_RATE="${LEARNING_RATE:-5e-5}"
export DIFF_HEAD_LR="${DIFF_HEAD_LR:-5e-6}"
export CAPTIONSLOT_LATENT_QUERY_LR="${CAPTIONSLOT_LATENT_QUERY_LR:-1e-6}"

export CAPTIONSLOT_TRAINING_STAGE=1
export CAPTIONSLOT_MAX_SLOTS="${CAPTIONSLOT_MAX_SLOTS:-1}"
export CAPTIONSLOT_N_REGISTER="${CAPTIONSLOT_N_REGISTER:-1}"
export CAPTIONSLOT_CMD_LENGTH="${CAPTIONSLOT_CMD_LENGTH:-8}"
export CAPTIONSLOT_RECON_LOSS_WEIGHT="${CAPTIONSLOT_RECON_LOSS_WEIGHT:-1.0}"
export CAPTIONSLOT_CAPTION_LOSS_WEIGHT="${CAPTIONSLOT_CAPTION_LOSS_WEIGHT:-0.0}"
export CAPTIONSLOT_DIVERSITY_LOSS_WEIGHT="${CAPTIONSLOT_DIVERSITY_LOSS_WEIGHT:-0.0}"
export CAPTIONSLOT_TRAIN_LATENT_QUERIES="${CAPTIONSLOT_TRAIN_LATENT_QUERIES:-False}"
export CAPTIONSLOT_CONDITION_GATE_INIT="${CAPTIONSLOT_CONDITION_GATE_INIT:-1.0}"
export CAPTIONSLOT_ATTENTION_USE_LAYERNORM="${CAPTIONSLOT_ATTENTION_USE_LAYERNORM:-True}"
export CAPTIONSLOT_ATTENTION_TEMPERATURE="${CAPTIONSLOT_ATTENTION_TEMPERATURE:-1.0}"
export CAPTIONSLOT_PRIOR_BIAS_SCALE="${CAPTIONSLOT_PRIOR_BIAS_SCALE:-1.0}"
export CAPTIONSLOT_EVAL_NUM_IMAGES="${CAPTIONSLOT_EVAL_NUM_IMAGES:-64}"

export CAPTIONSLOT_IMAGE_FOLDER="${CAPTIONSLOT_IMAGE_FOLDER:-/home/jovyan/data/coco/train2017}"
export CAPTIONSLOT_ANNOTATION_PATH="${CAPTIONSLOT_ANNOTATION_PATH:-/home/jovyan/AURORA/outputs/stagea_captionslot_train_cache_train2017/captionslot_annotations.json}"
export MAX_CAPTION_TOKENS="${MAX_CAPTION_TOKENS:-64}"
export DIFFUSION_NORM_STATS_PATH="${DIFFUSION_NORM_STATS_PATH:-/home/jovyan/data/siglip2_bn_stats.pt}"

echo "===== CaptionSlot Stage 1: First-Slot Head-Prior ====="
echo "Checkpoint source: ${MODEL_PATH}"
echo "Output dir:        ${OUTPUT_DIR}"
echo "GPUs:              ${NUM_GPUS}"
echo "Per-device batch:  ${PER_DEVICE_TRAIN_BATCH_SIZE}"
echo "Grad accum:        ${GRADIENT_ACCUMULATION_STEPS}"
echo "BF16:              ${BF16}"
echo "Max slots:         ${CAPTIONSLOT_MAX_SLOTS}"
echo "Registers:         ${CAPTIONSLOT_N_REGISTER}"
echo "Prior bias scale:  ${CAPTIONSLOT_PRIOR_BIAS_SCALE}"
echo "Train images:      ${CAPTIONSLOT_IMAGE_FOLDER}"
echo "Annotation path:   ${CAPTIONSLOT_ANNOTATION_PATH}"
echo "==============================================="

exec bash "${SCRIPT_DIR}/train_aurora.sh"
