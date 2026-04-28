#!/bin/bash
# Metrics-only 2-GPU eval built on the older eval_captionslot_checkpoint.py path.
# Computes: rFID / PSNR / SSIM / MSE
# Keeps fp32 by default and preserves the older path's torch.compile behavior.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

find_latest_checkpoint() {
  local run_dir="$1"
  find "${run_dir}" -maxdepth 1 -type d -name 'checkpoint-*' | sort -V | tail -n 1
}

export RUN_DIR="${RUN_DIR:-/home/jovyan/AURORA/checkpoints/captionslot_stage1_slotxattn_b200x2_v2}"
MODEL_PATH_INPUT="${MODEL_PATH:-${RUN_DIR}}"
if [[ -d "${MODEL_PATH_INPUT}" && ! -f "${MODEL_PATH_INPUT}/config.json" ]]; then
  RESOLVED_MODEL_PATH="$(find_latest_checkpoint "${MODEL_PATH_INPUT}")"
  if [[ -z "${RESOLVED_MODEL_PATH}" ]]; then
    echo "[run_eval_captionslot_checkpoint_metrics_parallel] no checkpoint-* found under ${MODEL_PATH_INPUT}" >&2
    exit 1
  fi
else
  RESOLVED_MODEL_PATH="${MODEL_PATH_INPUT}"
fi
export MODEL_PATH="${RESOLVED_MODEL_PATH}"

CHECKPOINT_TAG="$(basename "${MODEL_PATH}")"
export OUTPUT_DIR="${OUTPUT_DIR:-/home/jovyan/AURORA/outputs/${CHECKPOINT_TAG}_checkpoint_eval_metrics_fp32}"
export IMAGE_DIR="${IMAGE_DIR:-/home/jovyan/data/coco/val2017}"
export CAPTIONS_JSONL="${CAPTIONS_JSONL:-/home/jovyan/Scale-RAE/outputs/caption_to_image_recon_concise/captions.jsonl}"

export GPU0="${GPU0:-0}"
export GPU1="${GPU1:-1}"
export DTYPE="${DTYPE:-fp32}"
export PER_DEVICE_EVAL_BATCH_SIZE="${PER_DEVICE_EVAL_BATCH_SIZE:-8}"
export DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-4}"
export MAX_CAPTION_TOKENS="${MAX_CAPTION_TOKENS:-64}"
export MAX_SAMPLES="${MAX_SAMPLES:-}"
export GUIDANCE_LEVEL="${GUIDANCE_LEVEL:-1.0}"
export DIFFUSION_STEPS="${DIFFUSION_STEPS:-20}"
export TORCH_COMPILE="${TORCH_COMPILE:-1}"

# metrics-only
export SAVE_IMAGES="${SAVE_IMAGES:-0}"
export SAVE_LIMIT="${SAVE_LIMIT:-0}"
export SAVE_FIXED_FIRST_N="${SAVE_FIXED_FIRST_N:-0}"
export REPORT_LOSSES="${REPORT_LOSSES:-0}"
export EVAL_SLOT_ATTENTION="${EVAL_SLOT_ATTENTION:-0}"
export SAVE_ATTN_MAPS="${SAVE_ATTN_MAPS:-0}"
export SAVE_ATTN_LIMIT="${SAVE_ATTN_LIMIT:-0}"

echo "[run_eval_captionslot_checkpoint_metrics_parallel] model=${MODEL_PATH}"
echo "[run_eval_captionslot_checkpoint_metrics_parallel] output=${OUTPUT_DIR}"
echo "[run_eval_captionslot_checkpoint_metrics_parallel] gpus=${GPU0},${GPU1} dtype=${DTYPE} compile=${TORCH_COMPILE}"
echo "[run_eval_captionslot_checkpoint_metrics_parallel] metrics=rFID,PSNR,SSIM,MSE"

exec bash "${SCRIPT_DIR}/run_eval_captionslot_checkpoint_parallel.sh"
