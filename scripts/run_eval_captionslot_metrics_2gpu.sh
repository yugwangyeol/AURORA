#!/bin/bash
# Fast 2-GPU CaptionSlot checkpoint eval.
# Computes: rFID / PSNR / SSIM / MSE
# Defaults to fp32 weights+activations. TF32 GEMMs stay enabled by default for speed on B200.
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
    echo "[run_eval_captionslot_metrics_2gpu] no checkpoint-* found under ${MODEL_PATH_INPUT}" >&2
    exit 1
  fi
else
  RESOLVED_MODEL_PATH="${MODEL_PATH_INPUT}"
fi
export MODEL_PATH="${RESOLVED_MODEL_PATH}"

CHECKPOINT_TAG="$(basename "${MODEL_PATH}")"
export IMAGE_DIR="${IMAGE_DIR:-/home/jovyan/data/coco/val2017}"
export CAPTIONS_JSONL="${CAPTIONS_JSONL:-/home/jovyan/Scale-RAE/outputs/caption_to_image_recon_concise/captions.jsonl}"
export OUTPUT_DIR="${OUTPUT_DIR:-/home/jovyan/AURORA/outputs/${CHECKPOINT_TAG}_metrics_eval_fp32}"

export GPU0="${GPU0:-0}"
export GPU1="${GPU1:-1}"
export DTYPE="${DTYPE:-fp32}"
export ALLOW_TF32="${ALLOW_TF32:-1}"
export TORCH_COMPILE="${TORCH_COMPILE:-1}"
export BATCH_SIZE="${BATCH_SIZE:-16}"
export NUM_WORKERS="${NUM_WORKERS:-4}"
export DIFFUSION_STEPS="${DIFFUSION_STEPS:-20}"
export GUIDANCE_LEVEL="${GUIDANCE_LEVEL:-1.0}"
export MAX_CAPTION_TOKENS="${MAX_CAPTION_TOKENS:-64}"
export MAX_SAMPLES="${MAX_SAMPLES:-}"
export SAVE_IMAGES="${SAVE_IMAGES:-0}"
export SPACY_MODEL="${SPACY_MODEL:-en_core_web_sm}"

echo "[run_eval_captionslot_metrics_2gpu] model=${MODEL_PATH}"
echo "[run_eval_captionslot_metrics_2gpu] output=${OUTPUT_DIR}"
echo "[run_eval_captionslot_metrics_2gpu] gpus=${GPU0},${GPU1} dtype=${DTYPE} allow_tf32=${ALLOW_TF32} torch_compile=${TORCH_COMPILE}"
echo "[run_eval_captionslot_metrics_2gpu] metrics=rFID,PSNR,SSIM,MSE"

exec bash "${SCRIPT_DIR}/run_eval_fast.sh"
