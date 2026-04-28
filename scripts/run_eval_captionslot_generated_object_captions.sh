#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

find_latest_checkpoint() {
  local run_dir="$1"
  find "${run_dir}" -maxdepth 1 -type d -name 'checkpoint-*' | sort -V | tail -n 1
}

resolve_caption_jsonl() {
  local caption_dir="$1"
  local summary_path="${caption_dir}/summary.json"
  local predictions_path="${caption_dir}/predictions.jsonl"

  if [[ -f "${summary_path}" ]]; then
    local from_summary
    from_summary="$(python - <<'PY' "${summary_path}"
import json, sys
path = sys.argv[1]
with open(path, "r", encoding="utf-8") as f:
    obj = json.load(f)
print(obj.get("predictions_jsonl", ""))
PY
)"
    if [[ -n "${from_summary}" && -f "${from_summary}" ]]; then
      printf '%s\n' "${from_summary}"
      return 0
    fi
  fi

  if [[ -f "${predictions_path}" ]]; then
    printf '%s\n' "${predictions_path}"
    return 0
  fi

  return 1
}

resolve_checkpoint_path() {
  local model_input="$1"
  local prefer_best="$2"
  local input_base
  input_base="$(basename "${model_input}")"

  if [[ "${prefer_best}" == "1" && "${input_base}" != "best-checkpoint" && ! "${input_base}" =~ ^checkpoint- && -f "${model_input}/best-checkpoint/config.json" ]]; then
    printf '%s\n' "${model_input}/best-checkpoint"
    return 0
  fi

  if [[ -f "${model_input}/config.json" ]]; then
    printf '%s\n' "${model_input}"
    return 0
  fi

  local latest
  latest="$(find_latest_checkpoint "${model_input}")"
  if [[ -n "${latest}" && -f "${latest}/config.json" ]]; then
    printf '%s\n' "${latest}"
    return 0
  fi

  return 1
}

EVAL_MODE="${EVAL_MODE:-metrics}"   # metrics | full
RUN_DIR="${RUN_DIR:-/home/jovyan/AURORA/checkpoints/aurora_refcoco_multislot24_pair2_reg64_xattn_zeroinit_stage1_fp32}"
MODEL_PATH_INPUT="${MODEL_PATH:-${RUN_DIR}}"
PREFER_BEST_CHECKPOINT="${PREFER_BEST_CHECKPOINT:-1}"

CAPTION_OUTPUT_DIR="${CAPTION_OUTPUT_DIR:-/home/jovyan/AURORA/outputs/stagea_object_captions_val2017_refexp}"
CAPTIONS_JSONL="${CAPTIONS_JSONL:-}"
IMAGE_DIR="${IMAGE_DIR:-/home/jovyan/data/coco/val2017}"

GPU0="${GPU0:-0}"
GPU1="${GPU1:-1}"
DTYPE="${DTYPE:-fp32}"
MAX_SAMPLES="${MAX_SAMPLES:-}"
MAX_CAPTION_TOKENS="${MAX_CAPTION_TOKENS:-192}"
GUIDANCE_LEVEL="${GUIDANCE_LEVEL:-1.0}"
DIFFUSION_STEPS="${DIFFUSION_STEPS:-20}"
TORCH_COMPILE="${TORCH_COMPILE:-1}"

PER_DEVICE_EVAL_BATCH_SIZE="${PER_DEVICE_EVAL_BATCH_SIZE:-8}"
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-4}"
SAVE_IMAGES="${SAVE_IMAGES:-0}"
SAVE_LIMIT="${SAVE_LIMIT:-0}"
SAVE_FIXED_FIRST_N="${SAVE_FIXED_FIRST_N:-0}"
REPORT_LOSSES="${REPORT_LOSSES:-0}"
EVAL_SLOT_ATTENTION="${EVAL_SLOT_ATTENTION:-0}"
SAVE_ATTN_MAPS="${SAVE_ATTN_MAPS:-0}"
SAVE_ATTN_LIMIT="${SAVE_ATTN_LIMIT:-0}"
ATTN_THRESHOLD="${ATTN_THRESHOLD:-0.5}"
ATTN_THRESHOLDS="${ATTN_THRESHOLDS:-}"
SEGMENTATION_BG_THRESHOLD="${SEGMENTATION_BG_THRESHOLD:-0.5}"
SEGMENTATION_BG_THRESHOLDS="${SEGMENTATION_BG_THRESHOLDS:-}"
DRY_RUN="${DRY_RUN:-0}"

if [[ -z "${CAPTIONS_JSONL}" ]]; then
  if ! CAPTIONS_JSONL="$(resolve_caption_jsonl "${CAPTION_OUTPUT_DIR}")"; then
    echo "[run_eval_captionslot_generated_object_captions] could not resolve predictions.jsonl under ${CAPTION_OUTPUT_DIR}" >&2
    exit 1
  fi
fi

if ! RESOLVED_MODEL_PATH="$(resolve_checkpoint_path "${MODEL_PATH_INPUT}" "${PREFER_BEST_CHECKPOINT}")"; then
  echo "[run_eval_captionslot_generated_object_captions] could not resolve checkpoint from ${MODEL_PATH_INPUT}" >&2
  exit 1
fi

MODEL_PATH="${RESOLVED_MODEL_PATH}"
CHECKPOINT_TAG="$(basename "${MODEL_PATH}")"
if [[ "${CHECKPOINT_TAG}" == "best-checkpoint" ]]; then
  CHECKPOINT_TAG="$(basename "$(dirname "${MODEL_PATH}")")_best-checkpoint"
fi
CAPTION_TAG="$(basename "${CAPTION_OUTPUT_DIR}")"
OUTPUT_DIR_DEFAULT="/home/jovyan/AURORA/outputs/${CHECKPOINT_TAG}_${CAPTION_TAG}_${EVAL_MODE}_eval"
OUTPUT_DIR="${OUTPUT_DIR:-${OUTPUT_DIR_DEFAULT}}"

echo "============================================================"
echo "CaptionSlot eval from generated object captions"
echo "EVAL_MODE:            ${EVAL_MODE}"
echo "MODEL_PATH:           ${MODEL_PATH}"
echo "CAPTION_OUTPUT_DIR:   ${CAPTION_OUTPUT_DIR}"
echo "CAPTIONS_JSONL:       ${CAPTIONS_JSONL}"
echo "IMAGE_DIR:            ${IMAGE_DIR}"
echo "OUTPUT_DIR:           ${OUTPUT_DIR}"
echo "GPU0/GPU1:            ${GPU0},${GPU1}"
echo "DTYPE:                ${DTYPE}"
echo "MAX_SAMPLES:          ${MAX_SAMPLES:-all}"
echo "MAX_CAPTION_TOKENS:   ${MAX_CAPTION_TOKENS}"
echo "GUIDANCE_LEVEL:       ${GUIDANCE_LEVEL}"
echo "DIFFUSION_STEPS:      ${DIFFUSION_STEPS}"
echo "TORCH_COMPILE:        ${TORCH_COMPILE}"
echo "SEG_BG_THRESHOLD:     ${SEGMENTATION_BG_THRESHOLD}"
echo "DRY_RUN:              ${DRY_RUN}"
echo "============================================================"

COMMON_ENV=(
  "MODEL_PATH=${MODEL_PATH}"
  "IMAGE_DIR=${IMAGE_DIR}"
  "CAPTIONS_JSONL=${CAPTIONS_JSONL}"
  "OUTPUT_DIR=${OUTPUT_DIR}"
  "GPU0=${GPU0}"
  "GPU1=${GPU1}"
  "DTYPE=${DTYPE}"
  "MAX_CAPTION_TOKENS=${MAX_CAPTION_TOKENS}"
  "GUIDANCE_LEVEL=${GUIDANCE_LEVEL}"
  "DIFFUSION_STEPS=${DIFFUSION_STEPS}"
  "TORCH_COMPILE=${TORCH_COMPILE}"
  "ATTN_THRESHOLD=${ATTN_THRESHOLD}"
  "ATTN_THRESHOLDS=${ATTN_THRESHOLDS}"
  "SEGMENTATION_BG_THRESHOLD=${SEGMENTATION_BG_THRESHOLD}"
  "SEGMENTATION_BG_THRESHOLDS=${SEGMENTATION_BG_THRESHOLDS}"
)

if [[ "${EVAL_MODE}" == "metrics" ]]; then
  if [[ "${DRY_RUN}" == "1" ]]; then
    echo "[dry-run] bash ${SCRIPT_DIR}/run_eval_captionslot_checkpoint_metrics_parallel.sh"
    exit 0
  fi
  env \
    "${COMMON_ENV[@]}" \
    "PER_DEVICE_EVAL_BATCH_SIZE=${PER_DEVICE_EVAL_BATCH_SIZE}" \
    "DATALOADER_NUM_WORKERS=${DATALOADER_NUM_WORKERS}" \
    "SAVE_IMAGES=0" \
    "SAVE_LIMIT=0" \
    "SAVE_FIXED_FIRST_N=0" \
    "REPORT_LOSSES=0" \
    "EVAL_SLOT_ATTENTION=0" \
    "SAVE_ATTN_MAPS=0" \
    "SAVE_ATTN_LIMIT=0" \
    ${MAX_SAMPLES:+MAX_SAMPLES="${MAX_SAMPLES}"} \
    bash "${SCRIPT_DIR}/run_eval_captionslot_checkpoint_metrics_parallel.sh"
elif [[ "${EVAL_MODE}" == "full" ]]; then
  if [[ "${DRY_RUN}" == "1" ]]; then
    echo "[dry-run] bash ${SCRIPT_DIR}/run_eval_captionslot_checkpoint_parallel.sh"
    exit 0
  fi
  env \
    "${COMMON_ENV[@]}" \
    "PER_DEVICE_EVAL_BATCH_SIZE=${PER_DEVICE_EVAL_BATCH_SIZE}" \
    "DATALOADER_NUM_WORKERS=${DATALOADER_NUM_WORKERS}" \
    "SAVE_IMAGES=${SAVE_IMAGES}" \
    "SAVE_LIMIT=${SAVE_LIMIT}" \
    "SAVE_FIXED_FIRST_N=${SAVE_FIXED_FIRST_N}" \
    "REPORT_LOSSES=${REPORT_LOSSES}" \
    "EVAL_SLOT_ATTENTION=${EVAL_SLOT_ATTENTION}" \
    "SAVE_ATTN_MAPS=${SAVE_ATTN_MAPS}" \
    "SAVE_ATTN_LIMIT=${SAVE_ATTN_LIMIT}" \
    "ATTN_THRESHOLD=${ATTN_THRESHOLD}" \
    ${MAX_SAMPLES:+MAX_SAMPLES="${MAX_SAMPLES}"} \
    bash "${SCRIPT_DIR}/run_eval_captionslot_checkpoint_parallel.sh"
else
  echo "[run_eval_captionslot_generated_object_captions] unsupported EVAL_MODE=${EVAL_MODE} (use metrics or full)" >&2
  exit 1
fi
