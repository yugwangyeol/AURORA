#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/jovyan/.conda/envs/scale_rae/bin/python}"

MODEL_PATH="${MODEL_PATH:-/home/jovyan/data/Scale-RAE-Qwen1.5B_DiT2.4B}"
IMAGE_DIR="${IMAGE_DIR:-/home/jovyan/data/coco/val2017}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/home/jovyan/AURORA/outputs}"
OUTPUT_NAME="${OUTPUT_NAME:-stagea_object_captions_val2017_refexp}"

SHARD_OUTPUT_PREFIX="${SHARD_OUTPUT_PREFIX:-${OUTPUT_ROOT}/${OUTPUT_NAME}_shard}"
MERGED_RAW_DIR="${MERGED_RAW_DIR:-${OUTPUT_ROOT}/${OUTPUT_NAME}_raw}"
OUTPUT_DIR="${OUTPUT_DIR:-${OUTPUT_ROOT}/${OUTPUT_NAME}}"
LOG_DIR="${LOG_DIR:-${OUTPUT_ROOT}/${OUTPUT_NAME}_logs}"

NUM_SHARDS="${NUM_SHARDS:-2}"
GPU0="${GPU0:-0}"
GPU1="${GPU1:-1}"

DTYPE="${DTYPE:-fp32}"
CAPTION_BATCH_SIZE="${CAPTION_BATCH_SIZE:-16}"
CAPTION_MAX_NEW_TOKENS="${CAPTION_MAX_NEW_TOKENS:-128}"
MAX_CAPTION_TOKENS="${MAX_CAPTION_TOKENS:-192}"
MAX_SLOTS="${MAX_SLOTS:-15}"
LOADER_NUM_WORKERS="${LOADER_NUM_WORKERS:-8}"
FSYNC_EVERY_N_BATCHES="${FSYNC_EVERY_N_BATCHES:-20}"
SAVE_LIMIT="${SAVE_LIMIT:-100}"
SEED="${SEED:-42}"
TF32="${TF32:-1}"
PROMPT_PRESET="${PROMPT_PRESET:-object_refexp_detailed_en}"
PROMPT_FILE="${PROMPT_FILE:-}"
PROMPT_TEXT="${PROMPT_TEXT:-}"
REPAIR_EXISTING_OUTPUTS="${REPAIR_EXISTING_OUTPUTS:-1}"

MAX_SAMPLES="${MAX_SAMPLES:-}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

mkdir -p "${OUTPUT_ROOT}" "${LOG_DIR}"

run_shard() {
  local gpu="$1"
  local shard_index="$2"
  local shard_dir="$3"
  local log_file="$4"

  GPU="${gpu}" \
  NUM_SHARDS="${NUM_SHARDS}" \
  SHARD_INDEX="${shard_index}" \
  IMAGE_DIR="${IMAGE_DIR}" \
  MODEL_PATH="${MODEL_PATH}" \
  OUTPUT_DIR="${shard_dir}" \
  DTYPE="${DTYPE}" \
  CAPTION_BATCH_SIZE="${CAPTION_BATCH_SIZE}" \
  CAPTION_MAX_NEW_TOKENS="${CAPTION_MAX_NEW_TOKENS}" \
  MAX_CAPTION_TOKENS="${MAX_CAPTION_TOKENS}" \
  MAX_SLOTS="${MAX_SLOTS}" \
  PROMPT_PRESET="${PROMPT_PRESET}" \
  PROMPT_FILE="${PROMPT_FILE}" \
  PROMPT_TEXT="${PROMPT_TEXT}" \
  LOADER_NUM_WORKERS="${LOADER_NUM_WORKERS}" \
  FSYNC_EVERY_N_BATCHES="${FSYNC_EVERY_N_BATCHES}" \
  SAVE_LIMIT="${SAVE_LIMIT}" \
  SEED="${SEED}" \
  TF32="${TF32}" \
  REPAIR_EXISTING_OUTPUTS="${REPAIR_EXISTING_OUTPUTS}" \
  EXTRA_ARGS="${EXTRA_ARGS}" \
  MAX_SAMPLES="${MAX_SAMPLES}" \
  bash "${SCRIPT_DIR}/run_stagea_object_caption_shard.sh" \
    > "${log_file}" 2>&1
}

echo "============================================================"
echo "Stage A object caption generation (val2017)"
echo "MODEL_PATH:           ${MODEL_PATH}"
echo "IMAGE_DIR:            ${IMAGE_DIR}"
echo "PROMPT_PRESET:        ${PROMPT_PRESET}"
echo "OUTPUT_DIR:           ${OUTPUT_DIR}"
echo "MERGED_RAW_DIR:       ${MERGED_RAW_DIR}"
echo "SHARD_OUTPUT_PREFIX:  ${SHARD_OUTPUT_PREFIX}"
echo "NUM_SHARDS:           ${NUM_SHARDS}"
echo "DTYPE:                ${DTYPE}"
echo "CAPTION_BATCH_SIZE:   ${CAPTION_BATCH_SIZE}"
echo "CAPTION_MAX_NEW:      ${CAPTION_MAX_NEW_TOKENS}"
echo "MAX_CAPTION_TOKENS:   ${MAX_CAPTION_TOKENS}"
echo "MAX_SLOTS:            ${MAX_SLOTS}"
echo "MAX_SAMPLES:          ${MAX_SAMPLES:-all}"
echo "LOG_DIR:              ${LOG_DIR}"
echo "============================================================"

SHARD_DIRS=()
if [[ "${NUM_SHARDS}" == "1" ]]; then
  SHARD_DIR="${SHARD_OUTPUT_PREFIX}0_of_1"
  SHARD_DIRS+=("${SHARD_DIR}")
  run_shard "${GPU0}" 0 "${SHARD_DIR}" "${LOG_DIR}/shard_0_of_1.log"
elif [[ "${NUM_SHARDS}" == "2" ]]; then
  SHARD0_DIR="${SHARD_OUTPUT_PREFIX}0_of_2"
  SHARD1_DIR="${SHARD_OUTPUT_PREFIX}1_of_2"
  SHARD_DIRS+=("${SHARD0_DIR}" "${SHARD1_DIR}")

  run_shard "${GPU0}" 0 "${SHARD0_DIR}" "${LOG_DIR}/shard_0_of_2.log" &
  PID0=$!
  run_shard "${GPU1}" 1 "${SHARD1_DIR}" "${LOG_DIR}/shard_1_of_2.log" &
  PID1=$!

  FAIL=0
  wait "${PID0}" || FAIL=1
  wait "${PID1}" || FAIL=1
  if [[ "${FAIL}" -ne 0 ]]; then
    echo "[run_stagea_object_caption_val2017_parallel] shard failure; inspect ${LOG_DIR}" >&2
    exit 1
  fi
else
  echo "Unsupported NUM_SHARDS=${NUM_SHARDS}. Use 1 or 2." >&2
  exit 1
fi

PYTHONNOUSERSITE="${PYTHONNOUSERSITE:-1}" \
"${PYTHON_BIN}" "${SCRIPT_DIR}/merge_stagea_object_caption_shards.py" \
  --input-dirs "${SHARD_DIRS[@]}" \
  --output-dir "${MERGED_RAW_DIR}" \
  --overwrite

PYTHONNOUSERSITE="${PYTHONNOUSERSITE:-1}" \
"${PYTHON_BIN}" "${SCRIPT_DIR}/postprocess_stagea_object_captions.py" \
  --input-dir "${MERGED_RAW_DIR}" \
  --output-dir "${OUTPUT_DIR}" \
  --model-path "${MODEL_PATH}" \
  --max-slots "${MAX_SLOTS}" \
  --max-caption-tokens "${MAX_CAPTION_TOKENS}" \
  --save-limit "${SAVE_LIMIT}" \
  --overwrite

echo
echo "Finished val2017 object-caption build"
echo "Raw merged summary:   ${MERGED_RAW_DIR}/summary.json"
echo "Cleaned summary:      ${OUTPUT_DIR}/summary.json"
