#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-/home/jovyan/.conda/envs/scale_rae/bin/python}"
SCRIPT_PATH="/home/jovyan/AURORA/scripts/generate_stagea_object_captions.py"

GPU="${GPU:-0}"
NUM_SHARDS="${NUM_SHARDS:-1}"
SHARD_INDEX="${SHARD_INDEX:-0}"
IMAGE_DIR="${IMAGE_DIR:-/home/jovyan/data/coco/train2017}"
MODEL_PATH="${MODEL_PATH:-/home/jovyan/data/Scale-RAE-Qwen1.5B_DiT2.4B}"
OUTPUT_DIR="${OUTPUT_DIR:-/home/jovyan/AURORA/outputs/stagea_object_captions_train2017_shard${SHARD_INDEX}}"

DTYPE="${DTYPE:-bf16}"
CAPTION_BATCH_SIZE="${CAPTION_BATCH_SIZE:-64}"
CAPTION_MAX_NEW_TOKENS="${CAPTION_MAX_NEW_TOKENS:-128}"
MAX_CAPTION_TOKENS="${MAX_CAPTION_TOKENS:-192}"
MAX_SLOTS="${MAX_SLOTS:-15}"
PROMPT_PRESET="${PROMPT_PRESET:-object_refexp_detailed_en}"
PROMPT_FILE="${PROMPT_FILE:-}"
PROMPT_TEXT="${PROMPT_TEXT:-}"
LOADER_NUM_WORKERS="${LOADER_NUM_WORKERS:-8}"
FSYNC_EVERY_N_BATCHES="${FSYNC_EVERY_N_BATCHES:-20}"
SAVE_LIMIT="${SAVE_LIMIT:-100}"
SEED="${SEED:-42}"
TF32="${TF32:-1}"
REPAIR_EXISTING_OUTPUTS="${REPAIR_EXISTING_OUTPUTS:-1}"
MAX_SAMPLES="${MAX_SAMPLES:-}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

TF32_FLAG=()
if [[ "${TF32}" == "1" ]]; then
  TF32_FLAG=(--tf32)
fi

PROMPT_ARGS=(--prompt-preset "${PROMPT_PRESET}")
if [[ -n "${PROMPT_FILE}" ]]; then
  PROMPT_ARGS=(--prompt-file "${PROMPT_FILE}")
elif [[ -n "${PROMPT_TEXT}" ]]; then
  PROMPT_ARGS=(--prompt "${PROMPT_TEXT}")
fi

REPAIR_ARGS=()
if [[ "${REPAIR_EXISTING_OUTPUTS}" == "0" ]]; then
  REPAIR_ARGS=(--no-repair-existing-outputs)
fi

MAX_SAMPLE_ARGS=()
if [[ -n "${MAX_SAMPLES}" ]]; then
  MAX_SAMPLE_ARGS=(--max-samples "${MAX_SAMPLES}")
fi

CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON_BIN}" "${SCRIPT_PATH}" \
  --image-dir "${IMAGE_DIR}" \
  --output-dir "${OUTPUT_DIR}" \
  --model-path "${MODEL_PATH}" \
  --device cuda:0 \
  --dtype "${DTYPE}" \
  --caption-batch-size "${CAPTION_BATCH_SIZE}" \
  --caption-max-new-tokens "${CAPTION_MAX_NEW_TOKENS}" \
  --max-caption-tokens "${MAX_CAPTION_TOKENS}" \
  --max-slots "${MAX_SLOTS}" \
  "${PROMPT_ARGS[@]}" \
  --num-shards "${NUM_SHARDS}" \
  --shard-index "${SHARD_INDEX}" \
  --loader-num-workers "${LOADER_NUM_WORKERS}" \
  --fsync-every-n-batches "${FSYNC_EVERY_N_BATCHES}" \
  --save-limit "${SAVE_LIMIT}" \
  --seed "${SEED}" \
  "${MAX_SAMPLE_ARGS[@]}" \
  "${REPAIR_ARGS[@]}" \
  "${TF32_FLAG[@]}" \
  ${EXTRA_ARGS}
