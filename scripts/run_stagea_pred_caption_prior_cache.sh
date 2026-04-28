#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${GPU:-0}"

IMAGE_DIR="${IMAGE_DIR:-/home/jovyan/data/coco/train2017}"
OUTPUT_DIR="${OUTPUT_DIR:-/home/jovyan/AURORA/outputs/stagea_pred_caption_prior_cache_train2017}"
MODEL_PATH="${MODEL_PATH:-nyu-visionx/Scale-RAE-Qwen1.5B_DiT2.4B}"
PYTHON_BIN="${PYTHON_BIN:-/home/jovyan/.conda/envs/scale_rae/bin/python}"
DEVICE="${DEVICE:-cuda}"
DTYPE="${DTYPE:-bf16}"
CAPTION_MAX_NEW_TOKENS="${CAPTION_MAX_NEW_TOKENS:-64}"
MAX_CAPTION_TOKENS="${MAX_CAPTION_TOKENS:-64}"
ATTENTION_TEMPERATURE="${ATTENTION_TEMPERATURE:-1.0}"
NORMALIZE_ATTENTION_TOKENS="${NORMALIZE_ATTENTION_TOKENS:-1}"
MAP_SHARD_SIZE="${MAP_SHARD_SIZE:-1000}"
SPACY_MODEL="${SPACY_MODEL:-en_core_web_sm}"
SEED="${SEED:-42}"
SAVE_DEBUG_IMAGES="${SAVE_DEBUG_IMAGES:-0}"
DEBUG_LIMIT="${DEBUG_LIMIT:-50}"
NUM_SHARDS="${NUM_SHARDS:-1}"
SHARD_INDEX="${SHARD_INDEX:-0}"

EXTRA_ARGS=()
if [[ -n "${MAX_SAMPLES:-}" ]]; then
  EXTRA_ARGS+=(--max-samples "${MAX_SAMPLES}")
fi
if [[ "${NORMALIZE_ATTENTION_TOKENS}" == "1" ]]; then
  EXTRA_ARGS+=(--normalize-attention-tokens)
fi
if [[ "${SAVE_DEBUG_IMAGES}" == "1" ]]; then
  EXTRA_ARGS+=(--save-debug-images --debug-limit "${DEBUG_LIMIT}")
fi

"${PYTHON_BIN}" /home/jovyan/AURORA/scripts/build_stagea_pred_caption_prior_cache.py \
  --image-dir "${IMAGE_DIR}" \
  --output-dir "${OUTPUT_DIR}" \
  --model-path "${MODEL_PATH}" \
  --caption-prompt "Describe this image in one concise sentence." \
  --device "${DEVICE}" \
  --dtype "${DTYPE}" \
  --caption-max-new-tokens "${CAPTION_MAX_NEW_TOKENS}" \
  --max-caption-tokens "${MAX_CAPTION_TOKENS}" \
  --attention-temperature "${ATTENTION_TEMPERATURE}" \
  --map-shard-size "${MAP_SHARD_SIZE}" \
  --spacy-model "${SPACY_MODEL}" \
  --num-shards "${NUM_SHARDS}" \
  --shard-index "${SHARD_INDEX}" \
  --seed "${SEED}" \
  "${EXTRA_ARGS[@]}"
