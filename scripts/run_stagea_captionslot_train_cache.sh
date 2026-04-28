#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${GPU:-0}"

IMAGE_DIR="${IMAGE_DIR:-/home/jovyan/data/coco/train2017}"
OUTPUT_DIR="${OUTPUT_DIR:-/home/jovyan/AURORA/outputs/stagea_captionslot_train_cache_train2017}"
MODEL_PATH="${MODEL_PATH:-nyu-visionx/Scale-RAE-Qwen1.5B_DiT2.4B}"
PYTHON_BIN="${PYTHON_BIN:-/home/jovyan/.conda/envs/scale_rae/bin/python}"
DEVICE="${DEVICE:-cuda}"
DTYPE="${DTYPE:-bf16}"
CAPTION_MAX_NEW_TOKENS="${CAPTION_MAX_NEW_TOKENS:-32}"
MAX_CAPTION_TOKENS="${MAX_CAPTION_TOKENS:-64}"
TRACE_LAST_N_LAYERS="${TRACE_LAST_N_LAYERS:-4}"
CAPTION_BATCH_SIZE="${CAPTION_BATCH_SIZE:-8}"
TRACE_BATCH_SIZE="${TRACE_BATCH_SIZE:-4}"
SPACY_BATCH_SIZE="${SPACY_BATCH_SIZE:-128}"
MAX_SLOTS="${MAX_SLOTS:-10}"
MAP_SHARD_SIZE="${MAP_SHARD_SIZE:-1000}"
SPACY_MODEL="${SPACY_MODEL:-en_core_web_sm}"
SEED="${SEED:-42}"
NUM_SHARDS="${NUM_SHARDS:-1}"
SHARD_INDEX="${SHARD_INDEX:-0}"
SAVE_DEBUG_IMAGES="${SAVE_DEBUG_IMAGES:-0}"
DEBUG_LIMIT="${DEBUG_LIMIT:-25}"
FORCE_EAGER_ATTENTION="${FORCE_EAGER_ATTENTION:-1}"
USE_KV_CACHE="${USE_KV_CACHE:-1}"

EXTRA_ARGS=()
if [[ -n "${MAX_SAMPLES:-}" ]]; then
  EXTRA_ARGS+=(--max-samples "${MAX_SAMPLES}")
fi
if [[ "${SAVE_DEBUG_IMAGES}" == "1" ]]; then
  EXTRA_ARGS+=(--save-debug-images --debug-limit "${DEBUG_LIMIT}")
fi
if [[ "${FORCE_EAGER_ATTENTION}" == "1" ]]; then
  EXTRA_ARGS+=(--force-eager-attention)
fi
if [[ "${USE_KV_CACHE}" != "1" ]]; then
  EXTRA_ARGS+=(--disable-kv-cache)
fi

"${PYTHON_BIN}" /home/jovyan/AURORA/scripts/build_stagea_captionslot_train_cache.py \
  --image-dir "${IMAGE_DIR}" \
  --output-dir "${OUTPUT_DIR}" \
  --model-path "${MODEL_PATH}" \
  --caption-prompt "Describe this image in one concise sentence." \
  --device "${DEVICE}" \
  --dtype "${DTYPE}" \
  --caption-max-new-tokens "${CAPTION_MAX_NEW_TOKENS}" \
  --max-caption-tokens "${MAX_CAPTION_TOKENS}" \
  --trace-last-n-layers "${TRACE_LAST_N_LAYERS}" \
  --caption-batch-size "${CAPTION_BATCH_SIZE}" \
  --trace-batch-size "${TRACE_BATCH_SIZE}" \
  --spacy-batch-size "${SPACY_BATCH_SIZE}" \
  --max-slots "${MAX_SLOTS}" \
  --map-shard-size "${MAP_SHARD_SIZE}" \
  --spacy-model "${SPACY_MODEL}" \
  --seed "${SEED}" \
  --num-shards "${NUM_SHARDS}" \
  --shard-index "${SHARD_INDEX}" \
  "${EXTRA_ARGS[@]}"
