#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-/home/jovyan/.conda/envs/scale_rae/bin/python}"
SCRIPT_PATH="/home/jovyan/AURORA/scripts/build_stagea_object_steervit_prior_cache.py"

GPU="${GPU:-0}"
NUM_SHARDS="${NUM_SHARDS:-1}"
SHARD_INDEX="${SHARD_INDEX:-0}"
IMAGE_DIR="${IMAGE_DIR:-/home/jovyan/data/coco/train2017}"
MODEL_PATH="${MODEL_PATH:-/home/jovyan/data/Scale-RAE-Qwen1.5B_DiT2.4B}"
OUTPUT_DIR="${OUTPUT_DIR:-/home/jovyan/AURORA/outputs/stagea_object_steervit_prior_cache_train2017_shard${SHARD_INDEX}}"

DTYPE="${DTYPE:-fp32}"
CAPTION_BATCH_SIZE="${CAPTION_BATCH_SIZE:-64}"
CAPTION_MAX_NEW_TOKENS="${CAPTION_MAX_NEW_TOKENS:-192}"
MAX_CAPTION_TOKENS="${MAX_CAPTION_TOKENS:-256}"
MAX_SLOTS="${MAX_SLOTS:-15}"
GRID_SIDE="${GRID_SIDE:-16}"
STEERVIT_BATCH_SIZE="${STEERVIT_BATCH_SIZE:-128}"
MAP_SHARD_SIZE="${MAP_SHARD_SIZE:-1000}"
LOADER_NUM_WORKERS="${LOADER_NUM_WORKERS:-8}"
FSYNC_EVERY_N_BATCHES="${FSYNC_EVERY_N_BATCHES:-20}"
SEED="${SEED:-42}"
TF32="${TF32:-1}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

TF32_FLAG=()
if [[ "${TF32}" == "1" ]]; then
  TF32_FLAG=(--tf32)
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
  --grid-side "${GRID_SIDE}" \
  --steervit-batch-size "${STEERVIT_BATCH_SIZE}" \
  --map-shard-size "${MAP_SHARD_SIZE}" \
  --num-shards "${NUM_SHARDS}" \
  --shard-index "${SHARD_INDEX}" \
  --seed "${SEED}" \
  --loader-num-workers "${LOADER_NUM_WORKERS}" \
  --fsync-every-n-batches "${FSYNC_EVERY_N_BATCHES}" \
  "${TF32_FLAG[@]}" \
  ${EXTRA_ARGS}
