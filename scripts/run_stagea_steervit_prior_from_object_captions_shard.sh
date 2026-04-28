#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-/home/jovyan/.conda/envs/scale_rae/bin/python}"
SCRIPT_PATH="/home/jovyan/AURORA/scripts/build_stagea_steervit_prior_from_object_captions.py"

GPU="${GPU:-0}"
NUM_SHARDS="${NUM_SHARDS:-1}"
SHARD_INDEX="${SHARD_INDEX:-0}"
CAPTION_INPUT_DIR="${CAPTION_INPUT_DIR:-/home/jovyan/AURORA/outputs/stagea_object_captions_train2017_postprocessed}"
OUTPUT_DIR="${OUTPUT_DIR:-/home/jovyan/AURORA/outputs/stagea_steervit_prior_from_object_captions_train2017_shard${SHARD_INDEX}}"

DTYPE="${DTYPE:-bf16}"
GRID_SIDE="${GRID_SIDE:-16}"
MAX_SLOTS="${MAX_SLOTS:-15}"
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
  --caption-input-dir "${CAPTION_INPUT_DIR}" \
  --output-dir "${OUTPUT_DIR}" \
  --device cuda:0 \
  --dtype "${DTYPE}" \
  --grid-side "${GRID_SIDE}" \
  --max-slots "${MAX_SLOTS}" \
  --steervit-batch-size "${STEERVIT_BATCH_SIZE}" \
  --map-shard-size "${MAP_SHARD_SIZE}" \
  --num-shards "${NUM_SHARDS}" \
  --shard-index "${SHARD_INDEX}" \
  --loader-num-workers "${LOADER_NUM_WORKERS}" \
  --fsync-every-n-batches "${FSYNC_EVERY_N_BATCHES}" \
  --seed "${SEED}" \
  "${TF32_FLAG[@]}" \
  ${EXTRA_ARGS}
