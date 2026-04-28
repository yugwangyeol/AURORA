#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export GPU0="${GPU0:-0}"
export GPU1="${GPU1:-1}"
export OUTPUT_DIR="${OUTPUT_DIR:-/home/jovyan/AURORA/outputs/captionslot_eval_checkpoint}"
export NUM_SHARDS=2

LOG_DIR="${OUTPUT_DIR}/logs"
SHARD0_DIR="${OUTPUT_DIR}/shard_0_of_2"
SHARD1_DIR="${OUTPUT_DIR}/shard_1_of_2"
mkdir -p "${LOG_DIR}"

CUDA_VISIBLE_DEVICES="${GPU0}" \
OUTPUT_DIR="${SHARD0_DIR}" \
NUM_SHARDS=2 SHARD_INDEX=0 \
bash "${SCRIPT_DIR}/run_eval_captionslot_checkpoint.sh" \
  > "${LOG_DIR}/shard_0_of_2.log" 2>&1 &
PID0=$!

CUDA_VISIBLE_DEVICES="${GPU1}" \
OUTPUT_DIR="${SHARD1_DIR}" \
NUM_SHARDS=2 SHARD_INDEX=1 \
bash "${SCRIPT_DIR}/run_eval_captionslot_checkpoint.sh" \
  > "${LOG_DIR}/shard_1_of_2.log" 2>&1 &
PID1=$!

wait "${PID0}"
wait "${PID1}"

PYTHONNOUSERSITE="${PYTHONNOUSERSITE:-1}" \
/home/jovyan/.conda/envs/scale_rae/bin/python \
  "${SCRIPT_DIR}/merge_captionslot_eval_shards.py" \
  --root-output-dir "${OUTPUT_DIR}" \
  --shard-dir "${SHARD0_DIR}" \
  --shard-dir "${SHARD1_DIR}"
