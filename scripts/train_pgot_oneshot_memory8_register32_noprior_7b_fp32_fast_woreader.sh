#!/usr/bin/env bash
# 7B FP32 fast training with Reader supervision removed (Reader itself remains).
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
IFS=, read -r -a DEVICES <<< "${CUDA_VISIBLE_DEVICES}"
if (( ${#DEVICES[@]} != 4 )); then
    echo 'Set CUDA_VISIBLE_DEVICES to exactly four GPUs.' >&2; exit 2
fi
if [[ "${DEVICES[0]}" == "${DEVICES[1]}" || "${DEVICES[0]}" == "${DEVICES[2]}" || \
      "${DEVICES[0]}" == "${DEVICES[3]}" || "${DEVICES[1]}" == "${DEVICES[2]}" || \
      "${DEVICES[1]}" == "${DEVICES[3]}" || "${DEVICES[2]}" == "${DEVICES[3]}" ]]; then
    echo 'CUDA_VISIBLE_DEVICES must contain four distinct GPUs.' >&2; exit 2
fi
export NUM_GPUS=4 PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-4}"
export GLOBAL_BATCH_SIZE=32 GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-2}"
export FSDP_CONFIG="${FSDP_CONFIG:-${PROJECT_ROOT}/configs/fsdp_7b_full_shard_post.json}"
export TRAIN_ENTRYPOINT="${TRAIN_ENTRYPOINT:-${PROJECT_ROOT}/train.py}" PGOT_SKIP_FINAL_SAVE=0
export OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/checkpoints/pgot_oneshot_memory8_register32_noprior_7b_fp32_fast_woreader}"
export WANDB_NAME="${WANDB_NAME:-pgot_oneshot_memory8_register32_noprior_7b_fp32_fast_woreader}"
export MASTER_PORT="${MASTER_PORT:-29676}"

# This is the only scientific change from the successful 7B FP32 fast baseline.
export E8_READER_SUPERVISION_MODE=none
export E8_READER_OBJECT_WEIGHT=0.0
export E8_READER_BACKGROUND_WEIGHT=0.0

if [[ -e "${OUTPUT_DIR}/config.json" || \
      -n "$(find "${OUTPUT_DIR}" -maxdepth 1 -name 'checkpoint-*' -print -quit 2>/dev/null || true)" ]]; then
    echo "Training output already contains a model: ${OUTPUT_DIR}" >&2; exit 2
fi
exec bash "${PROJECT_ROOT}/scripts/train_pgot_oneshot_memory8_register32_noprior_7b_fp32.sh"
