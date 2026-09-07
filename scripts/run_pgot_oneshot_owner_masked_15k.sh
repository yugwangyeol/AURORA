#!/usr/bin/env bash
# Continue the one-shot owner-masked run for 10k more steps, then evaluate.
#
#   1) train  : init from pgot_oneshot_owner_masked/checkpoint-5000, 10k steps
#   2) eval   : teacher-forced CODA-512 (full 4,720 images, rFID + segmentation)
#   3) ar eval: autoregressive captions on the same full eval set
#
# The 5k checkpoint was written with SAVE_ONLY_MODEL=True, so it carries no
# optimizer/scheduler state.  This continues by re-initializing from those
# weights with a fresh optimizer rather than a true Trainer resume.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
INIT_CKPT="${INIT_CKPT:-${PROJECT_ROOT}/checkpoints/pgot_oneshot_owner_masked/checkpoint-5000}"
RUN_NAME="${RUN_NAME:-pgot_oneshot_owner_masked_15k}"
TRAIN_DIR="${PROJECT_ROOT}/checkpoints/${RUN_NAME}"
MAX_STEPS="${MAX_STEPS:-10000}"
FINAL_CKPT="${TRAIN_DIR}/checkpoint-${MAX_STEPS}"

test -d "${INIT_CKPT}" || { echo "Missing init checkpoint: ${INIT_CKPT}" >&2; exit 1; }
mkdir -p "${PROJECT_ROOT}/outputs"

echo "===== [1/3] train ${RUN_NAME}: ${MAX_STEPS} steps from $(basename "${INIT_CKPT}") ====="
MODEL_PATH="${INIT_CKPT}" \
OUTPUT_DIR="${TRAIN_DIR}" \
WANDB_NAME="${RUN_NAME}" \
MAX_STEPS="${MAX_STEPS}" \
PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-6}" \
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-24}" \
    bash "${PROJECT_ROOT}/scripts/train_pgot_oneshot_owner_masked.sh" \
    2>&1 | tee "${PROJECT_ROOT}/outputs/train_${RUN_NAME}.log"

test -d "${FINAL_CKPT}" || { echo "Training finished but ${FINAL_CKPT} is missing" >&2; exit 1; }

echo "===== [2/3] teacher-forced eval (full 4,720) ====="
MODEL_PATH="${FINAL_CKPT}" \
OUTPUT_DIR="${PROJECT_ROOT}/outputs/eval_${RUN_NAME}" \
    bash "${PROJECT_ROOT}/scripts/eval_pgot_oneshot_owner_masked.sh" \
    2>&1 | tee "${PROJECT_ROOT}/outputs/eval_${RUN_NAME}.log"

# Empty AR_MAX_SAMPLES means the full evaluation set, matching step 2.
# Generation is batched (fixed-length prompts, lockstep decode), so a larger
# batch is a throughput win over the batch-1 pilot setting.
AR_MAX_SAMPLES="${AR_MAX_SAMPLES:-}"
AR_TAG="${AR_MAX_SAMPLES:-full}"
AR_OUTPUT_DIR="${PROJECT_ROOT}/outputs/eval_${RUN_NAME}_ar_${AR_TAG}"

echo "===== [3/3] autoregressive-caption eval (${AR_TAG} eval set) ====="
MODEL_PATH="${FINAL_CKPT}" \
OUTPUT_DIR="${AR_OUTPUT_DIR}" \
MAX_SAMPLES="${AR_MAX_SAMPLES}" \
BATCH_SIZE="${AR_BATCH_SIZE:-4}" \
NUM_WORKERS="${AR_NUM_WORKERS:-4}" \
    bash "${PROJECT_ROOT}/scripts/eval_pgot_oneshot_owner_masked.sh" \
    --caption_mode autoregressive \
    --ar_max_new_tokens "${AR_MAX_NEW_TOKENS:-512}" \
    2>&1 | tee "${PROJECT_ROOT}/outputs/eval_${RUN_NAME}_ar.log"

echo "===== DONE: ${RUN_NAME} ====="
echo "  train ckpt : ${FINAL_CKPT}"
echo "  eval       : ${PROJECT_ROOT}/outputs/eval_${RUN_NAME}/summary.json"
echo "  ar eval    : ${AR_OUTPUT_DIR}/summary.json"
