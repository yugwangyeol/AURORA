#!/usr/bin/env bash
# Continue the one-shot pooled run for 10k more steps, then evaluate.
#
#   1) train  : init from pgot_oneshot_pooled/checkpoint-5000, 10k steps
#   2) eval   : teacher-forced CODA-512 (full 4,720 images, rFID + segmentation)
#   3) ar eval: autoregressive captions on the same full eval set
#
# Exact mirror of run_pgot_oneshot_owner_masked_15k.sh.  The two runs must stay
# step-for-step identical so that the pooled/owner_masked comparison isolates
# the readout mode and never the training budget.
#
# The 5k checkpoint was written with SAVE_ONLY_MODEL=True, so it carries no
# optimizer/scheduler state.  This continues by re-initializing from those
# weights with a fresh optimizer rather than a true Trainer resume.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
INIT_CKPT="${INIT_CKPT:-${PROJECT_ROOT}/checkpoints/pgot_oneshot_pooled/checkpoint-5000}"
RUN_NAME="${RUN_NAME:-pgot_oneshot_pooled_15k}"
TRAIN_DIR="${PROJECT_ROOT}/checkpoints/${RUN_NAME}"
MAX_STEPS="${MAX_STEPS:-10000}"
FINAL_CKPT="${TRAIN_DIR}/checkpoint-${MAX_STEPS}"

test -d "${INIT_CKPT}" || { echo "Missing init checkpoint: ${INIT_CKPT}" >&2; exit 1; }
mkdir -p "${PROJECT_ROOT}/outputs"

# --- MIG sizing -------------------------------------------------------------
# owner_masked_15k ran on two whole B200s (183GB) at per-device 6.  These GPUs
# are now MIG 4g.90gb slices (89GiB).  The ~35GB of fixed cost (fp32 weights +
# grads + Adam, gradient_checkpointing is False) does NOT shrink with the batch,
# so the activation budget drops from ~145GB to ~54GB -- roughly a third, not a
# half.  Hence per-device 6 -> 2.
#
# GLOBAL_BATCH_SIZE MUST STAY 24.  max_steps counts optimizer steps, so 10k
# steps x 24 samples is the same 240k samples owner_masked_15k saw; only the
# gradient-accumulation factor changes (2 -> 6).  Changing it would break the
# very comparison this run exists to make.
PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-2}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-24}"
NUM_GPUS="${NUM_GPUS:-2}"

# MIG instances cannot do peer-to-peer / CUDA IPC.  NCCL usually falls back on
# its own but is known to hang on MIG; disabling P2P up front costs ~40min of
# host-bounced allreduce over the whole run and removes the failure mode.
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"

if (( GLOBAL_BATCH_SIZE != 24 )); then
    echo "WARNING: GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE} != 24 -- this run is no" >&2
    echo "         longer step-for-step comparable to owner_masked_15k." >&2
fi
echo "----- batch config -----"
echo "  per-device ${PER_DEVICE_TRAIN_BATCH_SIZE} x ${NUM_GPUS} gpu -> micro $((NUM_GPUS * PER_DEVICE_TRAIN_BATCH_SIZE))"
echo "  global ${GLOBAL_BATCH_SIZE}, grad-accum $((GLOBAL_BATCH_SIZE / (NUM_GPUS * PER_DEVICE_TRAIN_BATCH_SIZE)))"
echo "  NCCL_P2P_DISABLE=${NCCL_P2P_DISABLE}"
echo "------------------------"

echo "===== [1/3] train ${RUN_NAME}: ${MAX_STEPS} steps from $(basename "${INIT_CKPT}") ====="
# Pin the readout explicitly.  train_pgot_oneshot_pooled.sh only defaults to
# "pooled" (${ONE_SHOT_READOUT_MODE:-pooled}), so a stale owner_masked value
# exported in the caller's shell would silently train the wrong model.
MODEL_PATH="${INIT_CKPT}" \
OUTPUT_DIR="${TRAIN_DIR}" \
WANDB_NAME="${RUN_NAME}" \
MAX_STEPS="${MAX_STEPS}" \
ONE_SHOT_READOUT_MODE=pooled \
NUM_GPUS="${NUM_GPUS}" \
PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE}" \
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE}" \
    bash "${PROJECT_ROOT}/scripts/train_pgot_oneshot_pooled.sh" \
    2>&1 | tee "${PROJECT_ROOT}/outputs/train_${RUN_NAME}.log"

test -d "${FINAL_CKPT}" || { echo "Training finished but ${FINAL_CKPT} is missing" >&2; exit 1; }

echo "===== [2/3] teacher-forced eval (full 4,720) ====="
MODEL_PATH="${FINAL_CKPT}" \
OUTPUT_DIR="${PROJECT_ROOT}/outputs/eval_${RUN_NAME}" \
    bash "${PROJECT_ROOT}/scripts/eval_pgot_oneshot_pooled.sh" \
    2>&1 | tee "${PROJECT_ROOT}/outputs/eval_${RUN_NAME}.log"

# RUN_AR=0 stops after the teacher-forced eval, which is the number the
# pooled/owner_masked ablation actually needs.
if [ "${RUN_AR:-1}" = "0" ]; then
    echo "===== DONE (RUN_AR=0, autoregressive eval skipped): ${RUN_NAME} ====="
    echo "  train ckpt : ${FINAL_CKPT}"
    echo "  eval       : ${PROJECT_ROOT}/outputs/eval_${RUN_NAME}/summary.json"
    exit 0
fi

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
    bash "${PROJECT_ROOT}/scripts/eval_pgot_oneshot_pooled.sh" \
    --caption_mode autoregressive \
    --ar_max_new_tokens "${AR_MAX_NEW_TOKENS:-512}" \
    2>&1 | tee "${PROJECT_ROOT}/outputs/eval_${RUN_NAME}_ar.log"

echo "===== DONE: ${RUN_NAME} ====="
echo "  train ckpt : ${FINAL_CKPT}"
echo "  eval       : ${PROJECT_ROOT}/outputs/eval_${RUN_NAME}/summary.json"
echo "  ar eval    : ${AR_OUTPUT_DIR}/summary.json"
