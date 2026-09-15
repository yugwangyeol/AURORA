#!/usr/bin/env bash
# Two-step train/save/reload smoke for CODA object-memory mixing.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-/home/jovyan/.conda/envs/scale_rae/bin/python}"
SMOKE_ROOT="$(mktemp -d "${PROJECT_ROOT}/.smoke_oneshot_memory8_register32_directrae_adapter_contrastive.XXXXXX")"
BASE_MODEL="${MODEL_PATH:-${PROJECT_ROOT}/checkpoints/pgot_e11_capacity_dit16/checkpoint-10000}"
RUN_ROOT="${SMOKE_ROOT}/content"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export PYTHONNOUSERSITE=1
export WANDB_MODE="${WANDB_MODE:-offline}"
export WANDB_PROJECT="${WANDB_PROJECT:-PGOT}"
export WANDB_DATA_DIR="${SMOKE_ROOT}/wandb_data"
export WANDB_CACHE_DIR="${SMOKE_ROOT}/wandb_cache"
export WANDB_RESUME=never
echo "Smoke workspace: ${SMOKE_ROOT}"

on_exit() {
    local status=$?
    if (( status )); then
        echo "Smoke failed; preserved for diagnosis at ${SMOKE_ROOT}" >&2
    fi
}
trap on_exit EXIT

mkdir -p "${RUN_ROOT}/wandb"
printf '%s\n' "${BASE_MODEL}" > "${RUN_ROOT}/source.txt"
run_id="$("${PYTHON}" -c 'import wandb; print(wandb.util.generate_id())')"
printf '%s\n' "${run_id}" > "${RUN_ROOT}/wandb_id.txt"

WANDB_RUN_ID="${run_id}" WANDB_NAME=smoke_memory8_register32_directrae_adapter_contrastive \
WANDB_DIR="${RUN_ROOT}/wandb" MODEL_PATH="${BASE_MODEL}" \
OUTPUT_DIR="${RUN_ROOT}/train" NUM_GPUS=2 \
PER_DEVICE_TRAIN_BATCH_SIZE="${SMOKE_BATCH_SIZE:-2}" \
GLOBAL_BATCH_SIZE="${SMOKE_GLOBAL_BATCH_SIZE:-24}" \
PGOT_CONTRASTIVE_WARMUP_STEPS=0 \
MAX_STEPS=2 SAVE_STEPS=2 SAVE_TOTAL_LIMIT=1 EVAL_STEPS=2 LOGGING_STEPS=1 \
PER_DEVICE_EVAL_BATCH_SIZE=1 EVAL_NUM_IMAGES=2 PGOT_EVAL_LOG_RECON_IMAGES=1 \
DATALOADER_NUM_WORKERS=1 REPORT_TO=wandb PGOT_SKIP_FINAL_SAVE=1 \
MASTER_PORT="${MASTER_PORT:-29683}" \
    bash "${PROJECT_ROOT}/scripts/train_pgot_oneshot_memory8_register32_noprior_directrae_adapter_contrastive.sh" \
    > "${RUN_ROOT}/train.log" 2>&1

"${PYTHON}" "${PROJECT_ROOT}/scripts/verify_oneshot_memory4_smoke.py" \
    "${SMOKE_ROOT}" content --stage train --owner-prior disabled \
    --direct-rae-query enabled --rae-query-adapter enabled \
    --contrastive enabled --contrastive-lambda 0.03 \
    --contrastive-sampling-rate 0.5 --contrastive-warmup 0 \
    --adapter-bottleneck 384 --object-memories 8 --register-memories 32

IFS=, read -r -a DEVICES <<< "${CUDA_VISIBLE_DEVICES}"
EVAL_DEVICE="${DEVICES[0]}"
CUDA_VISIBLE_DEVICES="${EVAL_DEVICE}" MODEL_PATH="${RUN_ROOT}/train/checkpoint-2" \
OUTPUT_DIR="${RUN_ROOT}/eval/tf" MAX_SAMPLES=2 BATCH_SIZE=1 NUM_WORKERS=1 \
DTYPE=fp32 DIFFUSION_INFERENCE_STEPS=2 COMPUTE_RFID=True COMPUTE_KID=True \
KID_SUBSETS=2 KID_SUBSET_SIZE=2 \
    bash "${PROJECT_ROOT}/scripts/eval_pgot_oneshot_memory8_register32_noprior_directrae_adapter_contrastive_tf.sh"

CUDA_VISIBLE_DEVICES="${EVAL_DEVICE}" MODEL_PATH="${RUN_ROOT}/train/checkpoint-2" \
OUTPUT_DIR="${RUN_ROOT}/eval/ar" MAX_SAMPLES=2 BATCH_SIZE=1 NUM_WORKERS=1 \
DTYPE=fp32 DIFFUSION_INFERENCE_STEPS=2 COMPUTE_RFID=True COMPUTE_KID=True \
KID_SUBSETS=2 KID_SUBSET_SIZE=2 AR_MAX_NEW_TOKENS=128 \
    bash "${PROJECT_ROOT}/scripts/eval_pgot_oneshot_memory8_register32_noprior_directrae_adapter_contrastive_ar.sh"

"${PYTHON}" "${PROJECT_ROOT}/scripts/verify_oneshot_memory4_smoke.py" \
    "${SMOKE_ROOT}" content --stage eval --owner-prior disabled \
    --direct-rae-query enabled --rae-query-adapter enabled \
    --contrastive enabled --contrastive-lambda 0.03 \
    --contrastive-sampling-rate 0.5 --contrastive-warmup 0 \
    --adapter-bottleneck 384 --object-memories 8 --register-memories 32 \
    --expect-kid

case "${SMOKE_ROOT}" in
    "${PROJECT_ROOT}"/.smoke_oneshot_memory8_register32_directrae_adapter_contrastive.*)
        rm -rf -- "${SMOKE_ROOT}"
        ;;
    *)
        echo "Refusing to remove unexpected smoke path: ${SMOKE_ROOT}" >&2
        exit 2
        ;;
esac
trap - EXIT
echo "PASS: CODA Object-8/Register-32 direct RAE adapter smoke verified; outputs removed."
