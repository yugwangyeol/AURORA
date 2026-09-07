#!/usr/bin/env bash
# Sequential order: pooled train/eval, then owner-masked train/eval.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MAX_STEPS="${MAX_STEPS:-5000}"
PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-6}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-24}"
mkdir -p "${PROJECT_ROOT}/outputs"

run_one() {
    local name="$1"
    local train_script="$2"
    local eval_script="$3"
    local train_dir="${PROJECT_ROOT}/checkpoints/${name}"
    local eval_dir="${PROJECT_ROOT}/outputs/eval_${name}"

    OUTPUT_DIR="${train_dir}" MAX_STEPS="${MAX_STEPS}" \
    PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE}" \
    GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE}" \
        bash "${PROJECT_ROOT}/scripts/${train_script}" \
        2>&1 | tee "${PROJECT_ROOT}/outputs/train_${name}.log"

    MODEL_PATH="${train_dir}/checkpoint-${MAX_STEPS}" OUTPUT_DIR="${eval_dir}" \
        bash "${PROJECT_ROOT}/scripts/${eval_script}" \
        2>&1 | tee "${PROJECT_ROOT}/outputs/eval_${name}.log"
}

run_one pgot_oneshot_pooled \
    train_pgot_oneshot_pooled.sh eval_pgot_oneshot_pooled.sh
run_one pgot_oneshot_owner_masked \
    train_pgot_oneshot_owner_masked.sh eval_pgot_oneshot_owner_masked.sh
