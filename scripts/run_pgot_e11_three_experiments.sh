#!/usr/bin/env bash
# Sequentially run: causal train/eval, direct train/eval, routed train/eval.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MAX_STEPS="${MAX_STEPS:-5000}"
mkdir -p "${PROJECT_ROOT}/outputs"

run_one() {
    local name="$1"
    local train_script="$2"
    local eval_script="$3"
    local train_dir="${PROJECT_ROOT}/checkpoints/${name}"
    local eval_dir="${PROJECT_ROOT}/outputs/eval_${name}"

    OUTPUT_DIR="${train_dir}" MAX_STEPS="${MAX_STEPS}" \
        bash "${PROJECT_ROOT}/scripts/${train_script}" \
        2>&1 | tee "${PROJECT_ROOT}/outputs/train_${name}.log"

    MODEL_PATH="${train_dir}/checkpoint-${MAX_STEPS}" OUTPUT_DIR="${eval_dir}" \
        bash "${PROJECT_ROOT}/scripts/${eval_script}" \
        2>&1 | tee "${PROJECT_ROOT}/outputs/eval_${name}.log"
}

run_one pgot_e11_causal_only \
    train_pgot_e11_causal_only.sh eval_pgot_e11_causal_only.sh
run_one pgot_e11_direct_causal \
    train_pgot_e11_direct_causal.sh eval_pgot_e11_direct_causal.sh
run_one pgot_e11_direct_causal_routed \
    train_pgot_e11_direct_causal_routed.sh eval_pgot_e11_direct_causal_routed.sh
