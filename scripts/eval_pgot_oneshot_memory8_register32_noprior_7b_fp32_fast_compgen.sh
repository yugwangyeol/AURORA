#!/usr/bin/env bash
# Same compositional protocol as the M8/R32 baseline, using the 7B FP32 fast run.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export MODEL_PATH="${MODEL_PATH:-${PROJECT_ROOT}/checkpoints/pgot_oneshot_memory8_register32_noprior_7b_fp32_fast/checkpoint-${MAX_STEPS:-5000}}"
export OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/outputs/compgen_pgot_oneshot_memory8_register32_noprior_7b_fp32_fast}"
export DTYPE=fp32
# This batch is also the donor mixing pool; retain four images for comparison.
export BATCH_SIZE="${BATCH_SIZE:-4}"
export MIX_MODE="${MIX_MODE:-random_slots}"
export REGISTER_SOURCE="${REGISTER_SOURCE:-random}"
export SEED="${SEED:-1234}"
export GUIDANCE_SCALE="${GUIDANCE_SCALE:-1.0}"
export DIFFUSION_INFERENCE_STEPS="${DIFFUSION_INFERENCE_STEPS:-10}"

mkdir -p "$(dirname "${OUTPUT_DIR}")"
bash "${PROJECT_ROOT}/scripts/eval_pgot_compositional_gen.sh" "$@" \
    2>&1 | tee "${OUTPUT_DIR}.log"
