#!/usr/bin/env bash
# Full 5K compositional-generation evaluation for the 1.5B O4/R32 ablation.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export MODEL_PATH="${MODEL_PATH:-${PROJECT_ROOT}/checkpoints/pgot_oneshot_memory4_register32_noprior_scratch/checkpoint-${MAX_STEPS:-5000}}"
export OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/outputs/eval_pgot_oneshot_memory4_register32_noprior_scratch/compgen}"
export DTYPE="${DTYPE:-fp32}"
export BATCH_SIZE="${COMP_BATCH_SIZE:-4}"
export EXPECTED_SAMPLES="${EXPECTED_SAMPLES:-5000}"
export METRIC_RESOLUTION="${METRIC_RESOLUTION:-0}"
export MIX_MODE="${MIX_MODE:-random_slots}"
export REGISTER_SOURCE="${REGISTER_SOURCE:-random}"
export SEED="${SEED:-1234}"
export GUIDANCE_SCALE="${GUIDANCE_SCALE:-1.0}"
export DIFFUSION_INFERENCE_STEPS="${DIFFUSION_INFERENCE_STEPS:-10}"

exec bash "${PROJECT_ROOT}/scripts/eval_pgot_compositional_gen.sh" "$@"
