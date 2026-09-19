#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export MODEL_PATH="${MODEL_PATH:-${PROJECT_ROOT}/checkpoints/pgot_oneshot_memory8_register32_noprior_7b_fp32_fast_woreader/checkpoint-${MAX_STEPS:-5000}}"
export OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/outputs/eval_pgot_oneshot_memory8_register32_noprior_7b_fp32_fast_woreader/compgen}"
AUDIT_PYTHON="${AUDIT_PYTHON:-/home/jovyan/.conda/envs/scale_rae/bin/python}"
PYTHONNOUSERSITE=1 PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}" \
LD_LIBRARY_PATH="$(dirname "${AUDIT_PYTHON}")/../lib:${LD_LIBRARY_PATH:-}" \
"${AUDIT_PYTHON}" -m pgot.eval.verify_woreader_config \
    --model_path "${MODEL_PATH}" --model_size 7b
export DTYPE=fp32 BATCH_SIZE="${BATCH_SIZE:-4}"
export MIX_MODE="${MIX_MODE:-random_slots}" REGISTER_SOURCE="${REGISTER_SOURCE:-random}"
export SEED="${SEED:-1234}" GUIDANCE_SCALE="${GUIDANCE_SCALE:-1.0}"
export DIFFUSION_INFERENCE_STEPS="${DIFFUSION_INFERENCE_STEPS:-10}"
exec bash "${PROJECT_ROOT}/scripts/eval_pgot_compositional_gen.sh" "$@"
