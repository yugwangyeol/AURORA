#!/usr/bin/env bash
# Existing 7B checkpoint: compositional generation, scored at 256x256.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GPU="${GPU:?Set GPU to one physical CUDA index}"
[[ "${GPU}" != *,* ]] || { echo "GPU must name exactly one device" >&2; exit 2; }
export CUDA_VISIBLE_DEVICES="${GPU}"
export MODEL_PATH="${MODEL_PATH:-${PROJECT_ROOT}/checkpoints/pgot_oneshot_memory8_register32_noprior_7b_fp32_fast/checkpoint-5000}"
export OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/outputs/compgen_pgot_oneshot_memory8_register32_noprior_7b_fp32_fast_256}"
AUDIT_PYTHON="${AUDIT_PYTHON:-/home/jovyan/.conda/envs/scale_rae/bin/python}"

PYTHONNOUSERSITE=1 PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}" \
LD_LIBRARY_PATH="$(dirname "${AUDIT_PYTHON}")/../lib:${LD_LIBRARY_PATH:-}" \
"${AUDIT_PYTHON}" -m pgot.eval.verify_oneshot_o8r32_config \
    --model_path "${MODEL_PATH}" --model_size 7b --require_scratch_init

export DTYPE=fp32
export BATCH_SIZE="${BATCH_SIZE:-4}"
export METRIC_RESOLUTION=256
export MIX_MODE="${MIX_MODE:-random_slots}"
export REGISTER_SOURCE="${REGISTER_SOURCE:-random}"
export DIFFUSION_INFERENCE_STEPS="${DIFFUSION_INFERENCE_STEPS:-10}"

mkdir -p "$(dirname "${OUTPUT_DIR}")"
bash "${PROJECT_ROOT}/scripts/eval_pgot_compositional_gen.sh" \
    2>&1 | tee "${OUTPUT_DIR}.log"
