#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-/home/jovyan/.conda/envs/scale_rae/bin/python}"
MODEL_PATH="${MODEL_PATH:?Set MODEL_PATH}"
OUTPUT_DIR="${OUTPUT_DIR:?Set OUTPUT_DIR}"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export PYTHONNOUSERSITE=1
export LD_LIBRARY_PATH="$(dirname "${PYTHON}")/../lib:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS="${CPU_THREADS:-3}"
export MKL_NUM_THREADS="${CPU_THREADS:-3}"

exec "${PYTHON}" -u -m pgot.eval.eval_object_memory_transfer \
    --model_path "${MODEL_PATH}" \
    --val_jsonl "${VAL_JSONL:-${PROJECT_ROOT}/data/pgot_pix2cap_generated_val5k.jsonl}" \
    --output_dir "${OUTPUT_DIR}" --max_pairs "${MAX_PAIRS:-100}" \
    --max_samples "${MAX_SAMPLES:-1024}" --batch_size "${BATCH_SIZE:-1}" \
    --num_workers "${NUM_WORKERS:-2}" --dtype "${DTYPE:-fp32}" \
    --diffusion_inference_steps "${DIFFUSION_INFERENCE_STEPS:-10}" \
    --guidance_scale "${GUIDANCE_SCALE:-1.0}" --seed "${SEED:-123}"
