#!/bin/bash
# Probe: does the OVT hidden state carry object appearance, or only semantics?
# No training — caches frozen features on the val set and fits closed-form ridge probes.
set -euo pipefail

MODEL_PATH="${MODEL_PATH:-/home/jovyan/PGOT/checkpoints/pgot_main_v15_bce_bottleneck}"
VAL_JSONL="${VAL_JSONL:-/home/jovyan/PGOT/data/pgot_val.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-/home/jovyan/PGOT/outputs/probe_ovt_appearance}"
MAX_SAMPLES="${MAX_SAMPLES:-}"
SPLIT_RATIO="${SPLIT_RATIO:-0.8}"
BATCH_SIZE="${BATCH_SIZE:-1}"
NUM_WORKERS="${NUM_WORKERS:-2}"
DTYPE="${DTYPE:-fp32}"
GPU="${GPU:-1}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

SCALE_RAE_ENV="${HOME}/.conda/envs/scale_rae"
PYTHON="${SCALE_RAE_ENV}/bin/python"
export LD_LIBRARY_PATH="${SCALE_RAE_ENV}/lib:${LD_LIBRARY_PATH:-}"
export PYTHONNOUSERSITE=1
export CUDA_VISIBLE_DEVICES="${GPU}"

mkdir -p "${OUTPUT_DIR}"

ARGS=(
    --model_path "${MODEL_PATH}"
    --val_jsonl "${VAL_JSONL}"
    --output_dir "${OUTPUT_DIR}"
    --split_ratio "${SPLIT_RATIO}"
    --batch_size "${BATCH_SIZE}"
    --num_workers "${NUM_WORKERS}"
    --dtype "${DTYPE}"
)
if [[ -n "${MAX_SAMPLES}" ]]; then
    ARGS+=(--max_samples "${MAX_SAMPLES}")
fi

echo "[probe] GPU=${GPU} model=${MODEL_PATH} out=${OUTPUT_DIR} max_samples=${MAX_SAMPLES:-ALL}"
"${PYTHON}" -m pgot.eval.probe_ovt_appearance "${ARGS[@]}" 2>&1 | tee "${OUTPUT_DIR}/probe.log"
