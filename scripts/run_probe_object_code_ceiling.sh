#!/bin/bash
# Probe: rFID ceiling of an object-centric code vs. number of code vectors per object.
# No training, no DiT, no projector — GT SigLIP features pooled per object mask, routed
# back onto the patch grid, decoded by the frozen RAE decoder.
set -euo pipefail

MODEL_PATH="${MODEL_PATH:-/home/jovyan/PGOT/checkpoints/pgot_main_v15_bce_bottleneck}"
VAL_JSONL="${VAL_JSONL:-/home/jovyan/PGOT/data/pgot_val.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-/home/jovyan/PGOT/outputs/probe_object_code_ceiling}"
CODES_PER_OBJECT="${CODES_PER_OBJECT:-1,2,4,8,16}"
MAX_SAMPLES="${MAX_SAMPLES:-}"
BATCH_SIZE="${BATCH_SIZE:-4}"
NUM_WORKERS="${NUM_WORKERS:-4}"
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
    --codes_per_object "${CODES_PER_OBJECT}"
    --batch_size "${BATCH_SIZE}"
    --num_workers "${NUM_WORKERS}"
    --dtype "${DTYPE}"
)
if [[ -n "${MAX_SAMPLES}" ]]; then
    ARGS+=(--max_samples "${MAX_SAMPLES}")
fi

echo "[ceiling] GPU=${GPU} n_q=${CODES_PER_OBJECT} out=${OUTPUT_DIR} max_samples=${MAX_SAMPLES:-ALL}"
"${PYTHON}" -m pgot.eval.probe_object_code_ceiling "${ARGS[@]}" 2>&1 | tee "${OUTPUT_DIR}/probe.log"
