#!/usr/bin/env bash
# One-pass E11 Dual-M4 follow-up: memory count, spatial assignment,
# donor appearance transfer, and register-only decomposition.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_PATH="${MODEL_PATH:-${PROJECT_ROOT}/checkpoints/pgot_e11_dual_m4/checkpoint-10000}"
VAL_JSONL="${VAL_JSONL:-${PROJECT_ROOT}/data/pgot_pix2cap_generated_val5k.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/outputs/analysis_e11_dual_m4_followup_512}"
PYTHON="${PYTHON:-/home/jovyan/.conda/envs/scale_rae/bin/python}"
SCALE_RAE_ENV="$(dirname "$(dirname "${PYTHON}")")"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="${SCALE_RAE_ENV}/lib:${LD_LIBRARY_PATH:-}"
export PYTHONNOUSERSITE=1

MAX_SAMPLES="${MAX_SAMPLES:-512}"
BATCH_SIZE="${BATCH_SIZE:-2}"
NUM_WORKERS="${NUM_WORKERS:-4}"
TRANSFER_MAX_PAIRS="${TRANSFER_MAX_PAIRS:-64}"
MAX_VISUALIZATIONS="${MAX_VISUALIZATIONS:-12}"
BRANCH_CHUNK_SIZE="${BRANCH_CHUNK_SIZE:-4}"

for path in "${MODEL_PATH}" "${VAL_JSONL}"; do
    if [[ ! -e "${path}" ]]; then
        echo "Missing required path: ${path}" >&2
        exit 1
    fi
done

mkdir -p "${OUTPUT_DIR}"

"${PYTHON}" -m pgot.eval.diagnose_e11_followup \
    --model_path "${MODEL_PATH}" \
    --val_jsonl "${VAL_JSONL}" \
    --output_dir "${OUTPUT_DIR}" \
    --max_samples "${MAX_SAMPLES}" \
    --batch_size "${BATCH_SIZE}" \
    --num_workers "${NUM_WORKERS}" \
    --dtype fp32 \
    --diffusion_inference_steps 10 \
    --guidance_scale 1.0 \
    --branch_chunk_size "${BRANCH_CHUNK_SIZE}" \
    --transfer_max_pairs "${TRANSFER_MAX_PAIRS}" \
    --max_visualizations "${MAX_VISUALIZATIONS}" \
    2>&1 | tee "${OUTPUT_DIR}/analysis.log"
