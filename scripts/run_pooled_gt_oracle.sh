#!/usr/bin/env bash
# Inference-only pooled-GT latent quantization sweep for E11 Dual-M4.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_PATH="${MODEL_PATH:-${PROJECT_ROOT}/checkpoints/pgot_e10_r_raw_value/checkpoint-9000}"
VAL_JSONL="${VAL_JSONL:-${PROJECT_ROOT}/data/pgot_pix2cap_generated_val5k.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/outputs/pooled_gt_oracle_e11}"
PYTHON="${PYTHON:-/home/jovyan/.conda/envs/scale_rae/bin/python}"

for path in "${MODEL_PATH}" "${VAL_JSONL}"; do
    test -e "${path}" || { echo "Missing required path: ${path}" >&2; exit 1; }
done

export CUDA_VISIBLE_DEVICES="${GPU:-0}"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export PYTHONNOUSERSITE=1
export LD_LIBRARY_PATH="$(dirname "${PYTHON}")/../lib:${LD_LIBRARY_PATH:-}"

EXTRA_ARGS=()
if [[ -n "${MAX_SAMPLES:-}" ]]; then
    EXTRA_ARGS+=(--max_samples "${MAX_SAMPLES}")
fi

mkdir -p "${OUTPUT_DIR}"

"${PYTHON}" -m pgot.eval.probe_pooled_gt_oracle \
    --model_path "${MODEL_PATH}" \
    --val_jsonl "${VAL_JSONL}" \
    --output_dir "${OUTPUT_DIR}" \
    --object_codes "${OBJECT_CODES:-2,4,8}" \
    --background_codes "${BACKGROUND_CODES:-4,8,16}" \
    --cluster_modes "${CLUSTER_MODES:-coordinate,kmeans}" \
    --fg_threshold "${FG_THRESHOLD:-0.5}" \
    --kmeans_iterations "${KMEANS_ITERATIONS:-20}" \
    --batch_size "${BATCH_SIZE:-4}" \
    --num_workers "${NUM_WORKERS:-4}" \
    --grid_size 32 \
    --max_caption_tokens 1024 \
    --n_ovt_per_object 1 \
    --max_objects 50 \
    --image_preprocess_mode coda_center_crop \
    --coda_crop_size 512 \
    --dtype "${DTYPE:-fp32}" \
    "${EXTRA_ARGS[@]}" \
    2>&1 | tee "${OUTPUT_DIR}/pooled_gt_oracle.log"
