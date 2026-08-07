#!/usr/bin/env bash
# Teacher-forced E8.1/E8.2 ownership and Scale-RAE reconstruction evaluation.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_PATH="${MODEL_PATH:-${PROJECT_ROOT}/checkpoints/pgot_e8_1_clean/checkpoint-10000}"
VAL_JSONL="${VAL_JSONL:-${PROJECT_ROOT}/data/pgot_pix2cap_generated_val5k.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/outputs/eval_pgot_e8_1_clean}"
COCO_MASK_CACHE="${COCO_MASK_CACHE:-${PROJECT_ROOT}/data/coco_inst_mask_cache_coda512}"
PYTHON="${PYTHON:-/home/jovyan/.conda/envs/scale_rae/bin/python}"

for path in "${MODEL_PATH}" "${VAL_JSONL}" "${COCO_MASK_CACHE}/meta.json"; do
    test -e "${path}" || { echo "Missing required path: ${path}" >&2; exit 1; }
done
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export PYTHONNOUSERSITE=1
export LD_LIBRARY_PATH="$(dirname "${PYTHON}")/../lib:${LD_LIBRARY_PATH:-}"

EXTRA_ARGS=()
if [[ -n "${MAX_SAMPLES:-}" ]]; then EXTRA_ARGS+=(--max_samples "${MAX_SAMPLES}"); fi
if [[ "${COMPUTE_RFID:-True}" == True ]]; then EXTRA_ARGS+=(--compute_rfid); fi

"${PYTHON}" -m pgot.eval.run_eval \
    --model_path "${MODEL_PATH}" --val_jsonl "${VAL_JSONL}" --output_dir "${OUTPUT_DIR}" \
    --batch_size "${BATCH_SIZE:-4}" --num_workers "${NUM_WORKERS:-4}" \
    --grid_size 32 --max_caption_tokens 1024 --n_ovt_per_object 1 --max_objects 50 \
    --eval_size 512 --readout ovt_owner --eval_merge mean \
    --gt_source coco_instance --coco_mask_cache "${COCO_MASK_CACHE}" \
    --image_preprocess_mode coda_center_crop --coda_crop_size 512 \
    --register_eval_route unrestricted \
    --dtype "${DTYPE:-fp32}" --guidance_scale "${GUIDANCE_SCALE:-1.0}" \
    --diffusion_inference_steps "${DIFFUSION_INFERENCE_STEPS:-10}" "${EXTRA_ARGS[@]}"
