#!/bin/bash
set -euo pipefail

MODEL_PATH="${MODEL_PATH:-/home/jovyan/PGOT/checkpoints/pgot_instance_e1/checkpoint-6000}"
VAL_JSONL="${VAL_JSONL:-/home/jovyan/PGOT/data/pgot_coco_instance_val.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-/home/jovyan/PGOT/outputs/eval_pgot_instance_e1}"
PYTHON="${PYTHON:-/home/jovyan/.conda/envs/scale_rae/bin/python}"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export PYTHONNOUSERSITE=1
CONDA_LIB="$(dirname "${PYTHON}")/../lib"
export LD_LIBRARY_PATH="${CONDA_LIB}:${LD_LIBRARY_PATH:-}"

EXTRA_ARGS=()
if [ -n "${MAX_SAMPLES:-}" ]; then EXTRA_ARGS+=(--max_samples "${MAX_SAMPLES}"); fi
if [ "${COMPUTE_RFID:-False}" = True ]; then EXTRA_ARGS+=(--compute_rfid); fi

"${PYTHON}" -m pgot.eval.run_eval \
    --model_path "${MODEL_PATH}" --val_jsonl "${VAL_JSONL}" --output_dir "${OUTPUT_DIR}" \
    --batch_size "${BATCH_SIZE:-2}" --num_workers "${NUM_WORKERS:-2}" \
    --grid_size 32 --max_caption_tokens 1024 --n_ovt_per_object 1 --max_objects 50 \
    --eval_size 224 --readout llm_attention --eval_merge mean \
    --gt_source coco_instance \
    --coco_mask_cache "${COCO_MASK_CACHE:-/home/jovyan/PGOT/data/coco_inst_mask_cache_coda256}" \
    --image_preprocess_mode coda_center_crop --coda_crop_size 512 \
    --dtype "${DTYPE:-fp32}" --guidance_scale "${GUIDANCE_SCALE:-2.5}" \
    --diffusion_inference_steps "${DIFFUSION_INFERENCE_STEPS:-10}" "${EXTRA_ARGS[@]}"
