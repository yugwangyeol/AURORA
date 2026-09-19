#!/usr/bin/env bash
# Recompute the three fixed qualitative figures with the checkpoint trained
# without Reader supervision. Baseline masks and selected sample IDs are fixed.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

PYTHON="${PYTHON:-/home/jovyan/.conda/envs/scale_rae/bin/python}"
GPU="${GPU:-0}"
MODEL_PATH="${PROJECT_ROOT}/checkpoints/pgot_oneshot_memory8_register32_noprior_woreader/checkpoint-5000"
SOURCE_DIR="${PROJECT_ROOT}/outputs/figure_segmentation_woreader_fixed"
REMOVE_DIR="${PROJECT_ROOT}/outputs/figure_model_editing_woreader_fixed3"
TRANSFER_DIR="${PROJECT_ROOT}/outputs/object_representation_transfer_woreader_fixed3"
PAIR_FILE="${PROJECT_ROOT}/configs/object_representation_transfer_woreader_figure3.jsonl"
COCO_CAPTIONS="${PROJECT_ROOT}/outputs/eval_pgot_oneshot_memory8_register32_noprior_woreader/ar/generated_captions.jsonl"
COCO_SUMMARY="${PROJECT_ROOT}/outputs/eval_pgot_oneshot_memory8_register32_noprior_woreader/ar/summary.json"

export CUDA_VISIBLE_DEVICES="${GPU}"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export PYTHONNOUSERSITE=1
export LD_LIBRARY_PATH="$(dirname "${PYTHON}")/../lib:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-3}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-3}"

"${PYTHON}" -m pgot.eval.verify_woreader_config --model_path "${MODEL_PATH}"
"${PYTHON}" -m pgot.eval.rebuild_woreader_paper_figures --stage prepare_source

"${PYTHON}" -m pgot.eval.figure_segmentation_examples \
    --stage infer --model ours --output_dir "${SOURCE_DIR}" \
    --model_path "${MODEL_PATH}" --seed 0 --dtype fp32 \
    2>&1 | tee "${SOURCE_DIR}/ours.log"

"${PYTHON}" -m pgot.eval.rebuild_woreader_paper_figures --stage compose_non_editing
"${PYTHON}" -m pgot.eval.rebuild_woreader_paper_figures --stage prepare_remove
"${PYTHON}" -m pgot.eval.rebuild_woreader_paper_figures --stage prepare_transfer

"${PYTHON}" -m pgot.eval.figure_model_editing \
    --stage render --model_path "${MODEL_PATH}" --source_dir "${SOURCE_DIR}" \
    --output_dir "${REMOVE_DIR}" --dtype fp32 --diffusion_inference_steps 10 \
    --guidance_scale 1 --seed 1234 \
    2>&1 | tee "${REMOVE_DIR}/render.log"

"${PYTHON}" -m pgot.eval.run_object_representation_transfer \
    --stage prepare --model_path "${MODEL_PATH}" --pairs "${PAIR_FILE}" \
    --captions "${COCO_CAPTIONS}" --caption_summary "${COCO_SUMMARY}" \
    --output_dir "${TRANSFER_DIR}" --diffusion_inference_steps 10 \
    --guidance_scale 1 --seed 1234

"${PYTHON}" -m pgot.eval.run_object_representation_transfer \
    --stage render --model_path "${MODEL_PATH}" --pairs "${PAIR_FILE}" \
    --captions "${COCO_CAPTIONS}" --caption_summary "${COCO_SUMMARY}" \
    --output_dir "${TRANSFER_DIR}" --diffusion_inference_steps 10 \
    --guidance_scale 1 --seed 1234 \
    2>&1 | tee "${TRANSFER_DIR}/render.log"

"${PYTHON}" -m pgot.eval.run_object_representation_transfer \
    --stage audit --model_path "${MODEL_PATH}" --pairs "${PAIR_FILE}" \
    --captions "${COCO_CAPTIONS}" --caption_summary "${COCO_SUMMARY}" \
    --output_dir "${TRANSFER_DIR}" --diffusion_inference_steps 10 \
    --guidance_scale 1 --seed 1234

"${PYTHON}" -m pgot.eval.rebuild_woreader_paper_figures --stage compose_editing

echo "All no-Reader-supervision figures are ready under outputs/paper_figures/."
