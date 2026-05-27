#!/bin/bash
# =============================================================================
# PGOT v3 — OVT-swap editing inference.
# For each pair (A, B), swaps a target OVT in A with one from B, re-runs the
# LAST layer, and reconstructs an edited image. Saves 5-up grids per pair.
# =============================================================================
set -e

MODEL_PATH="${MODEL_PATH:-/home/jovyan/PGOT/checkpoints/pgot_main_v3/checkpoint-14000}"
VAL_JSONL="${VAL_JSONL:-/home/jovyan/PGOT/data/pgot_val.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-/home/jovyan/PGOT/outputs/edit_v3_$(basename ${MODEL_PATH})}"
N_PAIRS="${N_PAIRS:-16}"
OBJ_A_IDX="${OBJ_A_IDX:-0}"
OBJ_B_IDX="${OBJ_B_IDX:-0}"
DIFF_INFER_STEPS="${DIFF_INFER_STEPS:-25}"
GUIDANCE_SCALE="${GUIDANCE_SCALE:-1.0}"   # CFG no-op; was previously seen to not help
SEED="${SEED:-42}"

mkdir -p "${OUTPUT_DIR}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH}"

SCALE_RAE_ENV="${HOME}/.conda/envs/scale_rae"
PYTHON="${SCALE_RAE_ENV}/bin/python"
CONDA_LIB="$(dirname "${PYTHON}")/../lib"
export LD_LIBRARY_PATH="${CONDA_LIB}:${LD_LIBRARY_PATH}"
export PYTHONNOUSERSITE=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

echo "===== PGOT v3 OVT-swap editing ====="
echo "Ckpt:           ${MODEL_PATH}"
echo "Val:            ${VAL_JSONL}"
echo "Out:            ${OUTPUT_DIR}"
echo "N pairs:        ${N_PAIRS}"
echo "Swap A_obj:     ${OBJ_A_IDX}  <-  B_obj: ${OBJ_B_IDX}"
echo "Inference step: ${DIFF_INFER_STEPS}"
echo "Guidance:       ${GUIDANCE_SCALE}"
echo "Seed:           ${SEED}"
echo "===================================="

"${PYTHON}" "${PROJECT_ROOT}/pgot/eval/run_swap_editing.py" \
    --model_path "${MODEL_PATH}" \
    --val_jsonl "${VAL_JSONL}" \
    --output_dir "${OUTPUT_DIR}" \
    --n_pairs "${N_PAIRS}" \
    --grid_size 32 \
    --max_caption_tokens 2048 \
    --n_ovt_per_object 2 \
    --max_objects 50 \
    --diffusion_inference_steps "${DIFF_INFER_STEPS}" \
    --guidance_scale "${GUIDANCE_SCALE}" \
    --obj_a_idx "${OBJ_A_IDX}" \
    --obj_b_idx "${OBJ_B_IDX}" \
    --seed "${SEED}" \
    --dtype fp32 \
    2>&1 | tee "${OUTPUT_DIR}/edit.log"
