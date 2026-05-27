#!/bin/bash
# =============================================================================
# PGOT v3 — Single-pair OVT-swap editing (qualitative).
# Save ONE PNG showing [Source A | GT-decode A | A self-recon | A with OVT(B) | Source B].
#
# Usage:
#   IMAGE_ID_A=302760 IMAGE_ID_B=452122 OBJ_A=0 OBJ_B=0 \
#     bash scripts/edit_pair.sh
#
#   IDX_A=10 IDX_B=200 OBJ_A=1 OBJ_B=0 \
#     bash scripts/edit_pair.sh
# =============================================================================
set -e

MODEL_PATH="${MODEL_PATH:-/home/jovyan/PGOT/checkpoints/pgot_main_v3/checkpoint-14000}"
VAL_JSONL="${VAL_JSONL:-/home/jovyan/PGOT/data/pgot_val.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-/home/jovyan/PGOT/outputs/edit_pairs}"

OBJ_A="${OBJ_A:-0}"
OBJ_B="${OBJ_B:-0}"
DIFF_INFER_STEPS="${DIFF_INFER_STEPS:-25}"
GUIDANCE_SCALE="${GUIDANCE_SCALE:-1.0}"
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

# Pair specification (one of two modes)
MANUAL_ARGS=()
if [ -n "${IMAGE_ID_A}" ] && [ -n "${IMAGE_ID_B}" ]; then
    MANUAL_ARGS+=(--image_id_a "${IMAGE_ID_A}" --image_id_b "${IMAGE_ID_B}")
    PAIR_TAG="imgA${IMAGE_ID_A}_imgB${IMAGE_ID_B}"
elif [ -n "${IDX_A}" ] && [ -n "${IDX_B}" ]; then
    MANUAL_ARGS+=(--idx_a "${IDX_A}" --idx_b "${IDX_B}")
    PAIR_TAG="idxA${IDX_A}_idxB${IDX_B}"
else
    echo "ERROR: must set either (IMAGE_ID_A and IMAGE_ID_B) or (IDX_A and IDX_B)"
    exit 1
fi

PAIR_OUT="${OUTPUT_DIR}/${PAIR_TAG}_objA${OBJ_A}_objB${OBJ_B}_g${GUIDANCE_SCALE//./_}"
mkdir -p "${PAIR_OUT}"

echo "===== PGOT single-pair edit ====="
echo "Ckpt:           ${MODEL_PATH}"
echo "Pair:           ${PAIR_TAG}"
echo "Obj swap:       A_obj=${OBJ_A}  <-  B_obj=${OBJ_B}"
echo "Guidance:       ${GUIDANCE_SCALE}"
echo "Inference step: ${DIFF_INFER_STEPS}"
echo "Output:         ${PAIR_OUT}"
echo "================================"

"${PYTHON}" "${PROJECT_ROOT}/pgot/eval/run_swap_editing.py" \
    --model_path "${MODEL_PATH}" \
    --val_jsonl "${VAL_JSONL}" \
    --output_dir "${PAIR_OUT}" \
    --grid_size 32 \
    --max_caption_tokens 2048 \
    --n_ovt_per_object 2 \
    --max_objects 50 \
    --diffusion_inference_steps "${DIFF_INFER_STEPS}" \
    --guidance_scale "${GUIDANCE_SCALE}" \
    --obj_a_idx "${OBJ_A}" \
    --obj_b_idx "${OBJ_B}" \
    --seed "${SEED}" \
    --dtype fp32 \
    "${MANUAL_ARGS[@]}" \
    2>&1 | tee "${PAIR_OUT}/edit.log"

echo ""
echo "✅ Saved single-pair edit to: ${PAIR_OUT}"
ls -la "${PAIR_OUT}"
