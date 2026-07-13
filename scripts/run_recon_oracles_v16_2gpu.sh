#!/bin/bash
# Reconstruction oracle ladder for V16/V15-style PGOT checkpoints.
# Uses two GPUs by default:
#   GPU0: O0 decoder floor + O2 GT mask-routed projected condition
#   GPU1: O1 GT projected condition + O3 current PGOT condition reference
set -euo pipefail

PGOT_N_NULL_BG="${PGOT_N_NULL_BG:-4}"
MODEL_PATH="${MODEL_PATH:-/home/jovyan/PGOT/checkpoints/pgot_main_v16_bce_bottleneck_void${PGOT_N_NULL_BG}}"
TRAIN_JSONL="${TRAIN_JSONL:-/home/jovyan/PGOT/data/pgot_train.jsonl}"
VAL_JSONL="${VAL_JSONL:-/home/jovyan/PGOT/data/pgot_val.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-/home/jovyan/PGOT/outputs/recon_oracles_$(basename "${MODEL_PATH}")}"

BATCH_SIZE="${BATCH_SIZE:-1}"
NUM_WORKERS="${NUM_WORKERS:-2}"
MAX_SAMPLES="${MAX_SAMPLES:-}"
MAX_TRAIN_SAMPLES="${MAX_TRAIN_SAMPLES:-}"
GRID_SIZE="${GRID_SIZE:-32}"
IMAGE_PREPROCESS_MODE="${IMAGE_PREPROCESS_MODE:-coda_center_crop}"
CODA_CROP_SIZE="${CODA_CROP_SIZE:-512}"
DTYPE="${DTYPE:-bf16}"
DIFF_INFER_STEPS="${DIFF_INFER_STEPS:-25}"
GUIDANCE_SCALE="${GUIDANCE_SCALE:-2.5}"
PROJECTOR_STEPS="${PROJECTOR_STEPS:-1000}"
PROJECTOR_LR="${PROJECTOR_LR:-1e-4}"
REFIT_PROJECTOR="${REFIT_PROJECTOR:-0}"

GPU0="${GPU0:-0}"
GPU1="${GPU1:-1}"
ORACLES_GPU0="${ORACLES_GPU0-o0_decoder_gt,o2_gt_mask_routed_projected}"
ORACLES_GPU1="${ORACLES_GPU1-o1_gt_projected,o3_pgot_condition}"
NUM_GPUS="${NUM_GPUS:-2}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

SCALE_RAE_ENV="${HOME}/.conda/envs/scale_rae"
PYTHON="${SCALE_RAE_ENV}/bin/python"
CONDA_LIB="$(dirname "${PYTHON}")/../lib"
export LD_LIBRARY_PATH="${CONDA_LIB}:${LD_LIBRARY_PATH:-}"
export PYTHONNOUSERSITE=1

mkdir -p "${OUTPUT_DIR}"

COMMON_ARGS=(
    --model_path "${MODEL_PATH}"
    --train_jsonl "${TRAIN_JSONL}"
    --val_jsonl "${VAL_JSONL}"
    --batch_size "${BATCH_SIZE}"
    --num_workers "${NUM_WORKERS}"
    --grid_size "${GRID_SIZE}"
    --max_caption_tokens 2048
    --n_ovt_per_object 2
    --max_objects 50
    --image_preprocess_mode "${IMAGE_PREPROCESS_MODE}"
    --coda_crop_size "${CODA_CROP_SIZE}"
    --dtype "${DTYPE}"
    --diffusion_inference_steps "${DIFF_INFER_STEPS}"
    --guidance_scale "${GUIDANCE_SCALE}"
    --projector_steps "${PROJECTOR_STEPS}"
    --projector_lr "${PROJECTOR_LR}"
)

if [ -n "${MAX_SAMPLES}" ]; then
    COMMON_ARGS+=(--max_samples "${MAX_SAMPLES}")
fi
if [ -n "${MAX_TRAIN_SAMPLES}" ]; then
    COMMON_ARGS+=(--max_train_samples "${MAX_TRAIN_SAMPLES}")
fi
if [ "${REFIT_PROJECTOR}" = "1" ]; then
    COMMON_ARGS+=(--refit_projector)
fi

echo "===== PGOT Reconstruction Oracles ====="
echo "Ckpt:       ${MODEL_PATH}"
echo "Out:        ${OUTPUT_DIR}"
echo "GPUs:       ${NUM_GPUS} (${GPU0}, ${GPU1})"
echo "Batch/GPU:  ${BATCH_SIZE}"
echo "Projector:  ${PROJECTOR_STEPS} steps, lr=${PROJECTOR_LR}"
echo "Sampler:    ${DIFF_INFER_STEPS} steps, guidance=${GUIDANCE_SCALE}"
echo "Max eval:   ${MAX_SAMPLES:-full}"
echo "======================================="

if [ "${NUM_GPUS}" = "1" ]; then
    SINGLE_ORACLES="${ORACLES_GPU0}"
    if [ -n "${ORACLES_GPU1}" ]; then
        SINGLE_ORACLES="${SINGLE_ORACLES},${ORACLES_GPU1}"
    fi
    CUDA_VISIBLE_DEVICES="${GPU0}" "${PYTHON}" "${PROJECT_ROOT}/pgot/eval/eval_recon_oracles.py" \
        "${COMMON_ARGS[@]}" \
        --oracles "${SINGLE_ORACLES}" \
        --output_dir "${OUTPUT_DIR}/single_gpu" \
        2>&1 | tee "${OUTPUT_DIR}/single_gpu.log"
else
    CUDA_VISIBLE_DEVICES="${GPU0}" "${PYTHON}" "${PROJECT_ROOT}/pgot/eval/eval_recon_oracles.py" \
        "${COMMON_ARGS[@]}" \
        --oracles "${ORACLES_GPU0}" \
        --output_dir "${OUTPUT_DIR}/gpu0" \
        2>&1 | tee "${OUTPUT_DIR}/gpu0.log" &
    PID0=$!

    CUDA_VISIBLE_DEVICES="${GPU1}" "${PYTHON}" "${PROJECT_ROOT}/pgot/eval/eval_recon_oracles.py" \
        "${COMMON_ARGS[@]}" \
        --oracles "${ORACLES_GPU1}" \
        --output_dir "${OUTPUT_DIR}/gpu1" \
        2>&1 | tee "${OUTPUT_DIR}/gpu1.log" &
    PID1=$!

    wait "${PID0}"
    wait "${PID1}"
fi

"${PYTHON}" - "${OUTPUT_DIR}" <<'PY'
import csv
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
rows = []
for path in sorted(root.glob("*/summary.json")):
    with open(path) as f:
        data = json.load(f)
    rows.extend(data.get("oracles", []))

order = {
    "o0_decoder_gt": 0,
    "o1_gt_projected": 1,
    "o2_gt_mask_routed_projected": 2,
    "o3_pgot_condition": 3,
}
rows.sort(key=lambda r: order.get(r.get("oracle", ""), 99))

with open(root / "summary.json", "w") as f:
    json.dump({"oracles": rows}, f, indent=2)

fields = [
    "oracle",
    "num_samples",
    "rFID",
    "recon_psnr",
    "recon_ssim",
    "recon_mse",
    "recon_mae",
    "guidance_scale",
    "diffusion_inference_steps",
]
with open(root / "summary.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    for row in rows:
        writer.writerow({k: row.get(k, "") for k in fields})

print(f"Wrote {root / 'summary.csv'}")
for row in rows:
    print(
        f"{row.get('oracle')}: "
        f"rFID={row.get('rFID'):.4f}, "
        f"PSNR={row.get('recon_psnr'):.4f}, "
        f"SSIM={row.get('recon_ssim'):.4f}"
    )
PY
