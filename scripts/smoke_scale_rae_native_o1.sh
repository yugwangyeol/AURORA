#!/bin/bash
# Tiny native O1 smoke. Deletes its outputs on success.
set -euo pipefail

OUTPUT_DIR="${OUTPUT_DIR:-/home/jovyan/PGOT/outputs/smoke_scale_rae_native_o1}"
GPU="${GPU:-0}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

SCALE_RAE_ENV="${HOME}/.conda/envs/scale_rae"
PYTHON="${PYTHON:-${SCALE_RAE_ENV}/bin/python}"
CONDA_LIB="$(dirname "${PYTHON}")/../lib"
export LD_LIBRARY_PATH="${CONDA_LIB}:${LD_LIBRARY_PATH:-}"
export PYTHONNOUSERSITE=1

rm -rf "${OUTPUT_DIR}"
mkdir -p "${OUTPUT_DIR}"

CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON}" "${PROJECT_ROOT}/pgot/eval/eval_scale_rae_native_o1.py" \
    --model_path /home/jovyan/data/Scale-RAE-Qwen1.5B_DiT2.4B \
    --val_jsonl /home/jovyan/PGOT/data/pgot_val.jsonl \
    --output_dir "${OUTPUT_DIR}" \
    --batch_size 1 \
    --num_workers 0 \
    --max_samples 2 \
    --image_preprocess_mode coda_center_crop \
    --dtype fp32 \
    --diffusion_inference_steps 2 \
    --guidance_scale 1.0

test -s "${OUTPUT_DIR}/summary.json"
test -s "${OUTPUT_DIR}/summary.csv"
"${PYTHON}" - "${OUTPUT_DIR}/summary.json" <<'PY'
import json
import math
import sys
with open(sys.argv[1]) as f:
    row = json.load(f)
required = ["num_samples", "recon_psnr", "recon_ssim", "recon_mse", "recon_mae"]
for key in required:
    val = row[key]
    if not math.isfinite(float(val)):
        raise SystemExit(f"{key} is not finite: {val}")
if int(row["num_samples"]) != 2:
    raise SystemExit(f"expected 2 samples, got {row['num_samples']}")
print("Smoke summary OK:", {k: row[k] for k in required})
PY

rm -rf "${OUTPUT_DIR}"
echo "Native O1 smoke passed and removed ${OUTPUT_DIR}"
