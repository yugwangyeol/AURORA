#!/bin/bash
# True O1 native Scale-RAE reconstruction oracle sweep.
# Two-GPU default:
#   GPU0: 25-step guidance sweep
#   GPU1: 50-step guidance sweep
set -euo pipefail

MODEL_PATH="${MODEL_PATH:-/home/jovyan/data/Scale-RAE-Qwen1.5B_DiT2.4B}"
VAL_JSONL="${VAL_JSONL:-/home/jovyan/PGOT/data/pgot_val.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-/home/jovyan/PGOT/outputs/scale_rae_native_o1_sweep}"

BATCH_SIZE="${BATCH_SIZE:-1}"
NUM_WORKERS="${NUM_WORKERS:-2}"
MAX_SAMPLES="${MAX_SAMPLES:-}"
IMAGE_PREPROCESS_MODE="${IMAGE_PREPROCESS_MODE:-coda_center_crop}"
CODA_CROP_SIZE="${CODA_CROP_SIZE:-512}"
DTYPE="${DTYPE:-fp32}"

GPU0="${GPU0:-0}"
GPU1="${GPU1:-1}"
NUM_GPUS="${NUM_GPUS:-2}"
GUIDANCE_VALUES="${GUIDANCE_VALUES:-1.0 1.5 2.5}"
STEPS_GPU0="${STEPS_GPU0-25}"
STEPS_GPU1="${STEPS_GPU1-50}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

SCALE_RAE_ENV="${HOME}/.conda/envs/scale_rae"
PYTHON="${PYTHON:-${SCALE_RAE_ENV}/bin/python}"
CONDA_LIB="$(dirname "${PYTHON}")/../lib"
export LD_LIBRARY_PATH="${CONDA_LIB}:${LD_LIBRARY_PATH:-}"
export PYTHONNOUSERSITE=1

mkdir -p "${OUTPUT_DIR}"

COMMON_ARGS=(
    --model_path "${MODEL_PATH}"
    --val_jsonl "${VAL_JSONL}"
    --batch_size "${BATCH_SIZE}"
    --num_workers "${NUM_WORKERS}"
    --image_preprocess_mode "${IMAGE_PREPROCESS_MODE}"
    --coda_crop_size "${CODA_CROP_SIZE}"
    --dtype "${DTYPE}"
)
if [ -n "${MAX_SAMPLES}" ]; then
    COMMON_ARGS+=(--max_samples "${MAX_SAMPLES}")
fi

run_group() {
    local gpu="$1"
    local steps_list="$2"
    local tag="$3"
    for steps in ${steps_list}; do
        for guidance in ${GUIDANCE_VALUES}; do
            local run_name="steps${steps}_guidance${guidance}"
            local run_dir="${OUTPUT_DIR}/${run_name}"
            echo "[${tag}] ${run_name} on GPU ${gpu}"
            CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON}" "${PROJECT_ROOT}/pgot/eval/eval_scale_rae_native_o1.py" \
                "${COMMON_ARGS[@]}" \
                --diffusion_inference_steps "${steps}" \
                --guidance_scale "${guidance}" \
                --output_dir "${run_dir}" \
                2>&1 | tee "${OUTPUT_DIR}/${run_name}.log"
        done
    done
}

echo "===== Native Scale-RAE O1 Sweep ====="
echo "Model:      ${MODEL_PATH}"
echo "Out:        ${OUTPUT_DIR}"
echo "GPUs:       ${NUM_GPUS} (${GPU0}, ${GPU1})"
echo "Guidance:   ${GUIDANCE_VALUES}"
echo "GPU0 steps: ${STEPS_GPU0}"
echo "GPU1 steps: ${STEPS_GPU1}"
echo "Max eval:   ${MAX_SAMPLES:-full}"
echo "====================================="

if [ "${NUM_GPUS}" = "1" ]; then
    SINGLE_STEPS="${STEPS_GPU0}"
    if [ -n "${STEPS_GPU1}" ]; then
        SINGLE_STEPS="${SINGLE_STEPS} ${STEPS_GPU1}"
    fi
    run_group "${GPU0}" "${SINGLE_STEPS}" "single"
else
    run_group "${GPU0}" "${STEPS_GPU0}" "gpu0" &
    PID0=$!
    run_group "${GPU1}" "${STEPS_GPU1}" "gpu1" &
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
for path in sorted(root.glob("steps*_guidance*/summary.json")):
    with open(path) as f:
        rows.append(json.load(f))

def key(row):
    return (int(row.get("diffusion_inference_steps", 0)), float(row.get("guidance_scale", 0.0)))

rows.sort(key=key)
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
    "condition_mean",
    "condition_std",
    "rae_hidden_mean",
    "rae_hidden_std",
    "dtype",
]
with open(root / "summary.json", "w") as f:
    json.dump({"runs": rows}, f, indent=2)
with open(root / "summary.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    for row in rows:
        writer.writerow({k: row.get(k, "") for k in fields})

print(f"Wrote {root / 'summary.csv'}")
for row in rows:
    print(
        f"steps={row.get('diffusion_inference_steps')} "
        f"guidance={row.get('guidance_scale')}: "
        f"rFID={row.get('rFID'):.4f}, "
        f"PSNR={row.get('recon_psnr'):.4f}, "
        f"SSIM={row.get('recon_ssim'):.4f}"
    )
PY
