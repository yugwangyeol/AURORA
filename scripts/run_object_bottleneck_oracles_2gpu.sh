#!/bin/bash
# Train and evaluate independent C_full, C_gtobj(K), and C_ovt ceilings.
set -euo pipefail

MODEL_PATH="${MODEL_PATH:-/home/jovyan/PGOT/checkpoints/pgot_main_v15_1_coda_mmproj}"
TRAIN_JSONL="${TRAIN_JSONL:-/home/jovyan/PGOT/data/pgot_train.jsonl}"
VAL_JSONL="${VAL_JSONL:-/home/jovyan/PGOT/data/pgot_val.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-/home/jovyan/PGOT/outputs/object_bottleneck_oracles_$(basename "${MODEL_PATH}")}"

GPU0="${GPU0:-0}"
GPU1="${GPU1:-1}"
NUM_GPUS="${NUM_GPUS:-2}"
TRAIN_STEPS="${TRAIN_STEPS:-5000}"
BATCH_SIZE="${BATCH_SIZE:-1}"
GRAD_ACCUM="${GRAD_ACCUM:-4}"
MAX_SAMPLES="${MAX_SAMPLES:-}"
MAX_TRAIN_SAMPLES="${MAX_TRAIN_SAMPLES:-}"
DIT_LAST_N_BLOCKS="${DIT_LAST_N_BLOCKS:-8}"
ADAPTER_LR="${ADAPTER_LR:-1e-4}"
DIT_LR="${DIT_LR:-1e-5}"
DIFF_INFER_STEPS="${DIFF_INFER_STEPS:-25}"
GUIDANCE_SCALE="${GUIDANCE_SCALE:-2.5}"
DTYPE="${DTYPE:-bf16}"
NUM_WORKERS="${NUM_WORKERS:-2}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON="${HOME}/.conda/envs/scale_rae/bin/python"
CONDA_LIB="$(dirname "${PYTHON}")/../lib"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export PYTHONNOUSERSITE=1
export LD_LIBRARY_PATH="${CONDA_LIB}:${LD_LIBRARY_PATH:-}"

mkdir -p "${OUTPUT_DIR}"

COMMON_ARGS=(
    --model_path "${MODEL_PATH}"
    --train_jsonl "${TRAIN_JSONL}"
    --val_jsonl "${VAL_JSONL}"
    --batch_size "${BATCH_SIZE}"
    --gradient_accumulation_steps "${GRAD_ACCUM}"
    --num_workers "${NUM_WORKERS}"
    --grid_size 32
    --n_ovt_per_object 2
    --max_objects 50
    --image_preprocess_mode coda_center_crop
    --coda_crop_size 512
    --dtype "${DTYPE}"
    --train_steps "${TRAIN_STEPS}"
    --dit_last_n_blocks "${DIT_LAST_N_BLOCKS}"
    --adapter_lr "${ADAPTER_LR}"
    --dit_lr "${DIT_LR}"
    --diffusion_inference_steps "${DIFF_INFER_STEPS}"
    --guidance_scale "${GUIDANCE_SCALE}"
)
if [ -n "${MAX_SAMPLES}" ]; then
    COMMON_ARGS+=(--max_samples "${MAX_SAMPLES}")
fi
if [ -n "${MAX_TRAIN_SAMPLES}" ]; then
    COMMON_ARGS+=(--max_train_samples "${MAX_TRAIN_SAMPLES}")
fi

run_one() {
    local gpu="$1"
    local oracle="$2"
    local object_tokens="$3"
    local tag="${oracle}"
    if [ "${oracle}" = "c_gtobj" ]; then
        tag="${oracle}_k${object_tokens}"
    fi
    echo "[$(date -u +%FT%TZ)] start ${tag} on GPU ${gpu}"
    CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON}" -m pgot.eval.train_object_bottleneck_oracle \
        "${COMMON_ARGS[@]}" \
        --oracle "${oracle}" \
        --object_tokens "${object_tokens}" \
        --output_dir "${OUTPUT_DIR}/${tag}" \
        2>&1 | tee "${OUTPUT_DIR}/${tag}.log"
}

echo "===== PGOT Object Bottleneck Oracle Ceilings ====="
echo "Model:        ${MODEL_PATH}"
echo "Output:       ${OUTPUT_DIR}"
echo "Train steps:  ${TRAIN_STEPS}"
echo "DiT blocks:   last ${DIT_LAST_N_BLOCKS}"
echo "GPUs:         ${NUM_GPUS} (${GPU0}, ${GPU1})"
echo "=================================================="

if [ "${NUM_GPUS}" = "1" ]; then
    run_one "${GPU0}" c_full 4
    run_one "${GPU0}" c_gtobj 2
    run_one "${GPU0}" c_gtobj 4
    run_one "${GPU0}" c_gtobj 8
    run_one "${GPU0}" c_ovt 4
else
    (
        run_one "${GPU0}" c_full 4
        run_one "${GPU0}" c_gtobj 4
        run_one "${GPU0}" c_ovt 4
    ) &
    PID0=$!
    (
        run_one "${GPU1}" c_gtobj 2
        run_one "${GPU1}" c_gtobj 8
    ) &
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
    with open(path) as handle:
        rows.append(json.load(handle))
rows.sort(key=lambda row: (row["oracle"], row.get("object_tokens") or 0))

with open(root / "summary.json", "w") as handle:
    json.dump({"oracles": rows}, handle, indent=2)

fields = [
    "oracle", "object_tokens", "num_samples", "rFID", "recon_psnr",
    "recon_ssim", "recon_mse", "recon_mae", "train_steps",
    "last_train_loss", "mean_last_20_train_loss", "dit_last_n_blocks",
]
with open(root / "summary.csv", "w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=fields)
    writer.writeheader()
    for row in rows:
        writer.writerow({key: row.get(key, "") for key in fields})

for row in rows:
    suffix = f" K={row['object_tokens']}" if row.get("object_tokens") else ""
    print(
        f"{row['oracle']}{suffix}: rFID={row['rFID']:.4f}, "
        f"PSNR={row['recon_psnr']:.4f}, loss={row['last_train_loss']:.4f}"
    )
PY

echo "Completed: ${OUTPUT_DIR}/summary.csv"
