#!/bin/bash
# Validate the frozen PGOT path, then train stable object-bottleneck oracles.
set -euo pipefail

MODEL_PATH="${MODEL_PATH:-/home/jovyan/PGOT/checkpoints/pgot_main_v15_1_coda_mmproj}"
TRAIN_JSONL="${TRAIN_JSONL:-/home/jovyan/PGOT/data/pgot_train.jsonl}"
VAL_JSONL="${VAL_JSONL:-/home/jovyan/PGOT/data/pgot_val.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-/home/jovyan/PGOT/outputs/object_bottleneck_oracles_v2_$(basename "${MODEL_PATH}")}"

GPU0="${GPU0:-0}"
GPU1="${GPU1:-1}"
NUM_GPUS="${NUM_GPUS:-2}"
RUN_SET="${RUN_SET:-gate}"
TRAIN_STEPS="${TRAIN_STEPS:-5000}"
DISTILL_STEPS="${DISTILL_STEPS:-2000}"
BATCH_SIZE="${BATCH_SIZE:-1}"
GRAD_ACCUM="${GRAD_ACCUM:-4}"
MAX_SAMPLES="${MAX_SAMPLES:-}"
MAX_TRAIN_SAMPLES="${MAX_TRAIN_SAMPLES:-}"
DIT_LAST_N_BLOCKS="${DIT_LAST_N_BLOCKS:-0}"
ADAPTER_LR="${ADAPTER_LR:-1e-4}"
DIT_LR="${DIT_LR:-1e-6}"
DISTILL_LOSS_WEIGHT="${DISTILL_LOSS_WEIGHT:-1.0}"
DIFFUSION_LOSS_WEIGHT="${DIFFUSION_LOSS_WEIGHT:-0.1}"
CFG_DROP_RATE="${CFG_DROP_RATE:-0.1}"
DIFF_INFER_STEPS="${DIFF_INFER_STEPS:-25}"
GUIDANCE_SCALE="${GUIDANCE_SCALE:-2.5}"
CURRENT_RFID_MAX="${CURRENT_RFID_MAX:-35.0}"
DTYPE="${DTYPE:-fp32}"
NUM_WORKERS="${NUM_WORKERS:-2}"

case "${RUN_SET}" in
    control|gate|all) ;;
    *) echo "RUN_SET must be one of: control, gate, all" >&2; exit 2 ;;
esac

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
    --dit_last_n_blocks "${DIT_LAST_N_BLOCKS}"
    --adapter_lr "${ADAPTER_LR}"
    --dit_lr "${DIT_LR}"
    --distill_loss_weight "${DISTILL_LOSS_WEIGHT}"
    --diffusion_loss_weight "${DIFFUSION_LOSS_WEIGHT}"
    --cfg_drop_rate "${CFG_DROP_RATE}"
    --diffusion_inference_steps "${DIFF_INFER_STEPS}"
    --guidance_scale "${GUIDANCE_SCALE}"
)
if [ -n "${MAX_SAMPLES}" ]; then
    COMMON_ARGS+=(--max_samples "${MAX_SAMPLES}")
fi
if [ -n "${MAX_TRAIN_SAMPLES}" ]; then
    COMMON_ARGS+=(--max_train_samples "${MAX_TRAIN_SAMPLES}")
fi

run_current() {
    echo "[$(date -u +%FT%TZ)] start c_current control on GPU ${GPU0}"
    CUDA_VISIBLE_DEVICES="${GPU0}" "${PYTHON}" -m pgot.eval.train_object_bottleneck_oracle \
        "${COMMON_ARGS[@]}" \
        --oracle c_current \
        --train_steps 0 \
        --distill_steps 0 \
        --no_save_checkpoint \
        --output_dir "${OUTPUT_DIR}/c_current" \
        2>&1 | tee "${OUTPUT_DIR}/c_current.log"

    "${PYTHON}" - "${OUTPUT_DIR}/c_current/summary.json" "${CURRENT_RFID_MAX}" <<'PY'
import json
import sys

with open(sys.argv[1]) as handle:
    summary = json.load(handle)
rfid = summary.get("rFID")
limit = float(sys.argv[2])
if rfid is None or not float(rfid) <= limit:
    raise SystemExit(f"C_current gate failed: rFID={rfid}, required <= {limit}")
print(f"C_current gate passed: rFID={float(rfid):.4f} <= {limit:.4f}")
PY
}

run_trainable() {
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
        --train_steps "${TRAIN_STEPS}" \
        --distill_steps "${DISTILL_STEPS}" \
        --output_dir "${OUTPUT_DIR}/${tag}" \
        2>&1 | tee "${OUTPUT_DIR}/${tag}.log"
}

summarize() {
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
    "recon_ssim", "recon_mse", "recon_mae", "train_steps", "distill_steps",
    "last_train_loss", "last_distill_loss", "last_diffusion_loss",
    "dit_last_n_blocks",
]
with open(root / "summary.csv", "w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=fields)
    writer.writeheader()
    for row in rows:
        writer.writerow({key: row.get(key, "") for key in fields})

for row in rows:
    suffix = f" K={row['object_tokens']}" if row.get("object_tokens") else ""
    rfid = row.get("rFID")
    loss = row.get("last_train_loss")
    rfid_text = "n/a" if rfid is None else f"{float(rfid):.4f}"
    loss_text = "n/a" if loss is None else f"{float(loss):.4f}"
    print(f"{row['oracle']}{suffix}: rFID={rfid_text}, loss={loss_text}")
PY
}

echo "===== PGOT Object Bottleneck Oracle v2 ====="
echo "Model:             ${MODEL_PATH}"
echo "Output:            ${OUTPUT_DIR}"
echo "Run set:           ${RUN_SET}"
echo "Optimizer steps:   ${TRAIN_STEPS}"
echo "Distill-only:      ${DISTILL_STEPS}"
echo "Gradient accum:    ${GRAD_ACCUM}"
echo "Trainable DiT:     last ${DIT_LAST_N_BLOCKS} blocks"
echo "C_current rFID <=  ${CURRENT_RFID_MAX}"
echo "============================================"

run_current
if [ "${RUN_SET}" = "control" ]; then
    summarize
    exit 0
fi

if [ "${RUN_SET}" = "gate" ]; then
    run_trainable "${GPU0}" c_full 4
else
    if [ "${NUM_GPUS}" = "1" ]; then
        run_trainable "${GPU0}" c_full 4
        run_trainable "${GPU0}" c_gtobj 2
        run_trainable "${GPU0}" c_gtobj 4
        run_trainable "${GPU0}" c_gtobj 8
        run_trainable "${GPU0}" c_ovt 4
    else
        (
            run_trainable "${GPU0}" c_full 4
            run_trainable "${GPU0}" c_gtobj 4
            run_trainable "${GPU0}" c_ovt 4
        ) &
        PID0=$!
        (
            run_trainable "${GPU1}" c_gtobj 2
            run_trainable "${GPU1}" c_gtobj 8
        ) &
        PID1=$!
        wait "${PID0}"
        wait "${PID1}"
    fi
fi

summarize
echo "Completed: ${OUTPUT_DIR}/summary.csv"
