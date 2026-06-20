#!/usr/bin/env bash
set -euo pipefail

# Re-evaluate PGOT versions against CODA-style COCO instance masks.
#
# This script forces the image preprocessing used by the model to match the
# CODA mask cache frame: resize the shorter side, then center-crop a square.
# It evaluates the full VAL_JSONL, currently pgot_val.jsonl has 2338 Pix2Coco
# samples. The CODA mask cache contains 5000 COCO val masks, but PGOT can only
# evaluate images that have Pix2Coco captions/OVTs in VAL_JSONL.
# It defaults to segmentation metrics only. Set COMPUTE_RFID=1 for slow rFID.

cd /home/jovyan/PGOT

PYTHON_BIN="${PYTHON_BIN:-/home/jovyan/.conda/envs/scale_rae/bin/python}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/home/jovyan/PGOT/outputs/eval_coda_aligned_versions}"
VAL_JSONL="${VAL_JSONL:-/home/jovyan/PGOT/data/pgot_val.jsonl}"
BATCH_SIZE="${BATCH_SIZE:-4}"
NUM_WORKERS="${NUM_WORKERS:-4}"
COMPUTE_RFID="${COMPUTE_RFID:-0}"
GUIDANCE_SCALE="${GUIDANCE_SCALE:-1.0}"
DIFFUSION_INFERENCE_STEPS="${DIFFUSION_INFERENCE_STEPS:-10}"
DTYPE="${DTYPE:-fp32}"
VERSIONS="${VERSIONS:-}"
WRITE_SUMMARY="${WRITE_SUMMARY:-1}"

mkdir -p "${OUTPUT_ROOT}"

COMMON_ARGS=(
  --val_jsonl "${VAL_JSONL}"
  --batch_size "${BATCH_SIZE}"
  --num_workers "${NUM_WORKERS}"
  --gt_source coco_instance
  --image_preprocess_mode coda_center_crop
  --coda_crop_size 512
  --dtype "${DTYPE}"
)

if [[ "${COMPUTE_RFID}" == "1" ]]; then
  COMMON_ARGS+=(
    --compute_rfid
    --guidance_scale "${GUIDANCE_SCALE}"
    --diffusion_inference_steps "${DIFFUSION_INFERENCE_STEPS}"
  )
fi

run_one() {
  local name="$1"
  local ckpt="$2"
  local readout="$3"
  local merge="$4"
  local out="${OUTPUT_ROOT}/${name}"

  if [[ -n "${VERSIONS}" ]]; then
    local selected=0
    IFS=',' read -ra requested_versions <<< "${VERSIONS}"
    for requested in "${requested_versions[@]}"; do
      if [[ "${name}" == "${requested}" ]]; then
        selected=1
        break
      fi
    done
    if [[ "${selected}" == "0" ]]; then
      echo "[SKIP] ${name}: not selected by VERSIONS=${VERSIONS}"
      return 0
    fi
  fi

  if [[ ! -d "${ckpt}" ]]; then
    echo "[SKIP] ${name}: checkpoint not found: ${ckpt}" | tee -a "${OUTPUT_ROOT}/eval.log"
    return 0
  fi

  echo
  echo "================================================================"
  echo "[EVAL] ${name}"
  echo "       ckpt=${ckpt}"
  echo "       readout=${readout}, merge=${merge}"
  echo "       output=${out}"
  echo "================================================================"

  PYTHONNOUSERSITE=1 \
  LD_LIBRARY_PATH="/home/jovyan/.conda/envs/scale_rae/lib:${LD_LIBRARY_PATH:-}" \
  "${PYTHON_BIN}" -m pgot.eval.run_eval \
    --model_path "${ckpt}" \
    --output_dir "${out}" \
    --readout "${readout}" \
    --eval_merge "${merge}" \
    "${COMMON_ARGS[@]}" \
    2>&1 | tee "${out}.log"
}

# name | checkpoint | train-matched/readable eval readout | OVT merge
run_one "V3_threshold" \
  "/home/jovyan/PGOT/checkpoints/pgot_main_v3/checkpoint-14000" \
  "threshold" "mean"

run_one "V4_competition" \
  "/home/jovyan/PGOT/checkpoints/pgot_main_v4/checkpoint-10000" \
  "competition" "max"

run_one "V5_competition" \
  "/home/jovyan/PGOT/checkpoints/pgot_main_v5/checkpoint-10000" \
  "competition" "max"

run_one "V6_competition" \
  "/home/jovyan/PGOT/checkpoints/pgot_main_v6/checkpoint-10000" \
  "competition" "max"

run_one "V7_nullbg" \
  "/home/jovyan/PGOT/checkpoints/pgot_main_v7_nullbg/checkpoint-10000" \
  "nullbg" "max"

run_one "V8_spatial" \
  "/home/jovyan/PGOT/checkpoints/pgot_main_v8/checkpoint-10000" \
  "spatial" "mean"

run_one "V8_1_spatial" \
  "/home/jovyan/PGOT/checkpoints/pgot_main_v8_1_llm_qk_outside/checkpoint-6000" \
  "spatial" "mean"

run_one "V8_2_spatial_trainmatch" \
  "/home/jovyan/PGOT/checkpoints/pgot_main_v8_2_spatial_outside_log/checkpoint-10000" \
  "spatial_trainmatch" "mean"

run_one "V8_2_R_spatial_trainmatch" \
  "/home/jovyan/PGOT/checkpoints/pgot_main_v8_2_noreg/checkpoint-10000" \
  "spatial_trainmatch" "mean"

run_one "V8_3_threshold" \
  "/home/jovyan/PGOT/checkpoints/pgot_main_v8_3_bce1_outlog1/checkpoint-10000" \
  "threshold" "mean"

run_one "V8_4_llm_attention" \
  "/home/jovyan/PGOT/checkpoints/pgot_main_v8_4_void_internal_outside/checkpoint-10000" \
  "llm_attention" "mean"

run_one "V8_5_llm_attention" \
  "/home/jovyan/PGOT/checkpoints/pgot_main_v8_5_llm_patch_outside/checkpoint-10000" \
  "llm_attention" "mean"

run_one "V9_threshold" \
  "/home/jovyan/PGOT/checkpoints/pgot_main_v9_bce_spatial_out03/checkpoint-10000" \
  "threshold" "mean"

run_one "V10_threshold" \
  "/home/jovyan/PGOT/checkpoints/pgot_main_v10_bce_ce01/checkpoint-10000" \
  "threshold" "mean"

run_one "V11_threshold" \
  "/home/jovyan/PGOT/checkpoints/pgot_main_v11_bal_bce_ce03_mean/checkpoint-10000" \
  "threshold" "mean"

run_one "V12_ovt_owner" \
  "/home/jovyan/PGOT/checkpoints/pgot_main_v12_ovt_update/checkpoint-10000" \
  "ovt_owner" "mean"

if [[ "${WRITE_SUMMARY}" == "1" ]]; then
"${PYTHON_BIN}" - <<'PY'
import csv
import json
import os
from pathlib import Path

root = Path(os.environ.get("OUTPUT_ROOT", "/home/jovyan/PGOT/outputs/eval_coda_aligned_versions"))
rows = []
for path in sorted(root.glob("*/summary.json")):
    data = json.loads(path.read_text())
    rows.append({
        "run": path.parent.name,
        "readout": data.get("readout"),
        "fARI": data.get("fARI"),
        "mBO": data.get("mBO"),
        "mIoU": data.get("mIoU"),
        "rFID": data.get("rFID", ""),
        "num_samples": data.get("num_samples"),
        "image_preprocess_mode": data.get("image_preprocess_mode"),
        "overlap_excluded": data.get("coda_overlap_excluded"),
        "ckpt": data.get("ckpt"),
    })

csv_path = root / "summary_table.csv"
with csv_path.open("w", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "run", "readout", "fARI", "mBO", "mIoU", "rFID",
            "num_samples", "image_preprocess_mode", "overlap_excluded", "ckpt",
        ],
    )
    writer.writeheader()
    writer.writerows(rows)

print(f"[DONE] Wrote {csv_path}")
for row in rows:
    print(
        f"{row['run']:28s} "
        f"fARI={row['fARI']:.4f} "
        f"mBO={row['mBO']:.4f} "
        f"mIoU={row['mIoU']:.4f} "
        f"rFID={row['rFID']}"
    )
PY
fi
