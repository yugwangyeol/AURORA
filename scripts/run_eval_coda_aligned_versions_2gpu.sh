#!/usr/bin/env bash
set -euo pipefail

# CODA-aligned re-evaluation split across two GPUs.
# This intentionally evaluates the full VAL_JSONL. For the current Pix2Coco
# PGOT val file that means 2338 samples, not all 5000 COCO val images.
# No max-sample shortcut is used.

cd /home/jovyan/PGOT

OUTPUT_ROOT="${OUTPUT_ROOT:-/home/jovyan/PGOT/outputs/eval_coda_aligned_versions}"
COMPUTE_RFID="${COMPUTE_RFID:-0}"
BATCH_SIZE="${BATCH_SIZE:-4}"
NUM_WORKERS="${NUM_WORKERS:-4}"

mkdir -p "${OUTPUT_ROOT}"

GPU0_VERSIONS="${GPU0_VERSIONS:-V3_threshold,V5_competition,V7_nullbg,V8_1_spatial,V8_2_R_spatial_trainmatch,V8_4_llm_attention,V9_threshold,V11_threshold}"
GPU1_VERSIONS="${GPU1_VERSIONS:-V4_competition,V6_competition,V8_spatial,V8_2_spatial_trainmatch,V8_3_threshold,V8_5_llm_attention,V10_threshold,V12_ovt_owner}"

echo "===== CODA-aligned full eval, 2 GPUs ====="
echo "Output:       ${OUTPUT_ROOT}"
echo "VAL_JSONL:    /home/jovyan/PGOT/data/pgot_val.jsonl ($(wc -l < /home/jovyan/PGOT/data/pgot_val.jsonl) samples)"
echo "GPU0 runs:    ${GPU0_VERSIONS}"
echo "GPU1 runs:    ${GPU1_VERSIONS}"
echo "rFID:         ${COMPUTE_RFID}"
echo "Batch size:   ${BATCH_SIZE}"
echo "Num workers:  ${NUM_WORKERS}"
echo "=========================================="

CUDA_VISIBLE_DEVICES=0 \
OUTPUT_ROOT="${OUTPUT_ROOT}" \
COMPUTE_RFID="${COMPUTE_RFID}" \
BATCH_SIZE="${BATCH_SIZE}" \
NUM_WORKERS="${NUM_WORKERS}" \
VERSIONS="${GPU0_VERSIONS}" \
WRITE_SUMMARY=0 \
bash scripts/run_eval_coda_aligned_versions.sh \
  2>&1 | tee "${OUTPUT_ROOT}/gpu0.log" &
pid0=$!

CUDA_VISIBLE_DEVICES=1 \
OUTPUT_ROOT="${OUTPUT_ROOT}" \
COMPUTE_RFID="${COMPUTE_RFID}" \
BATCH_SIZE="${BATCH_SIZE}" \
NUM_WORKERS="${NUM_WORKERS}" \
VERSIONS="${GPU1_VERSIONS}" \
WRITE_SUMMARY=0 \
bash scripts/run_eval_coda_aligned_versions.sh \
  2>&1 | tee "${OUTPUT_ROOT}/gpu1.log" &
pid1=$!

wait "${pid0}"
wait "${pid1}"

/home/jovyan/.conda/envs/scale_rae/bin/python - <<'PY'
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
