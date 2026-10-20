#!/usr/bin/env bash
# Two-GPU one-step train, checkpoint reload, and two-image COCO AR eval.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-/home/jovyan/.conda/envs/scale_rae/bin/python}"
SMOKE_ROOT="$(mktemp -d "${PROJECT_ROOT}/.smoke_fixedquery_k16.XXXXXX")"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export PYTHONNOUSERSITE=1
export WANDB_MODE="${WANDB_MODE:-offline}"
export WANDB_DIR="${SMOKE_ROOT}/wandb"
export WANDB_DATA_DIR="${SMOKE_ROOT}/wandb_data"
export WANDB_CACHE_DIR="${SMOKE_ROOT}/wandb_cache"
echo "Smoke workspace: ${SMOKE_ROOT}"

on_exit() {
    status=$?
    if (( status )); then
        echo "Fixed-query smoke failed; artifacts preserved at ${SMOKE_ROOT}" >&2
    fi
}
trap on_exit EXIT

OUTPUT_DIR="${SMOKE_ROOT}/train" WANDB_NAME=smoke_fixedquery_k16 \
NUM_GPUS=2 PER_DEVICE_TRAIN_BATCH_SIZE=2 GLOBAL_BATCH_SIZE=24 \
MAX_STEPS=1 SAVE_STEPS=1 SAVE_TOTAL_LIMIT=1 SAVE_ONLY_MODEL=True \
EVAL_STEPS=1 LOGGING_STEPS=1 PER_DEVICE_EVAL_BATCH_SIZE=1 \
EVAL_NUM_IMAGES=2 PGOT_EVAL_LOG_RECON_IMAGES=0 DATALOADER_NUM_WORKERS=1 \
REPORT_TO=none PGOT_SKIP_FINAL_SAVE=1 MASTER_PORT="${MASTER_PORT:-29761}" \
    bash "${PROJECT_ROOT}/scripts/train_pgot_fixedquery_k16_memory8_register32.sh" \
    > "${SMOKE_ROOT}/train.log" 2>&1

"${PYTHON}" "${PROJECT_ROOT}/scripts/verify_fixedquery_checkpoint.py" \
    "${SMOKE_ROOT}/train/checkpoint-1" --count 16

MODEL_PATH="${SMOKE_ROOT}/train/checkpoint-1" \
OUTPUT_ROOT="${SMOKE_ROOT}/eval" GPU=0 STAGE=coco SMOKE=1 \
DTYPE=fp32 NUM_WORKERS=1 BATCH_SIZE=1 COMPUTE_RFID=False COMPUTE_KID=False \
AR_MAX_NEW_TOKENS=128 \
    bash "${PROJECT_ROOT}/scripts/eval_pgot_fixedquery_k16.sh" \
    > "${SMOKE_ROOT}/eval.log" 2>&1

"${PYTHON}" - "${SMOKE_ROOT}" <<'PY'
import json
import shutil
import sys
from pathlib import Path

root = Path(sys.argv[1])
summary = root / "eval" / "coco_ar" / "summary.json"
data = json.loads(summary.read_text())
assert data["num_samples"] == 2, data.get("num_samples")
assert data["caption_mode"] == "autoregressive", data.get("caption_mode")
assert data["one_shot_memory_enabled"] is True
print(f"PASS: checkpoint reload and COCO AR eval: {summary}")
assert root.name.startswith(".smoke_fixedquery_k16.")
shutil.rmtree(root)
assert not root.exists()
PY
trap - EXIT
echo "PASS: FixedQuery K=16 matched-baseline smoke completed; artifacts removed."
