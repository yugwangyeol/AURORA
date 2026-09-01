#!/usr/bin/env bash
# Two-GPU FP32 smoke: train, in-train eval, save, W&B, and standalone eval.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SMOKE_ROOT="$(mktemp -d "${PROJECT_ROOT}/.smoke_e11_capacity_dit16.XXXXXX")"
TRAIN_OUTPUT_DIR="${SMOKE_ROOT}/checkpoint"
EVAL_OUTPUT_DIR="${SMOKE_ROOT}/eval"
SMOKE_CHECKPOINT="${TRAIN_OUTPUT_DIR}/checkpoint-1"
BASE_MODEL="${MODEL_PATH:-${PROJECT_ROOT}/checkpoints/pgot_e11_capacity/checkpoint-10000}"
TRAIN_JSONL="${TRAIN_JSONL:-${PROJECT_ROOT}/data/pgot_pix2cap_thing_train.jsonl}"
VAL_JSONL="${VAL_JSONL:-${PROJECT_ROOT}/data/pgot_pix2cap_generated_val5k.jsonl}"
PYTHON="${PYTHON:-/home/jovyan/.conda/envs/scale_rae/bin/python}"

case "${SMOKE_ROOT}" in
    "${PROJECT_ROOT}/.smoke_e11_capacity_dit16."*) ;;
    *) echo "Refusing unsafe smoke root: ${SMOKE_ROOT}" >&2; exit 1 ;;
esac
cleanup_smoke() {
    if [[ -d "${SMOKE_ROOT}" ]]; then
        find "${SMOKE_ROOT}" -depth -delete
    fi
}
preserve_failed_smoke() {
    local status=$?
    if (( status != 0 )); then
        echo "E11 Capacity DiT-16 smoke failed; outputs preserved at ${SMOKE_ROOT}" >&2
    fi
    return "${status}"
}
trap preserve_failed_smoke EXIT
mkdir -p "${TRAIN_OUTPUT_DIR}" "${EVAL_OUTPUT_DIR}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export WANDB_MODE=offline
export WANDB_PROJECT=PGOT
export WANDB_NAME=pgot_smoke_e11_capacity_dit16
export WANDB_DIR="${TRAIN_OUTPUT_DIR}/wandb"
export SMOKE_BATCH_SIZE="${SMOKE_BATCH_SIZE:-6}"
export SMOKE_GLOBAL_BATCH_SIZE="${SMOKE_GLOBAL_BATCH_SIZE:-24}"

MODEL_PATH="${BASE_MODEL}" OUTPUT_DIR="${TRAIN_OUTPUT_DIR}" \
NUM_GPUS=2 PER_DEVICE_TRAIN_BATCH_SIZE="${SMOKE_BATCH_SIZE}" GLOBAL_BATCH_SIZE="${SMOKE_GLOBAL_BATCH_SIZE}" \
MAX_STEPS=1 SAVE_STEPS=1 SAVE_TOTAL_LIMIT=1 EVAL_STEPS=1 LOGGING_STEPS=1 \
PER_DEVICE_EVAL_BATCH_SIZE=1 EVAL_NUM_IMAGES=1 PGOT_EVAL_LOG_RECON_IMAGES=1 \
DATALOADER_NUM_WORKERS=1 TRAIN_JSONL="${TRAIN_JSONL}" VAL_JSONL="${VAL_JSONL}" \
REPORT_TO=wandb bash "${PROJECT_ROOT}/scripts/train_pgot_e11_capacity_dit16.sh" \
    2>&1 | tee "${TRAIN_OUTPUT_DIR}/smoke_train.log"

test -f "${SMOKE_CHECKPOINT}/config.json"
test -f "${SMOKE_CHECKPOINT}/trainer_state.json"
test -f "${SMOKE_CHECKPOINT}/training_args.bin"
grep -q '\[PGOT/Freeze\] DiT last 16 blocks unfrozen (idx 16..31 of 32)' "${TRAIN_OUTPUT_DIR}/smoke_train.log"
grep -q 'loss_e8_owner' "${TRAIN_OUTPUT_DIR}/smoke_train.log"
grep -q 'loss_e8_reader' "${TRAIN_OUTPUT_DIR}/smoke_train.log"
grep -q 'loss_recon' "${TRAIN_OUTPUT_DIR}/smoke_train.log"
grep -q 'eval_loss' "${TRAIN_OUTPUT_DIR}/smoke_train.log"
grep -q 'train_runtime' "${TRAIN_OUTPUT_DIR}/smoke_train.log"
grep -q 'E11 heterogeneous memories: object=8; register=16; query_separation=False' "${TRAIN_OUTPUT_DIR}/smoke_train.log"

"${PYTHON}" - "${SMOKE_CHECKPOINT}" <<'PY'
import json
import os
import sys
import torch

root = sys.argv[1]
with open(os.path.join(root, "config.json")) as handle:
    config = json.load(handle)
assert config["pgot_e11_object_memories_per_owner"] == 8
assert config["pgot_e11_register_memories_per_owner"] == 16
assert config["pgot_e11_query_separation_enable"] is False
assert config["pgot_e8_causal_enable"] is False
args = torch.load(os.path.join(root, "training_args.bin"), map_location="cpu", weights_only=False)
assert args.bf16 is False and args.fp16 is False and args.tf32 is False
assert args.per_device_train_batch_size == int(os.environ["SMOKE_BATCH_SIZE"])
assert args.gradient_accumulation_steps * args.per_device_train_batch_size * 2 == int(
    os.environ["SMOKE_GLOBAL_BATCH_SIZE"]
)
print("E11 Capacity DiT-16 checkpoint configuration verified")
PY

WANDB_RUN_DIR="$(find "${TRAIN_OUTPUT_DIR}/wandb" -maxdepth 3 -type d -name 'offline-run-*' | head -n 1)"
test -n "${WANDB_RUN_DIR}"
test -n "$(find "${WANDB_RUN_DIR}" -maxdepth 1 -type f -name 'run-*.wandb' | head -n 1)"
WANDB_TABLE="$(find "${WANDB_RUN_DIR}/files/media/table" -type f -name '*.table.json' | head -n 1)"
test -n "${WANDB_TABLE}"
grep -q 'ovt_owner' "${WANDB_TABLE}"
grep -q 'our_recon' "${WANDB_TABLE}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
MODEL_PATH="${SMOKE_CHECKPOINT}" VAL_JSONL="${VAL_JSONL}" \
OUTPUT_DIR="${EVAL_OUTPUT_DIR}" MAX_SAMPLES=2 BATCH_SIZE=1 NUM_WORKERS=1 \
DTYPE=fp32 DIFFUSION_INFERENCE_STEPS=2 COMPUTE_RFID=True \
bash "${PROJECT_ROOT}/scripts/eval_pgot_e11_capacity_dit16.sh" \
    2>&1 | tee "${EVAL_OUTPUT_DIR}/smoke_eval.log"

test -f "${EVAL_OUTPUT_DIR}/summary.json"
grep -q '"e11_object_memories_per_owner": 8' "${EVAL_OUTPUT_DIR}/summary.json"
grep -q '"e11_register_memories_per_owner": 16' "${EVAL_OUTPUT_DIR}/summary.json"
grep -q '"background_visual_memories": 64' "${EVAL_OUTPUT_DIR}/summary.json"
grep -q '"readout": "ovt_owner"' "${EVAL_OUTPUT_DIR}/summary.json"
grep -q '"recon_mse"' "${EVAL_OUTPUT_DIR}/summary.json"

cleanup_smoke
test ! -e "${SMOKE_ROOT}"
trap - EXIT
echo "PGOT E11 Capacity DiT-16 smoke passed; smoke outputs were removed."
