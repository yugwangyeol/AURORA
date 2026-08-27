#!/usr/bin/env bash
# Two-GPU FP32 E10-R train/in-train-eval/save/offline-W&B/standalone-eval smoke.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SMOKE_ROOT="$(mktemp -d "${PROJECT_ROOT}/.smoke_e10_r.XXXXXX")"
OUTPUT_DIR="${SMOKE_ROOT}/checkpoint"
EVAL_OUTPUT_DIR="${SMOKE_ROOT}/eval"
BASE_MODEL="${MODEL_PATH:-${PROJECT_ROOT}/checkpoints/pgot_e8_2_paired_causal/checkpoint-10000}"
TRAIN_JSONL="${TRAIN_JSONL:-${PROJECT_ROOT}/data/pgot_pix2cap_thing_train.jsonl}"
VAL_JSONL="${VAL_JSONL:-${PROJECT_ROOT}/data/pgot_pix2cap_generated_val5k.jsonl}"
PYTHON="${PYTHON:-/home/jovyan/.conda/envs/scale_rae/bin/python}"

case "${SMOKE_ROOT}" in
    "${PROJECT_ROOT}/.smoke_e10_r."*) ;;
    *) echo "Refusing unsafe smoke root: ${SMOKE_ROOT}" >&2; exit 1 ;;
esac
cleanup_smoke() {
    if [[ -d "${SMOKE_ROOT}" ]]; then
        rm -rf -- "${SMOKE_ROOT}"
    fi
}
preserve_failed_smoke() {
    local status=$?
    if (( status != 0 )); then
        echo "E10-R smoke failed; outputs preserved at ${SMOKE_ROOT}" >&2
    fi
    return "${status}"
}
trap preserve_failed_smoke EXIT
mkdir -p "${OUTPUT_DIR}" "${EVAL_OUTPUT_DIR}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export WANDB_MODE=offline
export WANDB_PROJECT=PGOT
export WANDB_NAME=pgot_smoke_e10_r_raw_value
export WANDB_DIR="${OUTPUT_DIR}/wandb"

MODEL_PATH="${BASE_MODEL}" OUTPUT_DIR="${OUTPUT_DIR}" \
NUM_GPUS=2 PER_DEVICE_TRAIN_BATCH_SIZE=1 GLOBAL_BATCH_SIZE=2 \
PER_DEVICE_EVAL_BATCH_SIZE=1 MAX_STEPS=1 SAVE_STEPS=1 EVAL_STEPS=1 \
LOGGING_STEPS=1 EVAL_NUM_IMAGES=1 PGOT_EVAL_LOG_RECON_IMAGES=1 \
DATALOADER_NUM_WORKERS=1 TRAIN_JSONL="${TRAIN_JSONL}" VAL_JSONL="${VAL_JSONL}" \
REPORT_TO=wandb bash "${PROJECT_ROOT}/scripts/train_pgot_e10_r_raw_value.sh" \
    2>&1 | tee "${OUTPUT_DIR}/smoke_train.log"

test -f "${OUTPUT_DIR}/checkpoint-1/config.json"
test -f "${OUTPUT_DIR}/checkpoint-1/trainer_state.json"
grep -q '"pgot_e10_raw_value_enable": true' "${OUTPUT_DIR}/checkpoint-1/config.json"
grep -q '"pgot_e8_update_mode": "separate_memory"' "${OUTPUT_DIR}/checkpoint-1/config.json"
grep -q '"pgot_e8_reader_supervision_mode": "writer"' "${OUTPUT_DIR}/checkpoint-1/config.json"
grep -q '"pgot_e8_causal_enable": false' "${OUTPUT_DIR}/checkpoint-1/config.json"
grep -q 'e10_raw_value_enabled' "${OUTPUT_DIR}/smoke_train.log"
grep -q 'loss_e8_owner' "${OUTPUT_DIR}/smoke_train.log"
grep -q 'loss_e8_reader' "${OUTPUT_DIR}/smoke_train.log"
grep -q 'loss_recon' "${OUTPUT_DIR}/smoke_train.log"
grep -q 'eval_loss' "${OUTPUT_DIR}/smoke_train.log"

"${PYTHON}" - "${OUTPUT_DIR}/checkpoint-1" <<'PY'
import json
import os
import sys
from safetensors import safe_open

root = sys.argv[1]
index = os.path.join(root, "model.safetensors.index.json")
if os.path.exists(index):
    with open(index) as handle:
        keys = set(json.load(handle)["weight_map"])
else:
    with safe_open(
        os.path.join(root, "model.safetensors"), framework="pt", device="cpu"
    ) as handle:
        keys = set(handle.keys())
assert any("pgot_e8_writer.raw_value.weight" in key for key in keys)
assert any("pgot_e8_writer.raw_value_norm.weight" in key for key in keys)
assert any("pgot_e8_reader.value" in key for key in keys)
print("E10-R raw-value Writer and typed Reader tensors verified")
PY

WANDB_RUN_DIR="$(find "${OUTPUT_DIR}/wandb" -maxdepth 3 -type d -name 'offline-run-*' | head -n 1)"
test -n "${WANDB_RUN_DIR}"
test -n "$(find "${WANDB_RUN_DIR}" -maxdepth 1 -type f -name 'run-*.wandb' | head -n 1)"
WANDB_TABLE="$(find "${WANDB_RUN_DIR}/files/media/table" -type f -name '*.table.json' | head -n 1)"
test -n "${WANDB_TABLE}"
grep -q 'ovt_owner' "${WANDB_TABLE}"
grep -q 'our_recon' "${WANDB_TABLE}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
MODEL_PATH="${OUTPUT_DIR}/checkpoint-1" VAL_JSONL="${VAL_JSONL}" \
OUTPUT_DIR="${EVAL_OUTPUT_DIR}" MAX_SAMPLES=2 BATCH_SIZE=1 NUM_WORKERS=1 \
DTYPE=fp32 DIFFUSION_INFERENCE_STEPS=2 COMPUTE_RFID=True \
bash "${PROJECT_ROOT}/scripts/eval_pgot_e10_r_raw_value.sh" \
    2>&1 | tee "${EVAL_OUTPUT_DIR}/smoke_eval.log"

test -f "${EVAL_OUTPUT_DIR}/summary.json"
grep -q '"e10_raw_value_enabled": true' "${EVAL_OUTPUT_DIR}/summary.json"
grep -q '"visual_memory_value_source": "frozen source SigLIP pre-projector patches"' "${EVAL_OUTPUT_DIR}/summary.json"
grep -q '"e8_causal_training_enabled": false' "${EVAL_OUTPUT_DIR}/summary.json"
grep -q '"readout": "ovt_owner"' "${EVAL_OUTPUT_DIR}/summary.json"
grep -q '"recon_mse"' "${EVAL_OUTPUT_DIR}/summary.json"

cleanup_smoke
test ! -e "${SMOKE_ROOT}"
trap - EXIT
echo "PGOT E10-R smoke passed; smoke outputs were removed."
