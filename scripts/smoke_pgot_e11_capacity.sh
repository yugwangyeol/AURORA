#!/usr/bin/env bash
# Two-GPU FP32 smoke for either E11 capacity variant.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VARIANT="${E11_SMOKE_VARIANT:-capacity}"
case "${VARIANT}" in
    capacity)
        TRAIN_SCRIPT="${PROJECT_ROOT}/scripts/train_pgot_e11_capacity.sh"
        EVAL_SCRIPT="${PROJECT_ROOT}/scripts/eval_pgot_e11_capacity.sh"
        EXPECTED_QUERY=false
        ;;
    capacity_query_separation)
        TRAIN_SCRIPT="${PROJECT_ROOT}/scripts/train_pgot_e11_capacity_query_separation.sh"
        EVAL_SCRIPT="${PROJECT_ROOT}/scripts/eval_pgot_e11_capacity_query_separation.sh"
        EXPECTED_QUERY=true
        ;;
    *) echo "Unknown E11 smoke variant: ${VARIANT}" >&2; exit 2 ;;
esac

SMOKE_ROOT="$(mktemp -d "${PROJECT_ROOT}/.smoke_e11_${VARIANT}.XXXXXX")"
OUTPUT_DIR="${SMOKE_ROOT}/checkpoint"
EVAL_OUTPUT_DIR="${SMOKE_ROOT}/eval"
SMOKE_CHECKPOINT="${OUTPUT_DIR}/checkpoint-1"
BASE_MODEL="${MODEL_PATH:-${PROJECT_ROOT}/checkpoints/pgot_e11_dual_m4/checkpoint-10000}"
TRAIN_JSONL="${TRAIN_JSONL:-${PROJECT_ROOT}/data/pgot_pix2cap_thing_train.jsonl}"
VAL_JSONL="${VAL_JSONL:-${PROJECT_ROOT}/data/pgot_pix2cap_generated_val5k.jsonl}"
PYTHON="${PYTHON:-/home/jovyan/.conda/envs/scale_rae/bin/python}"

case "${SMOKE_ROOT}" in
    "${PROJECT_ROOT}/.smoke_e11_${VARIANT}."*) ;;
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
        echo "E11 ${VARIANT} smoke failed; outputs preserved at ${SMOKE_ROOT}" >&2
    fi
    return "${status}"
}
trap preserve_failed_smoke EXIT
mkdir -p "${OUTPUT_DIR}" "${EVAL_OUTPUT_DIR}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export WANDB_MODE=offline
export WANDB_PROJECT=PGOT
export WANDB_NAME="pgot_smoke_e11_${VARIANT}"
export WANDB_DIR="${OUTPUT_DIR}/wandb"

MODEL_PATH="${BASE_MODEL}" OUTPUT_DIR="${OUTPUT_DIR}" \
NUM_GPUS=2 PER_DEVICE_TRAIN_BATCH_SIZE=1 GLOBAL_BATCH_SIZE=2 \
PER_DEVICE_EVAL_BATCH_SIZE=1 MAX_STEPS=1 SAVE_STEPS=1 EVAL_STEPS=1 \
LOGGING_STEPS=1 EVAL_NUM_IMAGES=1 PGOT_EVAL_LOG_RECON_IMAGES=1 \
DATALOADER_NUM_WORKERS=1 TRAIN_JSONL="${TRAIN_JSONL}" VAL_JSONL="${VAL_JSONL}" \
REPORT_TO=wandb bash "${TRAIN_SCRIPT}" \
    2>&1 | tee "${OUTPUT_DIR}/smoke_train.log"

test -f "${SMOKE_CHECKPOINT}/config.json"
test -f "${SMOKE_CHECKPOINT}/trainer_state.json"
grep -q '"pgot_e11_dual_m4_enable": true' "${SMOKE_CHECKPOINT}/config.json"
grep -q '"pgot_e11_object_memories_per_owner": 8' "${SMOKE_CHECKPOINT}/config.json"
grep -q '"pgot_e11_register_memories_per_owner": 16' "${SMOKE_CHECKPOINT}/config.json"
grep -q "\"pgot_e11_query_separation_enable\": ${EXPECTED_QUERY}" "${SMOKE_CHECKPOINT}/config.json"
grep -q '"pgot_n_register": 4' "${SMOKE_CHECKPOINT}/config.json"
grep -q '"pgot_e10_raw_value_enable": true' "${SMOKE_CHECKPOINT}/config.json"
grep -q '"pgot_e8_reader_supervision_mode": "writer"' "${SMOKE_CHECKPOINT}/config.json"
grep -q '"pgot_e8_causal_enable": false' "${SMOKE_CHECKPOINT}/config.json"
grep -q '\[PGOT/E11\] initialized expanded Writer/Reader memory IDs' "${OUTPUT_DIR}/smoke_train.log"
grep -q 'e11_object_memories_per_owner' "${OUTPUT_DIR}/smoke_train.log"
grep -q 'e11_register_memories_per_owner' "${OUTPUT_DIR}/smoke_train.log"
grep -q 'e11_query_separation_enabled' "${OUTPUT_DIR}/smoke_train.log"
grep -q 'loss_e8_owner' "${OUTPUT_DIR}/smoke_train.log"
grep -q 'loss_e8_reader' "${OUTPUT_DIR}/smoke_train.log"
grep -q 'loss_recon' "${OUTPUT_DIR}/smoke_train.log"
grep -q 'eval_loss' "${OUTPUT_DIR}/smoke_train.log"

"${PYTHON}" - "${SMOKE_CHECKPOINT}" <<'PY'
import json
import os
import sys
from safetensors import safe_open

root = sys.argv[1]
with open(os.path.join(root, "model.safetensors.index.json")) as handle:
    weight_map = json.load(handle)["weight_map"]
checks = {
    "pgot_e8_writer.memory_id_embeddings": (16, 1536),
    "pgot_e8_reader.memory_key_embeddings": (16, 1536),
}
for key, expected in checks.items():
    shard = weight_map[key]
    with safe_open(os.path.join(root, shard), framework="pt", device="cpu") as sf:
        tensor = sf.get_tensor(key)
    assert tuple(tensor.shape) == expected, (key, tensor.shape, expected)
    assert tensor.isfinite().all(), key
print("E11 heterogeneous memory tensors verified")
PY

WANDB_RUN_DIR="$(find "${OUTPUT_DIR}/wandb" -maxdepth 3 -type d -name 'offline-run-*' | head -n 1)"
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
bash "${EVAL_SCRIPT}" 2>&1 | tee "${EVAL_OUTPUT_DIR}/smoke_eval.log"

test -f "${EVAL_OUTPUT_DIR}/summary.json"
grep -q '"e11_object_memories_per_owner": 8' "${EVAL_OUTPUT_DIR}/summary.json"
grep -q '"e11_register_memories_per_owner": 16' "${EVAL_OUTPUT_DIR}/summary.json"
grep -q "\"e11_query_separation_enabled\": ${EXPECTED_QUERY}" "${EVAL_OUTPUT_DIR}/summary.json"
grep -q '"background_semantic_registers": 4' "${EVAL_OUTPUT_DIR}/summary.json"
grep -q '"background_visual_memories": 64' "${EVAL_OUTPUT_DIR}/summary.json"
grep -q '"readout": "ovt_owner"' "${EVAL_OUTPUT_DIR}/summary.json"
grep -q '"recon_mse"' "${EVAL_OUTPUT_DIR}/summary.json"

cleanup_smoke
test ! -e "${SMOKE_ROOT}"
trap - EXIT
echo "PGOT E11 ${VARIANT} smoke passed; smoke outputs were removed."
