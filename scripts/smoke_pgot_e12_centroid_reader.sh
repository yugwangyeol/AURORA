#!/usr/bin/env bash
# Two-GPU FP32 E12 train/eval/save/offline-W&B/standalone-eval smoke.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SMOKE_ROOT="$(mktemp -d "${PROJECT_ROOT}/.smoke_e12_centroid_reader.XXXXXX")"
OUTPUT_DIR="${SMOKE_ROOT}/checkpoint"
EVAL_OUTPUT_DIR="${SMOKE_ROOT}/eval"
SMOKE_CHECKPOINT="${OUTPUT_DIR}/checkpoint-2"
BASE_MODEL="${MODEL_PATH:-${PROJECT_ROOT}/checkpoints/pgot_e11_dual_m4/checkpoint-10000}"
TRAIN_JSONL="${TRAIN_JSONL:-${PROJECT_ROOT}/data/pgot_pix2cap_thing_train.jsonl}"
VAL_JSONL="${VAL_JSONL:-${PROJECT_ROOT}/data/pgot_pix2cap_generated_val5k.jsonl}"
PYTHON="${PYTHON:-/home/jovyan/.conda/envs/scale_rae/bin/python}"

case "${SMOKE_ROOT}" in
    "${PROJECT_ROOT}/.smoke_e12_centroid_reader."*) ;;
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
        echo "E12 smoke failed; outputs preserved at ${SMOKE_ROOT}" >&2
    fi
    return "${status}"
}
trap preserve_failed_smoke EXIT
mkdir -p "${OUTPUT_DIR}" "${EVAL_OUTPUT_DIR}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export WANDB_MODE=offline
export WANDB_PROJECT=PGOT
export WANDB_NAME=pgot_smoke_e12_centroid_reader
export WANDB_DIR="${OUTPUT_DIR}/wandb"

MODEL_PATH="${BASE_MODEL}" OUTPUT_DIR="${OUTPUT_DIR}" \
NUM_GPUS=2 PER_DEVICE_TRAIN_BATCH_SIZE=1 GLOBAL_BATCH_SIZE=2 \
PER_DEVICE_EVAL_BATCH_SIZE=1 MAX_STEPS=2 SAVE_STEPS=1 EVAL_STEPS=1 \
LOGGING_STEPS=1 EVAL_NUM_IMAGES=1 PGOT_EVAL_LOG_RECON_IMAGES=1 \
DATALOADER_NUM_WORKERS=1 TRAIN_JSONL="${TRAIN_JSONL}" VAL_JSONL="${VAL_JSONL}" \
REPORT_TO=wandb bash "${PROJECT_ROOT}/scripts/train_pgot_e12_centroid_reader.sh" \
    2>&1 | tee "${OUTPUT_DIR}/smoke_train.log"

test -f "${SMOKE_CHECKPOINT}/config.json"
test -f "${SMOKE_CHECKPOINT}/trainer_state.json"
grep -q '"pgot_e11_dual_m4_enable": true' "${SMOKE_CHECKPOINT}/config.json"
grep -q '"pgot_e11_memories_per_owner": 4' "${SMOKE_CHECKPOINT}/config.json"
grep -q '"pgot_e12_centroid_reader_enable": true' "${SMOKE_CHECKPOINT}/config.json"
grep -q '"pgot_e12_centroid_gate_init": 0.0' "${SMOKE_CHECKPOINT}/config.json"
grep -q '"pgot_e10_raw_value_enable": true' "${SMOKE_CHECKPOINT}/config.json"
grep -q '"pgot_e8_reader_supervision_mode": "writer"' "${SMOKE_CHECKPOINT}/config.json"
grep -q '"pgot_e8_causal_enable": false' "${SMOKE_CHECKPOINT}/config.json"
grep -q '\[PGOT/E12\] deterministically initialized centroid Reader at gate zero' "${OUTPUT_DIR}/smoke_train.log"
grep -q 'e12_centroid_reader_enabled' "${OUTPUT_DIR}/smoke_train.log"
grep -q 'e12_centroid_object_gate' "${OUTPUT_DIR}/smoke_train.log"
grep -q 'e12_centroid_register_gate' "${OUTPUT_DIR}/smoke_train.log"
grep -q 'e12_centroid_position_rms' "${OUTPUT_DIR}/smoke_train.log"
grep -q 'e12_centroid_mean_radius' "${OUTPUT_DIR}/smoke_train.log"
grep -q 'loss_e8_owner' "${OUTPUT_DIR}/smoke_train.log"
grep -q 'loss_e8_reader' "${OUTPUT_DIR}/smoke_train.log"
grep -q 'loss_recon' "${OUTPUT_DIR}/smoke_train.log"
grep -q 'eval_loss' "${OUTPUT_DIR}/smoke_train.log"

"${PYTHON}" - "${SMOKE_CHECKPOINT}" <<'PY'
import json
import math
import os
import sys

from safetensors import safe_open

root = sys.argv[1]
index = os.path.join(root, "model.safetensors.index.json")
if os.path.exists(index):
    with open(index) as handle:
        weight_map = json.load(handle)["weight_map"]
    keys = set(weight_map)
    shards = sorted(set(weight_map.values()))
else:
    weight_map = None
    shards = ["model.safetensors"]
    with safe_open(os.path.join(root, shards[0]), framework="pt", device="cpu") as handle:
        keys = set(handle.keys())

required_suffixes = (
    "pgot_e8_reader.centroid_position_projector.weight",
    "pgot_e8_reader.centroid_position_norm.weight",
    "pgot_e8_reader.centroid_object_gate",
    "pgot_e8_reader.centroid_register_gate",
)
resolved = {}
for suffix in required_suffixes:
    matches = [key for key in keys if key.endswith(suffix)]
    assert len(matches) == 1, (suffix, matches)
    resolved[suffix] = matches[0]

values = {}
for shard in shards:
    shard_path = os.path.join(root, shard)
    shard_keys = {
        key for key in resolved.values()
        if weight_map is None or weight_map[key] == shard
    }
    if not shard_keys:
        continue
    with safe_open(shard_path, framework="pt", device="cpu") as handle:
        for key in shard_keys:
            values[key] = handle.get_tensor(key)

object_gate = values[resolved["pgot_e8_reader.centroid_object_gate"]].item()
register_gate = values[resolved["pgot_e8_reader.centroid_register_gate"]].item()
assert math.isfinite(object_gate) and math.isfinite(register_gate)
assert abs(object_gate) + abs(register_gate) > 0.0, (object_gate, register_gate)
projector = values[resolved["pgot_e8_reader.centroid_position_projector.weight"]]
assert projector.isfinite().all() and projector.abs().sum().item() > 0.0
print(
    "E12 positional tensors verified; learned raw gates:",
    object_gate,
    register_gate,
)
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
bash "${PROJECT_ROOT}/scripts/eval_pgot_e12_centroid_reader.sh" \
    2>&1 | tee "${EVAL_OUTPUT_DIR}/smoke_eval.log"

test -f "${EVAL_OUTPUT_DIR}/summary.json"
grep -q '"e12_centroid_reader_enabled": true' "${EVAL_OUTPUT_DIR}/summary.json"
grep -q '"e11_dual_m4_enabled": true' "${EVAL_OUTPUT_DIR}/summary.json"
grep -q '"e11_memories_per_owner": 4' "${EVAL_OUTPUT_DIR}/summary.json"
grep -q '"background_semantic_registers": 4' "${EVAL_OUTPUT_DIR}/summary.json"
grep -q '"background_visual_memories": 16' "${EVAL_OUTPUT_DIR}/summary.json"
grep -q '"e12_centroid_object_gate"' "${EVAL_OUTPUT_DIR}/summary.json"
grep -q '"e12_centroid_register_gate"' "${EVAL_OUTPUT_DIR}/summary.json"
grep -q '"readout": "ovt_owner"' "${EVAL_OUTPUT_DIR}/summary.json"
grep -q '"recon_mse"' "${EVAL_OUTPUT_DIR}/summary.json"

cleanup_smoke
test ! -e "${SMOKE_ROOT}"
trap - EXIT
echo "PGOT E12 centroid Reader smoke passed; smoke outputs were removed."
