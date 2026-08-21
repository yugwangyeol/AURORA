#!/usr/bin/env bash
# Two-GPU FP32 E9.1 train/in-train-eval/save/W&B/standalone-eval smoke.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/checkpoints/pgot_smoke_e9_1_final_ovt}"
EVAL_OUTPUT_DIR="${EVAL_OUTPUT_DIR:-${PROJECT_ROOT}/outputs/smoke_pgot_e9_1_final_ovt_eval}"
BASE_MODEL="${MODEL_PATH:-${PROJECT_ROOT}/checkpoints/pgot_e9_unified_gru/checkpoint-10000}"
TRAIN_JSONL="${TRAIN_JSONL:-${PROJECT_ROOT}/data/pgot_pix2cap_thing_train.jsonl}"
VAL_JSONL="${VAL_JSONL:-${PROJECT_ROOT}/data/pgot_pix2cap_generated_val5k.jsonl}"
PYTHON="${PYTHON:-/home/jovyan/.conda/envs/scale_rae/bin/python}"

case "$(realpath -m "${OUTPUT_DIR}")" in
    "${PROJECT_ROOT}/checkpoints/"*smoke*) ;;
    *) echo "Refusing unsafe smoke checkpoint path: ${OUTPUT_DIR}" >&2; exit 1 ;;
esac
case "$(realpath -m "${EVAL_OUTPUT_DIR}")" in
    "${PROJECT_ROOT}/outputs/"*smoke*) ;;
    *) echo "Refusing unsafe smoke eval path: ${EVAL_OUTPUT_DIR}" >&2; exit 1 ;;
esac

cleanup_smoke() {
    local path
    for path in "${OUTPUT_DIR}" "${EVAL_OUTPUT_DIR}"; do
        if [[ -e "${path}" ]]; then
            rm -rf -- "${path}"
        fi
    done
}
preserve_failed_smoke() {
    local status=$?
    if (( status != 0 )); then
        echo "Smoke failed; outputs were preserved for diagnosis." >&2
    fi
    return "${status}"
}
trap preserve_failed_smoke EXIT
cleanup_smoke
mkdir -p "${OUTPUT_DIR}" "${EVAL_OUTPUT_DIR}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export WANDB_MODE=offline
export WANDB_PROJECT=PGOT
export WANDB_DIR="${OUTPUT_DIR}/wandb"

MODEL_PATH="${BASE_MODEL}" OUTPUT_DIR="${OUTPUT_DIR}" \
WANDB_NAME=pgot_smoke_e9_1_final_ovt \
NUM_GPUS=2 PER_DEVICE_TRAIN_BATCH_SIZE=1 GLOBAL_BATCH_SIZE=2 \
PER_DEVICE_EVAL_BATCH_SIZE=1 MAX_STEPS=1 SAVE_STEPS=1 EVAL_STEPS=1 \
LOGGING_STEPS=1 EVAL_NUM_IMAGES=1 PGOT_EVAL_LOG_RECON_IMAGES=1 \
DATALOADER_NUM_WORKERS=1 TRAIN_JSONL="${TRAIN_JSONL}" VAL_JSONL="${VAL_JSONL}" \
E8_CAUSAL_BATCH_PROBABILITY=1.0 E8_CAUSAL_RAMP_STEPS=1 REPORT_TO=wandb \
bash "${PROJECT_ROOT}/scripts/train_pgot_e9_1_final_ovt.sh" \
    2>&1 | tee "${OUTPUT_DIR}/smoke_train.log"

test -f "${OUTPUT_DIR}/checkpoint-1/config.json"
test -f "${OUTPUT_DIR}/checkpoint-1/trainer_state.json"
grep -q '"pgot_e8_update_mode": "final_ovt"' "${OUTPUT_DIR}/checkpoint-1/config.json"
grep -q '"pgot_e8_layers": "20,23,26"' "${OUTPUT_DIR}/checkpoint-1/config.json"
grep -q '"pgot_e8_reader_supervision_mode": "writer"' "${OUTPUT_DIR}/checkpoint-1/config.json"
grep -q '"pgot_e8_causal_enable": true' "${OUTPUT_DIR}/checkpoint-1/config.json"
grep -q '"pgot_e8_register_bg_weight": 0.005' "${OUTPUT_DIR}/checkpoint-1/config.json"
grep -q '"pgot_e8_register_fg_weight": 0.01' "${OUTPUT_DIR}/checkpoint-1/config.json"
grep -q 'loss_e8_owner' "${OUTPUT_DIR}/smoke_train.log"
grep -q 'loss_e8_reader' "${OUTPUT_DIR}/smoke_train.log"
grep -q 'loss_e8_causal' "${OUTPUT_DIR}/smoke_train.log"
grep -q 'e9_final_ovt_bottleneck_enabled' "${OUTPUT_DIR}/smoke_train.log"
grep -q 'e9_causal_intervenes_reader_keys' "${OUTPUT_DIR}/smoke_train.log"
grep -q 'e8_owner_l20_fg_acc' "${OUTPUT_DIR}/smoke_train.log"
grep -q 'e8_owner_l23_fg_acc' "${OUTPUT_DIR}/smoke_train.log"
grep -q 'e8_owner_l26_fg_acc' "${OUTPUT_DIR}/smoke_train.log"
grep -q 'eval_loss' "${OUTPUT_DIR}/smoke_train.log"

"${PYTHON}" - "${OUTPUT_DIR}/smoke_train.log" <<'PY'
import re
import sys

text = open(sys.argv[1], encoding="utf-8", errors="replace").read()

def values(name):
    return [float(x) for x in re.findall(rf"'{name}': ([0-9.eE+-]+)", text)]

retain = values("e9_gru_retain_gate_mean")
assert retain and all(0.0 < value < 1.0 for value in retain), retain
kv_diff = values("e9_reader_key_value_max_diff")
assert kv_diff and all(value == 0.0 for value in kv_diff), kv_diff
bottleneck = values("e9_final_ovt_bottleneck_enabled")
assert bottleneck and all(value == 1.0 for value in bottleneck), bottleneck
causal_keys = values("e9_causal_intervenes_reader_keys")
assert causal_keys and any(value == 1.0 for value in causal_keys), causal_keys
print("E9.1 final-OVT Reader K/V and causal-key intervention verified")
PY

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
    with safe_open(os.path.join(root, "model.safetensors"), framework="pt", device="cpu") as handle:
        keys = set(handle.keys())
assert any("pgot_e9_writer.gru_x_gates" in key for key in keys)
assert any("pgot_e9_writer.slot_up" in key for key in keys)
assert any("pgot_e8_reader.value" in key for key in keys)
print("E9.1 unified Writer and Reader checkpoint tensors verified")
PY

WANDB_RUN_DIR="$(find "${OUTPUT_DIR}/wandb" -maxdepth 3 -type d -name 'offline-run-*' | head -n 1)"
test -n "${WANDB_RUN_DIR}"
test -n "$(find "${WANDB_RUN_DIR}" -maxdepth 1 -type f -name 'run-*.wandb' | head -n 1)"
WANDB_TABLE="$(find "${WANDB_RUN_DIR}/files/media/table" -type f -name '*.table.json' | head -n 1)"
test -n "${WANDB_TABLE}"
grep -q 'ovt_owner' "${WANDB_TABLE}"
grep -q 'our_recon' "${WANDB_TABLE}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" MODEL_PATH="${OUTPUT_DIR}/checkpoint-1" \
VAL_JSONL="${VAL_JSONL}" OUTPUT_DIR="${EVAL_OUTPUT_DIR}" MAX_SAMPLES=2 \
BATCH_SIZE=1 NUM_WORKERS=1 DTYPE=fp32 DIFFUSION_INFERENCE_STEPS=2 \
COMPUTE_RFID=True bash "${PROJECT_ROOT}/scripts/eval_pgot_e9_1_final_ovt.sh" \
    2>&1 | tee "${EVAL_OUTPUT_DIR}/smoke_eval.log"

test -f "${EVAL_OUTPUT_DIR}/summary.json"
grep -q '"e8_update_mode": "final_ovt"' "${EVAL_OUTPUT_DIR}/summary.json"
grep -q '"e9_unified_ovt_update": true' "${EVAL_OUTPUT_DIR}/summary.json"
grep -q '"e9_final_ovt_bottleneck": true' "${EVAL_OUTPUT_DIR}/summary.json"
grep -q '"causal_intervention_target": "final Reader key/value OVT"' "${EVAL_OUTPUT_DIR}/summary.json"
grep -q '"owner_source": "e9_final_ovt_gru_writer"' "${EVAL_OUTPUT_DIR}/summary.json"
grep -q '"reader_supervision_mode": "writer"' "${EVAL_OUTPUT_DIR}/summary.json"
grep -q '"recon_mse"' "${EVAL_OUTPUT_DIR}/summary.json"

cleanup_smoke
test ! -e "${OUTPUT_DIR}"
test ! -e "${EVAL_OUTPUT_DIR}"
trap - EXIT
echo "PGOT E9.1 smoke passed; smoke outputs were removed."
