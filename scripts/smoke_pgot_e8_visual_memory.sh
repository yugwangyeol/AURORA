#!/usr/bin/env bash
# Two-GPU E8.1 + E8.2 train/eval/save/offline-W&B/standalone-eval smoke.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/checkpoints/pgot_smoke_e8_1_clean}"
CAUSAL_OUTPUT_DIR="${CAUSAL_OUTPUT_DIR:-${PROJECT_ROOT}/checkpoints/pgot_smoke_e8_2_paired_causal}"
EVAL_OUTPUT_DIR="${EVAL_OUTPUT_DIR:-${PROJECT_ROOT}/outputs/smoke_pgot_e8_2_paired_causal_eval}"
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
case "$(realpath -m "${CAUSAL_OUTPUT_DIR}")" in
    "${PROJECT_ROOT}/checkpoints/"*smoke*) ;;
    *) echo "Refusing unsafe smoke causal checkpoint path: ${CAUSAL_OUTPUT_DIR}" >&2; exit 1 ;;
esac
cleanup_smoke() {
    local path
    for path in "${OUTPUT_DIR}" "${CAUSAL_OUTPUT_DIR}" "${EVAL_OUTPUT_DIR}"; do
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
mkdir -p "${OUTPUT_DIR}" "${CAUSAL_OUTPUT_DIR}" "${EVAL_OUTPUT_DIR}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export WANDB_MODE=offline
export WANDB_PROJECT=PGOT
export WANDB_NAME=pgot_smoke_e8_visual_memory
export WANDB_DIR="${OUTPUT_DIR}/wandb"

NUM_GPUS=2 PER_DEVICE_TRAIN_BATCH_SIZE=1 GLOBAL_BATCH_SIZE=2 \
PER_DEVICE_EVAL_BATCH_SIZE=1 MAX_STEPS=1 SAVE_STEPS=1 EVAL_STEPS=1 \
LOGGING_STEPS=1 EVAL_NUM_IMAGES=1 PGOT_EVAL_LOG_RECON_IMAGES=1 \
DATALOADER_NUM_WORKERS=1 TRAIN_JSONL="${TRAIN_JSONL}" VAL_JSONL="${VAL_JSONL}" \
OUTPUT_DIR="${OUTPUT_DIR}" REPORT_TO=wandb WANDB_NAME=pgot_smoke_e8_visual_memory \
E8_READER_SUPERVISION_MODE=writer \
bash "${PROJECT_ROOT}/scripts/train_pgot_e8_visual_memory.sh" \
    2>&1 | tee "${OUTPUT_DIR}/smoke_train.log"

test -f "${OUTPUT_DIR}/checkpoint-1/config.json"
test -f "${OUTPUT_DIR}/checkpoint-1/trainer_state.json"
grep -q '"pgot_e8_visual_memory_enable": true' "${OUTPUT_DIR}/checkpoint-1/config.json"
grep -q '"pgot_n_null_bg": 0' "${OUTPUT_DIR}/checkpoint-1/config.json"
grep -q '"pgot_e8_layers": "21,24,27"' "${OUTPUT_DIR}/checkpoint-1/config.json"
grep -q '"pgot_e8_clean_refinement": true' "${OUTPUT_DIR}/checkpoint-1/config.json"
grep -q '"pgot_e8_inject_memory": false' "${OUTPUT_DIR}/checkpoint-1/config.json"
grep -q '"pgot_e8_causal_enable": false' "${OUTPUT_DIR}/checkpoint-1/config.json"
grep -q 'loss_e8_owner' "${OUTPUT_DIR}/smoke_train.log"
grep -q 'loss_e8_reader' "${OUTPUT_DIR}/smoke_train.log"
grep -q 'e8_owner_l21_fg_acc' "${OUTPUT_DIR}/smoke_train.log"
grep -q 'e8_reader_register_mass_on_fg' "${OUTPUT_DIR}/smoke_train.log"
grep -q 'e8_visual_memory_rms' "${OUTPUT_DIR}/smoke_train.log"
grep -q 'e8_typed_reader_only' "${OUTPUT_DIR}/smoke_train.log"
grep -q 'eval_loss' "${OUTPUT_DIR}/smoke_train.log"

"${PYTHON}" - "${OUTPUT_DIR}/checkpoint-1" <<'PY'
import json, os, sys
from safetensors import safe_open
root = sys.argv[1]
index = os.path.join(root, "model.safetensors.index.json")
if os.path.exists(index):
    with open(index) as f:
        keys = set(json.load(f)["weight_map"])
else:
    with safe_open(os.path.join(root, "model.safetensors"), framework="pt", device="cpu") as f:
        keys = set(f.keys())
assert any("pgot_e8_writer.memory_to_query" in k for k in keys)
assert any("pgot_e8_reader.value" in k for k in keys)
print("E8.1 writer/reader checkpoint tensors verified")
PY

WANDB_RUN_DIR="$(find "${OUTPUT_DIR}/wandb" -maxdepth 3 -type d -name 'offline-run-*' | head -n 1)"
test -n "${WANDB_RUN_DIR}"
test -n "$(find "${WANDB_RUN_DIR}" -maxdepth 1 -type f -name 'run-*.wandb' | head -n 1)"
WANDB_TABLE="$(find "${WANDB_RUN_DIR}/files/media/table" -type f -name '*.table.json' | head -n 1)"
test -n "${WANDB_TABLE}"
grep -q 'ovt_owner' "${WANDB_TABLE}"
grep -q 'our_recon' "${WANDB_TABLE}"

# Stage 2 starts from the saved E8.1 checkpoint and forces the paired branch
# on this one-step smoke so all causal losses and the 3-way DiT call execute.
WANDB_NAME=pgot_smoke_e8_2_paired_causal WANDB_DIR="${CAUSAL_OUTPUT_DIR}/wandb" \
MODEL_PATH="${OUTPUT_DIR}/checkpoint-1" OUTPUT_DIR="${CAUSAL_OUTPUT_DIR}" \
NUM_GPUS=2 PER_DEVICE_TRAIN_BATCH_SIZE=1 GLOBAL_BATCH_SIZE=2 \
PER_DEVICE_EVAL_BATCH_SIZE=1 MAX_STEPS=1 SAVE_STEPS=1 EVAL_STEPS=1 \
LOGGING_STEPS=1 EVAL_NUM_IMAGES=1 PGOT_EVAL_LOG_RECON_IMAGES=1 \
DATALOADER_NUM_WORKERS=1 TRAIN_JSONL="${TRAIN_JSONL}" VAL_JSONL="${VAL_JSONL}" \
E8_CAUSAL_BATCH_PROBABILITY=1.0 E8_CAUSAL_RAMP_STEPS=1 REPORT_TO=wandb \
bash "${PROJECT_ROOT}/scripts/train_pgot_e8_2_paired_causal.sh" \
    2>&1 | tee "${CAUSAL_OUTPUT_DIR}/smoke_train.log"

test -f "${CAUSAL_OUTPUT_DIR}/checkpoint-1/config.json"
test -f "${CAUSAL_OUTPUT_DIR}/checkpoint-1/trainer_state.json"
grep -q '"pgot_e8_causal_enable": true' "${CAUSAL_OUTPUT_DIR}/checkpoint-1/config.json"
grep -q 'loss_e8_causal' "${CAUSAL_OUTPUT_DIR}/smoke_train.log"
grep -q 'e8_object_ablation_error_ratio' "${CAUSAL_OUTPUT_DIR}/smoke_train.log"
grep -q 'loss_e8_need_weighted' "${CAUSAL_OUTPUT_DIR}/smoke_train.log"
grep -q 'e8_need_margin_satisfied' "${CAUSAL_OUTPUT_DIR}/smoke_train.log"
grep -q 'e8_causal_active' "${CAUSAL_OUTPUT_DIR}/smoke_train.log"
grep -q 'eval_loss' "${CAUSAL_OUTPUT_DIR}/smoke_train.log"
grep -q '\[PGOT/LoRA\] reloaded checkpoint adapter weights' "${CAUSAL_OUTPUT_DIR}/smoke_train.log"
grep -q 'missing_lora=0 unexpected=0' "${CAUSAL_OUTPUT_DIR}/smoke_train.log"
grep -q '"pgot_e8_reader_supervision_mode": "writer"' "${CAUSAL_OUTPUT_DIR}/checkpoint-1/config.json"

CAUSAL_WANDB_RUN_DIR="$(find "${CAUSAL_OUTPUT_DIR}/wandb" -maxdepth 3 -type d -name 'offline-run-*' | head -n 1)"
test -n "${CAUSAL_WANDB_RUN_DIR}"
test -n "$(find "${CAUSAL_WANDB_RUN_DIR}" -maxdepth 1 -type f -name 'run-*.wandb' | head -n 1)"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" MODEL_PATH="${CAUSAL_OUTPUT_DIR}/checkpoint-1" \
VAL_JSONL="${VAL_JSONL}" OUTPUT_DIR="${EVAL_OUTPUT_DIR}" MAX_SAMPLES=2 \
BATCH_SIZE=1 NUM_WORKERS=1 DTYPE=fp32 DIFFUSION_INFERENCE_STEPS=2 \
COMPUTE_RFID=True bash "${PROJECT_ROOT}/scripts/eval_pgot_e8_visual_memory.sh" \
    2>&1 | tee "${EVAL_OUTPUT_DIR}/smoke_eval.log"

test -f "${EVAL_OUTPUT_DIR}/summary.json"
grep -q '"readout": "ovt_owner"' "${EVAL_OUTPUT_DIR}/summary.json"
grep -q '"e8_visual_memory_bottleneck": true' "${EVAL_OUTPUT_DIR}/summary.json"
grep -q '"e8_clean_refinement": true' "${EVAL_OUTPUT_DIR}/summary.json"
grep -q '"e8_memory_injection_enabled": false' "${EVAL_OUTPUT_DIR}/summary.json"
grep -q '"e8_causal_training_enabled": true' "${EVAL_OUTPUT_DIR}/summary.json"
grep -q '"rae_standard_slot_attention_blocked": true' "${EVAL_OUTPUT_DIR}/summary.json"
grep -q '"owner_source": "e8_competitive_visual_memory_writer"' "${EVAL_OUTPUT_DIR}/summary.json"
grep -q '"recon_mse"' "${EVAL_OUTPUT_DIR}/summary.json"

cleanup_smoke
test ! -e "${OUTPUT_DIR}"
test ! -e "${CAUSAL_OUTPUT_DIR}"
test ! -e "${EVAL_OUTPUT_DIR}"
trap - EXIT
echo "PGOT E8.1/E8.2 smoke passed; smoke outputs were removed."
