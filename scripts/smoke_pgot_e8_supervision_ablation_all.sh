#!/usr/bin/env bash
# Two-GPU FP32 smoke for all five Writer/Reader supervision combinations.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TRAIN_JSONL="${TRAIN_JSONL:-${PROJECT_ROOT}/data/pgot_pix2cap_thing_train.jsonl}"
VAL_JSONL="${VAL_JSONL:-${PROJECT_ROOT}/data/pgot_pix2cap_generated_val5k.jsonl}"
PYTHON="${PYTHON:-/home/jovyan/.conda/envs/scale_rae/bin/python}"
SMOKE_ROOT="$(mktemp -d "${PROJECT_ROOT}/.smoke_e8_supervision.XXXXXX")"

case "$(realpath -m "${SMOKE_ROOT}")" in
    "${PROJECT_ROOT}/.smoke_e8_supervision."*) ;;
    *) echo "Refusing unsafe smoke path: ${SMOKE_ROOT}" >&2; exit 1 ;;
esac
cleanup_smoke() { rm -rf -- "${SMOKE_ROOT}"; }
trap cleanup_smoke EXIT

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export WANDB_MODE=offline
export WANDB_PROJECT=PGOT

experiments=(A B C D E)
owner_weights=(1.0 1.0 0.0 0.0 0.0)
reader_modes=(writer none gt writer none)

for index in "${!experiments[@]}"; do
    experiment="${experiments[$index]}"
    owner_weight="${owner_weights[$index]}"
    reader_mode="${reader_modes[$index]}"
    run_root="${SMOKE_ROOT}/run_${experiment}"
    checkpoint_root="${run_root}/checkpoint"
    eval_root="${run_root}/eval"
    mkdir -p "${checkpoint_root}" "${eval_root}"

    echo "Smoke training E8 supervision ablation ${experiment}"
    EXPERIMENT="${experiment}" OUTPUT_DIR="${checkpoint_root}" \
    WANDB_NAME="pgot_smoke_e8_supervision_${experiment}" \
    WANDB_DIR="${checkpoint_root}/wandb" NUM_GPUS=2 \
    PER_DEVICE_TRAIN_BATCH_SIZE=1 GLOBAL_BATCH_SIZE=2 \
    PER_DEVICE_EVAL_BATCH_SIZE=1 MAX_STEPS=1 SAVE_STEPS=1 EVAL_STEPS=1 \
    LOGGING_STEPS=1 EVAL_NUM_IMAGES=1 PGOT_EVAL_LOG_RECON_IMAGES=1 \
    DATALOADER_NUM_WORKERS=1 TRAIN_JSONL="${TRAIN_JSONL}" VAL_JSONL="${VAL_JSONL}" \
    REPORT_TO=wandb bash \
        "${PROJECT_ROOT}/scripts/train_pgot_e8_supervision_ablation.sh" \
        2>&1 | tee "${run_root}/train.log"

    checkpoint="${checkpoint_root}/checkpoint-1"
    test -f "${checkpoint}/config.json"
    test -f "${checkpoint}/trainer_state.json"
    "${PYTHON}" - "${checkpoint}" "${owner_weight}" "${reader_mode}" <<'PY'
import json
import os
import sys
from safetensors import safe_open

root, expected_owner_weight, expected_reader_mode = sys.argv[1:]
with open(os.path.join(root, "config.json")) as f:
    config = json.load(f)
assert config["pgot_e8_visual_memory_enable"] is True
assert config["pgot_e8_clean_refinement"] is True
assert config["pgot_e8_inject_memory"] is False
assert config["pgot_e8_causal_enable"] is False
assert float(config["pgot_e8_owner_weight"]) == float(expected_owner_weight)
assert config["pgot_e8_reader_supervision_mode"] == expected_reader_mode

index_path = os.path.join(root, "model.safetensors.index.json")
if os.path.exists(index_path):
    with open(index_path) as f:
        keys = set(json.load(f)["weight_map"])
else:
    with safe_open(os.path.join(root, "model.safetensors"), framework="pt", device="cpu") as f:
        keys = set(f.keys())
assert any("pgot_e8_writer.memory_to_query" in key for key in keys)
assert any("pgot_e8_reader.value" in key for key in keys)
PY

    grep -q 'loss_e8_owner' "${run_root}/train.log"
    grep -q 'loss_e8_reader' "${run_root}/train.log"
    grep -q 'loss_e8_reader_gt_object_diagnostic' "${run_root}/train.log"
    grep -q 'loss_e8_reader_writer_object_diagnostic' "${run_root}/train.log"
    grep -q 'e8_reader_writer_kl' "${run_root}/train.log"
    grep -q "e8_reader_supervision_${reader_mode}" "${run_root}/train.log"
    grep -q 'eval_loss' "${run_root}/train.log"

    wandb_run="$(find "${checkpoint_root}/wandb" -maxdepth 3 -type d -name 'offline-run-*' | head -n 1)"
    test -n "${wandb_run}"
    test -n "$(find "${wandb_run}" -maxdepth 1 -type f -name 'run-*.wandb' | head -n 1)"
    wandb_table="$(find "${wandb_run}/files/media/table" -type f -name '*.table.json' | head -n 1)"
    test -n "${wandb_table}"
    grep -q 'ovt_owner' "${wandb_table}"
    grep -q 'our_recon' "${wandb_table}"

    echo "Standalone evaluation E8 supervision ablation ${experiment}"
    MODEL_PATH="${checkpoint}" OUTPUT_DIR="${eval_root}" VAL_JSONL="${VAL_JSONL}" \
    MAX_SAMPLES=1 BATCH_SIZE=1 NUM_WORKERS=1 DTYPE=fp32 \
    DIFFUSION_INFERENCE_STEPS=2 COMPUTE_RFID=True bash \
        "${PROJECT_ROOT}/scripts/eval_pgot_e8_visual_memory.sh" \
        2>&1 | tee "${run_root}/eval.log"

    "${PYTHON}" - "${eval_root}/summary.json" "${owner_weight}" "${reader_mode}" <<'PY'
import json
import sys

summary_path, expected_owner_weight, expected_reader_mode = sys.argv[1:]
with open(summary_path) as f:
    summary = json.load(f)
assert summary["readout"] == "ovt_owner"
assert summary["e8_visual_memory_bottleneck"] is True
assert summary["e8_clean_refinement"] is True
assert summary["e8_memory_injection_enabled"] is False
assert float(summary["owner_weight"]) == float(expected_owner_weight)
assert summary["reader_supervision_mode"] == expected_reader_mode
assert "recon_mse" in summary
PY
done

cleanup_smoke
test ! -e "${SMOKE_ROOT}"
trap - EXIT
echo "All five E8 supervision smokes passed; smoke outputs were removed."
