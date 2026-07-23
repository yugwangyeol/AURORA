#!/bin/bash
# Train + in-train eval + offline W&B + save + standalone eval, then remove artifacts.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/checkpoints/pgot_smoke_core_alllayer}"
EVAL_OUTPUT_DIR="${EVAL_OUTPUT_DIR:-${PROJECT_ROOT}/outputs/smoke_pgot_core_alllayer_eval}"
TRAIN_JSONL="${TRAIN_JSONL:-${PROJECT_ROOT}/data/pgot_train.jsonl}"
VAL_JSONL="${VAL_JSONL:-${PROJECT_ROOT}/data/pgot_val.jsonl}"
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
    rm -rf "${OUTPUT_DIR}" "${EVAL_OUTPUT_DIR}"
}
trap cleanup_smoke EXIT
rm -rf "${OUTPUT_DIR}" "${EVAL_OUTPUT_DIR}"
mkdir -p "${OUTPUT_DIR}" "${EVAL_OUTPUT_DIR}"

export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export PYTHONNOUSERSITE=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export WANDB_MODE=offline WANDB_PROJECT=PGOT WANDB_NAME=pgot_smoke_core_alllayer
export WANDB_DIR="${OUTPUT_DIR}/wandb"

NUM_GPUS=1 PER_DEVICE_TRAIN_BATCH_SIZE=1 PER_DEVICE_EVAL_BATCH_SIZE=1 \
GRADIENT_ACCUMULATION_STEPS=1 MAX_STEPS=1 SAVE_STEPS=1 EVAL_STEPS=1 \
LOGGING_STEPS=1 EVAL_NUM_IMAGES=1 PGOT_EVAL_LOG_RECON_IMAGES=1 \
PGOT_N_VOID=1 CORE_OUTSIDE_WEIGHT=1.0 CORE_OUTSIDE_LAYERS=all CORE_VOID_WEIGHT=1.0 CORE_TAIL_WEIGHT=0.0 \
TRAIN_JSONL="${TRAIN_JSONL}" VAL_JSONL="${VAL_JSONL}" OUTPUT_DIR="${OUTPUT_DIR}" \
WANDB_NAME=pgot_smoke_core_alllayer \
bash "${PROJECT_ROOT}/scripts/train_pgot_core_alllayer.sh" 2>&1 | tee "${OUTPUT_DIR}/smoke_train.log"

test -f "${OUTPUT_DIR}/checkpoint-1/config.json"
test -f "${OUTPUT_DIR}/checkpoint-1/trainer_state.json"
grep -q '"pgot_n_ovt_per_object": 1' "${OUTPUT_DIR}/config.json"
grep -q '"pgot_n_null_bg": 1' "${OUTPUT_DIR}/config.json"
grep -q '"pgot_core_outside_layers": "all"' "${OUTPUT_DIR}/config.json"
grep -q 'loss_core_outside' "${OUTPUT_DIR}/smoke_train.log"
grep -q 'core_outside_layer_00' "${OUTPUT_DIR}/smoke_train.log"
grep -q 'core_outside_layer_27' "${OUTPUT_DIR}/smoke_train.log"
grep -q 'core_void_outside_layer_00' "${OUTPUT_DIR}/smoke_train.log"
grep -q 'core_void_outside_layer_27' "${OUTPUT_DIR}/smoke_train.log"
grep -q 'eval_loss' "${OUTPUT_DIR}/smoke_train.log"

WANDB_RUN_DIR="$(find "${OUTPUT_DIR}/wandb" -maxdepth 3 -type d -name 'offline-run-*' | head -n 1)"
test -n "${WANDB_RUN_DIR}"
test -n "$(find "${WANDB_RUN_DIR}" -maxdepth 1 -type f -name 'run-*.wandb' | head -n 1)"
test -n "$(find "${WANDB_RUN_DIR}/files/media/table/eval" -type f -name '*.table.json' | head -n 1)"

MODEL_PATH="${OUTPUT_DIR}" VAL_JSONL="${VAL_JSONL}" OUTPUT_DIR="${EVAL_OUTPUT_DIR}" \
MAX_SAMPLES=1 BATCH_SIZE=1 NUM_WORKERS=1 DTYPE=fp32 \
bash "${PROJECT_ROOT}/scripts/eval_pgot_core_alllayer.sh" 2>&1 | tee "${EVAL_OUTPUT_DIR}/smoke_eval.log"
test -f "${EVAL_OUTPUT_DIR}/summary.json"
grep -q '"readout": "llm_attention"' "${EVAL_OUTPUT_DIR}/summary.json"

rm -rf "${OUTPUT_DIR}" "${EVAL_OUTPUT_DIR}"
test ! -e "${OUTPUT_DIR}"
test ! -e "${EVAL_OUTPUT_DIR}"
trap - EXIT
echo "PGOT core smoke passed; smoke outputs were removed."
