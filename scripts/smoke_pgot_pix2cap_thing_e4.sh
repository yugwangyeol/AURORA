#!/usr/bin/env bash
# One-GPU E4 train + in-train eval + save + offline W&B + standalone eval.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/checkpoints/pgot_smoke_e4}"
EVAL_OUTPUT_DIR="${EVAL_OUTPUT_DIR:-${PROJECT_ROOT}/outputs/smoke_pgot_e4_eval}"
TRAIN_JSONL="${TRAIN_JSONL:-${PROJECT_ROOT}/data/pgot_pix2cap_thing_train.jsonl}"
VAL_JSONL="${VAL_JSONL:-${PROJECT_ROOT}/data/pgot_pix2cap_thing_val.jsonl}"
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
export WANDB_MODE=offline
export WANDB_PROJECT=PGOT
export WANDB_NAME=pgot_smoke_e4
export WANDB_DIR="${OUTPUT_DIR}/wandb"

NUM_GPUS=1 PER_DEVICE_TRAIN_BATCH_SIZE=1 GLOBAL_BATCH_SIZE=1 \
PER_DEVICE_EVAL_BATCH_SIZE=1 MAX_STEPS=1 SAVE_STEPS=1 EVAL_STEPS=1 \
LOGGING_STEPS=1 EVAL_NUM_IMAGES=1 PGOT_EVAL_LOG_RECON_IMAGES=1 \
E4_LOSS_WARMUP_STEPS=0 DATALOADER_NUM_WORKERS=1 \
TRAIN_JSONL="${TRAIN_JSONL}" VAL_JSONL="${VAL_JSONL}" OUTPUT_DIR="${OUTPUT_DIR}" \
WANDB_NAME=pgot_smoke_e4 \
bash "${PROJECT_ROOT}/scripts/train_pgot_pix2cap_thing_e4.sh" 2>&1 | tee "${OUTPUT_DIR}/smoke_train.log"

test -f "${OUTPUT_DIR}/checkpoint-1/config.json"
test -f "${OUTPUT_DIR}/checkpoint-1/trainer_state.json"
grep -q '"pgot_e4_rae_isolated": true' "${OUTPUT_DIR}/checkpoint-1/config.json"
grep -q '"pgot_e4_full_inside_target": 0.3' "${OUTPUT_DIR}/checkpoint-1/config.json"
grep -q '"pgot_e4_rae_bind_layers": "last8"' "${OUTPUT_DIR}/checkpoint-1/config.json"
grep -q 'loss_e4_full_inside' "${OUTPUT_DIR}/smoke_train.log"
grep -q 'loss_e4_rae_bind' "${OUTPUT_DIR}/smoke_train.log"
grep -q 'e4_rae_bind_layer_20' "${OUTPUT_DIR}/smoke_train.log"
grep -q 'e4_rae_bind_layer_27' "${OUTPUT_DIR}/smoke_train.log"
grep -q 'e4_rae_other_query_mass' "${OUTPUT_DIR}/smoke_train.log"
grep -q 'eval_loss' "${OUTPUT_DIR}/smoke_train.log"

WANDB_RUN_DIR="$(find "${OUTPUT_DIR}/wandb" -maxdepth 3 -type d -name 'offline-run-*' | head -n 1)"
test -n "${WANDB_RUN_DIR}"
test -n "$(find "${WANDB_RUN_DIR}" -maxdepth 1 -type f -name 'run-*.wandb' | head -n 1)"
test -n "$(find "${WANDB_RUN_DIR}/files/media/table/eval" -type f -name '*.table.json' | head -n 1)"

MODEL_PATH="${OUTPUT_DIR}/checkpoint-1" VAL_JSONL="${VAL_JSONL}" \
OUTPUT_DIR="${EVAL_OUTPUT_DIR}" MAX_SAMPLES=1 BATCH_SIZE=1 NUM_WORKERS=1 \
DTYPE=fp32 DIFFUSION_INFERENCE_STEPS=2 \
bash "${PROJECT_ROOT}/scripts/eval_pgot_pix2cap_thing_e4.sh" 2>&1 | tee "${EVAL_OUTPUT_DIR}/smoke_eval.log"
test -f "${EVAL_OUTPUT_DIR}/summary.json"
grep -q '"readout": "llm_attention"' "${EVAL_OUTPUT_DIR}/summary.json"

rm -rf "${OUTPUT_DIR}" "${EVAL_OUTPUT_DIR}"
test ! -e "${OUTPUT_DIR}"
test ! -e "${EVAL_OUTPUT_DIR}"
trap - EXIT
echo "PGOT E4 smoke passed; smoke outputs were removed."
