#!/usr/bin/env bash
# Train/eval/save/offline-W&B/standalone-eval smoke for E5.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/checkpoints/pgot_smoke_e5_causal_mixed}"
EVAL_OUTPUT_DIR="${EVAL_OUTPUT_DIR:-${PROJECT_ROOT}/outputs/smoke_pgot_e5_causal_mixed_eval}"
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
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export WANDB_MODE=offline
export WANDB_PROJECT=PGOT
export WANDB_NAME=pgot_smoke_e5_causal_mixed
export WANDB_DIR="${OUTPUT_DIR}/wandb"

NUM_GPUS=2 PER_DEVICE_TRAIN_BATCH_SIZE=1 GLOBAL_BATCH_SIZE=2 \
PER_DEVICE_EVAL_BATCH_SIZE=1 MAX_STEPS=1 SAVE_STEPS=1 EVAL_STEPS=1 \
LOGGING_STEPS=1 EVAL_NUM_IMAGES=2 PGOT_EVAL_LOG_RECON_IMAGES=1 \
DATALOADER_NUM_WORKERS=1 E5_FORCING_PROBABILITY=1.0 E4_LOSS_WARMUP_STEPS=0 \
TRAIN_JSONL="${TRAIN_JSONL}" VAL_JSONL="${VAL_JSONL}" OUTPUT_DIR="${OUTPUT_DIR}" \
WANDB_NAME=pgot_smoke_e5_causal_mixed \
bash "${PROJECT_ROOT}/scripts/train_pgot_e5_causal_mixed.sh" 2>&1 | tee "${OUTPUT_DIR}/smoke_train.log"

test -f "${OUTPUT_DIR}/checkpoint-1/config.json"
test -f "${OUTPUT_DIR}/checkpoint-1/trainer_state.json"
grep -q '"pgot_ovt_caption_init": true' "${OUTPUT_DIR}/checkpoint-1/config.json"
grep -q '"pgot_ovt_isolated_attention": true' "${OUTPUT_DIR}/checkpoint-1/config.json"
grep -q '"pgot_ovt_attends_own_caption": true' "${OUTPUT_DIR}/checkpoint-1/config.json"
grep -q '"pgot_e5_forcing_probability": 1.0' "${OUTPUT_DIR}/checkpoint-1/config.json"
grep -q '"pgot_register_hard_gt_mask": true' "${OUTPUT_DIR}/checkpoint-1/config.json"
grep -q '"pgot_e4_rae_isolated": true' "${OUTPUT_DIR}/checkpoint-1/config.json"
grep -q '"pgot_e3_attention_competition_weight": 0.0' "${OUTPUT_DIR}/checkpoint-1/config.json"
grep -q '"pgot_e4_rae_bind_weight": 0.0' "${OUTPUT_DIR}/checkpoint-1/config.json"
grep -q '"pgot_e4_full_inside_weight": 0.1' "${OUTPUT_DIR}/checkpoint-1/config.json"
grep -q 'loss_core_outside' "${OUTPUT_DIR}/smoke_train.log"
grep -q 'loss_e4_full_inside' "${OUTPUT_DIR}/smoke_train.log"
grep -q 'e5_forcing_fraction' "${OUTPUT_DIR}/smoke_train.log"
grep -q 'register_hard_mask_blocked_patch_fraction' "${OUTPUT_DIR}/smoke_train.log"
grep -q 'eval_loss' "${OUTPUT_DIR}/smoke_train.log"
grep -Fq '[PGOT/LoRA] reloaded checkpoint adapter weights' "${OUTPUT_DIR}/smoke_train.log"
grep -Eq 'missing_lora=0 unexpected=0' "${OUTPUT_DIR}/smoke_train.log"

WANDB_RUN_DIR="$(find "${OUTPUT_DIR}/wandb" -maxdepth 3 -type d -name 'offline-run-*' | head -n 1)"
test -n "${WANDB_RUN_DIR}"
test -n "$(find "${WANDB_RUN_DIR}" -maxdepth 1 -type f -name 'run-*.wandb' | head -n 1)"
test -n "$(find "${WANDB_RUN_DIR}/files/media/table/eval" -type f -name '*.table.json' | head -n 1)"

CUDA_VISIBLE_DEVICES=0 MODEL_PATH="${OUTPUT_DIR}/checkpoint-1" VAL_JSONL="${VAL_JSONL}" \
OUTPUT_DIR="${EVAL_OUTPUT_DIR}" MAX_SAMPLES=1 BATCH_SIZE=1 NUM_WORKERS=1 \
DTYPE=fp32 DIFFUSION_INFERENCE_STEPS=2 REGISTER_EVAL_ROUTE=predicted_ovt \
bash "${PROJECT_ROOT}/scripts/eval_pgot_e5_causal_mixed.sh" 2>&1 | tee "${EVAL_OUTPUT_DIR}/smoke_eval.log"
test -f "${EVAL_OUTPUT_DIR}/summary.json"
grep -q '"readout": "llm_attention"' "${EVAL_OUTPUT_DIR}/summary.json"
grep -q '"register_eval_route": "predicted_ovt"' "${EVAL_OUTPUT_DIR}/summary.json"

rm -rf "${OUTPUT_DIR}" "${EVAL_OUTPUT_DIR}"
test ! -e "${OUTPUT_DIR}"
test ! -e "${EVAL_OUTPUT_DIR}"
trap - EXIT
echo "PGOT E5 causal-mixed smoke passed; smoke outputs were removed."
