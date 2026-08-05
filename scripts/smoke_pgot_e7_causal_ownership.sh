#!/usr/bin/env bash
# One-GPU E7 train/eval/save/offline-W&B/standalone-eval smoke.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/checkpoints/pgot_smoke_e7_causal_ownership}"
EVAL_OUTPUT_DIR="${EVAL_OUTPUT_DIR:-${PROJECT_ROOT}/outputs/smoke_pgot_e7_causal_ownership_eval}"
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
cleanup_smoke() { rm -rf "${OUTPUT_DIR}" "${EVAL_OUTPUT_DIR}"; }
trap cleanup_smoke EXIT
rm -rf "${OUTPUT_DIR}" "${EVAL_OUTPUT_DIR}"
mkdir -p "${OUTPUT_DIR}" "${EVAL_OUTPUT_DIR}"

export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export PYTHONNOUSERSITE=1
export CUDA_VISIBLE_DEVICES=0
export WANDB_MODE=offline
export WANDB_PROJECT=PGOT
export WANDB_NAME=pgot_smoke_e7_causal_ownership
export WANDB_DIR="${OUTPUT_DIR}/wandb"

# Batch 8 makes a same-category cross-image match deterministic for seed 42;
# timestep floor 0 exercises the counterfactual branch in the one-step smoke.
NUM_GPUS=1 PER_DEVICE_TRAIN_BATCH_SIZE=8 GLOBAL_BATCH_SIZE=8 \
PER_DEVICE_EVAL_BATCH_SIZE=1 MAX_STEPS=1 SAVE_STEPS=1 EVAL_STEPS=1 \
LOGGING_STEPS=1 EVAL_NUM_IMAGES=1 PGOT_EVAL_LOG_RECON_IMAGES=1 \
E7_EVAL_INFERENCE_STEPS=1 E7_CAUSAL_MIN_TIMESTEP=0 \
E7_CAUSAL_WARMUP_START_STEPS=0 E7_CAUSAL_RAMP_STEPS=0 \
DATALOADER_NUM_WORKERS=1 TRAIN_JSONL="${TRAIN_JSONL}" VAL_JSONL="${VAL_JSONL}" \
OUTPUT_DIR="${OUTPUT_DIR}" WANDB_NAME=pgot_smoke_e7_causal_ownership \
bash "${PROJECT_ROOT}/scripts/train_pgot_e7_causal_ownership.sh" 2>&1 | tee "${OUTPUT_DIR}/smoke_train.log"

test -f "${OUTPUT_DIR}/checkpoint-1/config.json"
test -f "${OUTPUT_DIR}/checkpoint-1/trainer_state.json"
grep -q '"pgot_e7_enable": true' "${OUTPUT_DIR}/checkpoint-1/config.json"
grep -q '"pgot_e6_enable": false' "${OUTPUT_DIR}/checkpoint-1/config.json"
grep -q '"pgot_register_hard_gt_mask": true' "${OUTPUT_DIR}/checkpoint-1/config.json"
grep -q 'loss_e7_diffusion' "${OUTPUT_DIR}/smoke_train.log"
grep -q 'loss_e7_owner' "${OUTPUT_DIR}/smoke_train.log"
grep -q 'loss_e7_causal' "${OUTPUT_DIR}/smoke_train.log"
grep -q 'e7_causal_error_gap' "${OUTPUT_DIR}/smoke_train.log"
grep -q 'loss_core_outside' "${OUTPUT_DIR}/smoke_train.log"
grep -q 'eval_loss' "${OUTPUT_DIR}/smoke_train.log"
if grep -q "'loss_mask':" "${OUTPUT_DIR}/smoke_train.log"; then
    echo "Unused aggregate loss_mask leaked into E7 W&B metrics" >&2; exit 1
fi
if grep -q "'epoch':" "${OUTPUT_DIR}/smoke_train.log"; then
    echo "Non-actionable epoch chart leaked into E7 W&B metrics" >&2; exit 1
fi

"${PYTHON}" - "${OUTPUT_DIR}/checkpoint-1" <<'PY'
import json, os, sys
from safetensors import safe_open
root=sys.argv[1]
index=os.path.join(root,'model.safetensors.index.json')
if os.path.exists(index):
    with open(index) as f: keys=set(json.load(f)['weight_map'])
else:
    with safe_open(os.path.join(root,'model.safetensors'),framework='pt',device='cpu') as f:
        keys=set(f.keys())
assert any('pgot_e7_decoder.owner_write.owner_query' in k for k in keys)
assert any('pgot_e7_decoder.context_projector' in k for k in keys)
assert any('pgot_e7_decoder.unet.' in k and '.lora_up.' in k for k in keys)
assert not any('pgot_e7_decoder.unet.' in k and '.base_layer.' in k for k in keys)
assert not any('pgot_e7_decoder.vae.' in k for k in keys)
assert not any(k.startswith('diff_head.') for k in keys)
print('compact E7 checkpoint verified:', len(keys), 'keys')
PY

WANDB_RUN_DIR="$(find "${OUTPUT_DIR}/wandb" -maxdepth 3 -type d -name 'offline-run-*' | head -n 1)"
test -n "${WANDB_RUN_DIR}"
test -n "$(find "${WANDB_RUN_DIR}" -maxdepth 1 -type f -name 'run-*.wandb' | head -n 1)"
test -n "$(find "${WANDB_RUN_DIR}/files/media/table" -type f -name '*.table.json' | head -n 1)"

CUDA_VISIBLE_DEVICES=0 MODEL_PATH="${OUTPUT_DIR}/checkpoint-1" VAL_JSONL="${VAL_JSONL}" \
OUTPUT_DIR="${EVAL_OUTPUT_DIR}" MAX_SAMPLES=2 BATCH_SIZE=32 NUM_WORKERS=1 \
DTYPE=bf16 DIFFUSION_INFERENCE_STEPS=1 COMPUTE_RFID=True \
bash "${PROJECT_ROOT}/scripts/eval_pgot_e7_causal_ownership.sh" 2>&1 | tee "${EVAL_OUTPUT_DIR}/smoke_eval.log"

test -f "${EVAL_OUTPUT_DIR}/summary.json"
grep -q '"readout": "e7_owner"' "${EVAL_OUTPUT_DIR}/summary.json"
grep -q '"e7_causal_ownership_bottleneck": true' "${EVAL_OUTPUT_DIR}/summary.json"
grep -q '"positive_null_prompt_tokens": 0' "${EVAL_OUTPUT_DIR}/summary.json"
grep -q '"scale_rae_queries_in_sequence": 0' "${EVAL_OUTPUT_DIR}/summary.json"
grep -q '"register_eval_route": "unrestricted"' "${EVAL_OUTPUT_DIR}/summary.json"
grep -q '"recon_mse"' "${EVAL_OUTPUT_DIR}/summary.json"

rm -rf "${OUTPUT_DIR}" "${EVAL_OUTPUT_DIR}"
test ! -e "${OUTPUT_DIR}"
test ! -e "${EVAL_OUTPUT_DIR}"
trap - EXIT
echo "PGOT E7 causal-ownership smoke passed; smoke outputs were removed."
