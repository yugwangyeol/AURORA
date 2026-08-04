#!/usr/bin/env bash
# One-GPU E6 train/eval/save/offline-W&B/standalone-eval smoke.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/checkpoints/pgot_smoke_e6_coda_direct}"
EVAL_OUTPUT_DIR="${EVAL_OUTPUT_DIR:-${PROJECT_ROOT}/outputs/smoke_pgot_e6_coda_direct_eval}"
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
export WANDB_NAME=pgot_smoke_e6_coda_direct
export WANDB_DIR="${OUTPUT_DIR}/wandb"

NUM_GPUS=1 PER_DEVICE_TRAIN_BATCH_SIZE=1 GLOBAL_BATCH_SIZE=1 \
PER_DEVICE_EVAL_BATCH_SIZE=1 MAX_STEPS=1 SAVE_STEPS=1 EVAL_STEPS=1 \
LOGGING_STEPS=1 EVAL_NUM_IMAGES=1 PGOT_EVAL_LOG_RECON_IMAGES=1 \
E6_EVAL_INFERENCE_STEPS=1 DATALOADER_NUM_WORKERS=1 \
TRAIN_JSONL="${TRAIN_JSONL}" VAL_JSONL="${VAL_JSONL}" OUTPUT_DIR="${OUTPUT_DIR}" \
WANDB_NAME=pgot_smoke_e6_coda_direct \
bash "${PROJECT_ROOT}/scripts/train_pgot_e6_coda_direct.sh" 2>&1 | tee "${OUTPUT_DIR}/smoke_train.log"

test -f "${OUTPUT_DIR}/checkpoint-1/config.json"
test -f "${OUTPUT_DIR}/checkpoint-1/trainer_state.json"
grep -q '"pgot_e6_enable": true' "${OUTPUT_DIR}/checkpoint-1/config.json"
grep -q '"pgot_ovt_attends_own_caption": true' "${OUTPUT_DIR}/checkpoint-1/config.json"
grep -q '"pgot_register_hard_gt_mask": true' "${OUTPUT_DIR}/checkpoint-1/config.json"
grep -q 'loss_e6_diffusion' "${OUTPUT_DIR}/smoke_train.log"
grep -q 'loss_core_outside' "${OUTPUT_DIR}/smoke_train.log"
grep -q 'eval_loss' "${OUTPUT_DIR}/smoke_train.log"
if grep -q "'loss_mask':" "${OUTPUT_DIR}/smoke_train.log"; then
    echo "Unused loss_mask leaked into the E6 metric profile" >&2; exit 1
fi
if grep -q "'n_objects_mean':" "${OUTPUT_DIR}/smoke_train.log"; then
    echo "Constant n_objects_mean leaked into the E6 metric profile" >&2; exit 1
fi
if grep -q "'core_num_layers':" "${OUTPUT_DIR}/smoke_train.log"; then
    echo "Constant core_num_layers leaked into the E6 metric profile" >&2; exit 1
fi
if grep -q "'epoch':" "${OUTPUT_DIR}/smoke_train.log"; then
    echo "Non-actionable epoch chart leaked into the E6 metric profile" >&2; exit 1
fi

"${PYTHON}" - "${OUTPUT_DIR}/checkpoint-1" <<'PY'
import json, os, sys
root=sys.argv[1]
index=os.path.join(root,'model.safetensors.index.json')
if os.path.exists(index):
    with open(index) as f: keys=set(json.load(f)['weight_map'])
else:
    from safetensors import safe_open
    with safe_open(os.path.join(root,'model.safetensors'),framework='pt',device='cpu') as f:
        keys=set(f.keys())
assert any('pgot_e6_decoder.context_projector' in k for k in keys)
assert any('pgot_e6_decoder.unet.' in k and '.attn2.' in k for k in keys)
assert not any('pgot_e6_decoder.vae.' in k for k in keys)
assert not any(k.startswith('diff_head.') for k in keys)
print('compact E6 checkpoint verified:', len(keys), 'keys')
PY

WANDB_RUN_DIR="$(find "${OUTPUT_DIR}/wandb" -maxdepth 3 -type d -name 'offline-run-*' | head -n 1)"
test -n "${WANDB_RUN_DIR}"
test -n "$(find "${WANDB_RUN_DIR}" -maxdepth 1 -type f -name 'run-*.wandb' | head -n 1)"
test -n "$(find "${WANDB_RUN_DIR}/files/media/table" -type f -name '*.table.json' | head -n 1)"

CUDA_VISIBLE_DEVICES=0 MODEL_PATH="${OUTPUT_DIR}/checkpoint-1" VAL_JSONL="${VAL_JSONL}" \
OUTPUT_DIR="${EVAL_OUTPUT_DIR}" MAX_SAMPLES=2 BATCH_SIZE=32 NUM_WORKERS=1 \
DTYPE=bf16 DIFFUSION_INFERENCE_STEPS=1 REGISTER_EVAL_ROUTE=predicted_ovt COMPUTE_RFID=True \
bash "${PROJECT_ROOT}/scripts/eval_pgot_e6_coda_direct.sh" 2>&1 | tee "${EVAL_OUTPUT_DIR}/smoke_eval.log"

test -f "${EVAL_OUTPUT_DIR}/summary.json"
grep -q '"readout": "e6_decoder"' "${EVAL_OUTPUT_DIR}/summary.json"
grep -q '"e6_coda_direct_bottleneck": true' "${EVAL_OUTPUT_DIR}/summary.json"
grep -q '"scale_rae_queries_in_sequence": 0' "${EVAL_OUTPUT_DIR}/summary.json"
grep -q '"register_eval_route": "predicted_ovt"' "${EVAL_OUTPUT_DIR}/summary.json"
grep -q '"recon_mse"' "${EVAL_OUTPUT_DIR}/summary.json"

rm -rf "${OUTPUT_DIR}" "${EVAL_OUTPUT_DIR}"
test ! -e "${OUTPUT_DIR}"
test ! -e "${EVAL_OUTPUT_DIR}"
trap - EXIT
echo "PGOT E6 CODA-direct smoke passed; smoke outputs were removed."
