#!/usr/bin/env bash
# Two-GPU train/eval/save/W&B smoke, then sequential TF and AR checkpoint reload.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-/home/jovyan/.conda/envs/scale_rae/bin/python}"
SMOKE_ROOT="$(mktemp -d "${PROJECT_ROOT}/.smoke_oneshot_memory8_register32_noprior.XXXXXX")"
BASE_MODEL="${MODEL_PATH:-${PROJECT_ROOT}/checkpoints/pgot_e11_capacity_dit16/checkpoint-10000}"
RUN_ROOT="${SMOKE_ROOT}/content"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export PYTHONNOUSERSITE=1
export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_PROJECT="${WANDB_PROJECT:-PGOT}"
export WANDB_DATA_DIR="${SMOKE_ROOT}/wandb_data"
export WANDB_CACHE_DIR="${SMOKE_ROOT}/wandb_cache"
export WANDB_RESUME=never
export SMOKE_ROOT RUN_ROOT
echo "Smoke workspace: ${SMOKE_ROOT}"

on_exit() {
    local status=$?
    if (( status )); then echo "Smoke failed; preserved at ${SMOKE_ROOT}" >&2; fi
}
trap on_exit EXIT

mkdir -p "${RUN_ROOT}/wandb"
printf '%s\n' "${BASE_MODEL}" > "${RUN_ROOT}/source.txt"
run_id="$("${PYTHON}" -c 'import wandb; print(wandb.util.generate_id())')"
printf '%s\n' "${run_id}" > "${RUN_ROOT}/wandb_id.txt"
WANDB_RUN_ID="${run_id}" WANDB_NAME=smoke_oneshot_memory8_register32_noprior \
WANDB_DIR="${RUN_ROOT}/wandb" MODEL_PATH="${BASE_MODEL}" \
OUTPUT_DIR="${RUN_ROOT}/train" NUM_GPUS=2 \
PER_DEVICE_TRAIN_BATCH_SIZE="${SMOKE_BATCH_SIZE:-2}" \
GLOBAL_BATCH_SIZE="${SMOKE_GLOBAL_BATCH_SIZE:-24}" \
MAX_STEPS=2 SAVE_STEPS=2 SAVE_TOTAL_LIMIT=1 EVAL_STEPS=2 LOGGING_STEPS=1 \
PER_DEVICE_EVAL_BATCH_SIZE=1 EVAL_NUM_IMAGES=2 PGOT_EVAL_LOG_RECON_IMAGES=1 \
DATALOADER_NUM_WORKERS=1 REPORT_TO=wandb PGOT_SKIP_FINAL_SAVE=1 \
MASTER_PORT="${MASTER_PORT:-29667}" \
    bash "${PROJECT_ROOT}/scripts/train_pgot_oneshot_memory8_register32_noprior.sh" \
    > "${RUN_ROOT}/train.log" 2>&1

"${PYTHON}" "${PROJECT_ROOT}/scripts/verify_oneshot_memory4_smoke.py" \
    "${SMOKE_ROOT}" content --stage train --owner-prior disabled \
    --object-memories 8 --register-memories 32 --semantic-registers 4

COMMON_EVAL_ENV=(
    MODEL_PATH="${RUN_ROOT}/train/checkpoint-2"
    MAX_SAMPLES=2 BATCH_SIZE=1 AR_BATCH_SIZE=1 NUM_WORKERS=1
    DTYPE=fp32 DIFFUSION_INFERENCE_STEPS=2 COMPUTE_RFID=True COMPUTE_KID=True
    KID_SUBSETS=2 KID_SUBSET_SIZE=2 AR_MAX_NEW_TOKENS=128
)
env "${COMMON_EVAL_ENV[@]}" CUDA_VISIBLE_DEVICES=0 OUTPUT_DIR="${RUN_ROOT}/eval/tf" \
    bash "${PROJECT_ROOT}/scripts/eval_pgot_oneshot_memory8_register32_noprior_tf.sh" \
    > "${RUN_ROOT}/eval_tf.log" 2>&1
env "${COMMON_EVAL_ENV[@]}" CUDA_VISIBLE_DEVICES=1 OUTPUT_DIR="${RUN_ROOT}/eval/ar" \
    bash "${PROJECT_ROOT}/scripts/eval_pgot_oneshot_memory8_register32_noprior_ar.sh" \
    > "${RUN_ROOT}/eval_ar.log" 2>&1

"${PYTHON}" "${PROJECT_ROOT}/scripts/verify_oneshot_memory4_smoke.py" \
    "${SMOKE_ROOT}" content --stage eval --owner-prior disabled --expect-kid \
    --expect-class-metrics --object-memories 8 --register-memories 32 \
    --semantic-registers 4

"${PYTHON}" - <<'PY'
import os
from pathlib import Path
import wandb

root = Path(os.environ["RUN_ROOT"])
if os.environ["WANDB_MODE"] == "online":
    api = wandb.Api(timeout=30)
    entity = os.environ.get("WANDB_ENTITY") or api.default_entity
    run = api.run(f"{entity}/{os.environ['WANDB_PROJECT']}/{(root / 'wandb_id.txt').read_text().strip()}")
    history = list(run.scan_history())
    keys = {key for row in history for key in row}
    required = (
        "train/loss_recon", "train/loss_e8_owner", "train/loss_e8_reader",
        "train/memory_write_entropy", "train/memory_reader_entropy",
        "eval/loss", "eval/loss_recon", "eval/memory_reader_entropy", "eval/eval_table",
    )
    for key in required: assert key in keys, (key, sorted(keys))
    banned = (
        "loss_e8_causal", "loss_contrastive", "write_gate_mean", "latent_distill",
        "owner_gradient_scale", "allocation", "e9_gru", "e12_centroid",
        "dit_soft_routing",
    )
    assert not any(any(item in key for item in banned) for key in keys), sorted(keys)
    assert run.state == "finished", run.state
    print(f"W&B server verified: {run.path}; deleting temporary smoke run")
    run.delete(delete_artifacts=True)
elif os.environ["WANDB_MODE"] != "offline":
    raise AssertionError("Smoke requires online or offline W&B logging")
PY

case "${SMOKE_ROOT}" in
    "${PROJECT_ROOT}/.smoke_oneshot_memory8_register32_noprior."*) ;;
    *) exit 2 ;;
esac
"${PYTHON}" - <<'PY'
import os
import shutil
from pathlib import Path

path = Path(os.environ["SMOKE_ROOT"])
assert path.name.startswith(".smoke_oneshot_memory8_register32_noprior.")
shutil.rmtree(path)
assert not path.exists()
PY
trap - EXIT
echo "PASS: 8-object/32-register no-prior training, in-train eval, save/reload, filtered W&B, sequential TF/AR rFID+KID/count verified; smoke artifacts removed."
