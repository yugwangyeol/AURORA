#!/usr/bin/env bash
# Real two-GPU FP32 train/eval/save/W&B, then simultaneous TF/AR checkpoint reloads.
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-/home/jovyan/.conda/envs/scale_rae/bin/python}"
SMOKE_ROOT="$(mktemp -d "${PROJECT_ROOT}/.smoke_oneshot_memory4.XXXXXX")"
BASE_MODEL="${MODEL_PATH:-${PROJECT_ROOT}/checkpoints/pgot_e11_capacity_dit16/checkpoint-10000}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export PYTHONNOUSERSITE=1
export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_PROJECT="${WANDB_PROJECT:-PGOT}"
export WANDB_DATA_DIR="${SMOKE_ROOT}/wandb_data"
export WANDB_CACHE_DIR="${SMOKE_ROOT}/wandb_cache"
export WANDB_RESUME=never
export SMOKE_ROOT
echo "Smoke workspace: ${SMOKE_ROOT}"
# Preserve failures for diagnosis. On success delete only the runs created here.
on_exit() {
    local status=$?
    if (( status )); then echo "Smoke failed; preserved at ${SMOKE_ROOT}" >&2; fi
}
trap on_exit EXIT
for mode in ${SMOKE_MODES:-content id}; do
    run_root="${SMOKE_ROOT}/${mode}"
    mkdir -p "${run_root}/wandb"
    printf '%s\n' "${BASE_MODEL}" > "${run_root}/source.txt"
    run_id="$("${PYTHON}" -c 'import wandb; print(wandb.util.generate_id())')"
    printf '%s\n' "${run_id}" > "${run_root}/wandb_id.txt"
    WANDB_RUN_ID="${run_id}" WANDB_NAME="smoke_oneshot_memory4_${mode}" WANDB_DIR="${run_root}/wandb" \
    MEMORY_KEY_MODE="${mode}" MODEL_PATH="${BASE_MODEL}" OUTPUT_DIR="${run_root}/train" \
    NUM_GPUS=2 PER_DEVICE_TRAIN_BATCH_SIZE="${SMOKE_BATCH_SIZE:-2}" GLOBAL_BATCH_SIZE="${SMOKE_GLOBAL_BATCH_SIZE:-24}" \
    MAX_STEPS=2 SAVE_STEPS=2 SAVE_TOTAL_LIMIT=1 EVAL_STEPS=2 LOGGING_STEPS=1 \
    PER_DEVICE_EVAL_BATCH_SIZE=1 EVAL_NUM_IMAGES=2 PGOT_EVAL_LOG_RECON_IMAGES=1 \
    DATALOADER_NUM_WORKERS=1 REPORT_TO=wandb PGOT_SKIP_FINAL_SAVE=1 \
    MASTER_PORT="${MASTER_PORT:-29643}" \
        bash "${PROJECT_ROOT}/scripts/train_pgot_oneshot_memory4.sh" \
        > "${run_root}/train.log" 2>&1
    "${PYTHON}" "${PROJECT_ROOT}/scripts/verify_oneshot_memory4_smoke.py" "${SMOKE_ROOT}" "${mode}" --stage train
    MEMORY_KEY_MODE="${mode}" MODEL_PATH="${run_root}/train/checkpoint-2" EVAL_ROOT="${run_root}/eval" \
    MAX_SAMPLES=2 BATCH_SIZE=1 AR_BATCH_SIZE=1 NUM_WORKERS=1 \
    DTYPE=fp32 DIFFUSION_INFERENCE_STEPS=2 COMPUTE_RFID=True AR_MAX_NEW_TOKENS=128 \
        bash "${PROJECT_ROOT}/scripts/eval_pgot_oneshot_memory4_parallel.sh"
    "${PYTHON}" "${PROJECT_ROOT}/scripts/verify_oneshot_memory4_smoke.py" "${SMOKE_ROOT}" "${mode}" --stage eval
done
"${PYTHON}" - <<'PY'
import json, os
from pathlib import Path
import wandb
root = Path(os.environ["SMOKE_ROOT"])
if os.environ["WANDB_MODE"] == "online":
    api = wandb.Api(timeout=30)
    entity = os.environ.get("WANDB_ENTITY") or api.default_entity
    for marker in root.glob("*/wandb_id.txt"):
        run = api.run(f"{entity}/{os.environ['WANDB_PROJECT']}/{marker.read_text().strip()}")
        history = list(run.scan_history())
        keys = {k for row in history for k in row}
        for key in ("train/loss_recon", "train/memory_reader_entropy", "eval/loss", "eval/loss_recon", "eval/eval_table"):
            assert key in keys, (run.id, key, sorted(keys))
        assert not any("loss_e8_causal" in k or "loss_contrastive" in k or "write_gate_mean" in k for k in keys)
        assert run.state == "finished", (run.id, run.state)
        print(f"W&B server verified: {run.path}; deleting temporary smoke run")
        run.delete(delete_artifacts=True)
elif os.environ["WANDB_MODE"] != "offline":
    raise AssertionError("Smoke requires online or offline W&B logging")
PY
case "${SMOKE_ROOT}" in "${PROJECT_ROOT}/.smoke_oneshot_memory4."*) ;; *) exit 2;; esac
"${PYTHON}" - <<'PY'
import os, shutil
from pathlib import Path
p = Path(os.environ["SMOKE_ROOT"])
assert p.name.startswith('.smoke_oneshot_memory4.')
shutil.rmtree(p)
assert not p.exists()
PY
trap - EXIT
echo "PASS: requested memory modes trained, evaluated, saved and logged; smoke outputs removed."
