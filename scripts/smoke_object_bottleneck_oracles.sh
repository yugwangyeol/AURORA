#!/bin/bash
# Exercise the frozen control and all trainable source paths, then remove artifacts.
set -euo pipefail

MODEL_PATH="${MODEL_PATH:-/home/jovyan/PGOT/checkpoints/pgot_main_v15_1_coda_mmproj}"
OUTPUT_DIR="${OUTPUT_DIR:-/home/jovyan/PGOT/outputs/smoke_object_bottleneck_oracles_v2}"
GPU="${GPU:-0}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON="${HOME}/.conda/envs/scale_rae/bin/python"
CONDA_LIB="$(dirname "${PYTHON}")/../lib"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export PYTHONNOUSERSITE=1
export LD_LIBRARY_PATH="${CONDA_LIB}:${LD_LIBRARY_PATH:-}"

rm -rf "${OUTPUT_DIR}"
mkdir -p "${OUTPUT_DIR}"
cleanup() { rm -rf "${OUTPUT_DIR}"; }
trap cleanup EXIT

run_smoke() {
    local oracle="$1"
    local object_tokens="$2"
    local train_steps="$3"
    local distill_steps="$4"
    local tag="${oracle}"
    if [ "${oracle}" = "c_gtobj" ]; then
        tag="${oracle}_k${object_tokens}"
    fi

    local checkpoint_args=()
    if [ "${oracle}" = "c_current" ]; then
        checkpoint_args+=(--no_save_checkpoint)
    fi

    CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON}" -m pgot.eval.train_object_bottleneck_oracle \
        --model_path "${MODEL_PATH}" \
        --train_jsonl "${PROJECT_ROOT}/data/pgot_train.jsonl" \
        --val_jsonl "${PROJECT_ROOT}/data/pgot_val.jsonl" \
        --output_dir "${OUTPUT_DIR}/${tag}" \
        --oracle "${oracle}" \
        --object_tokens "${object_tokens}" \
        --batch_size 1 \
        --num_workers 0 \
        --max_train_samples 2 \
        --max_samples 1 \
        --train_steps "${train_steps}" \
        --distill_steps "${distill_steps}" \
        --gradient_accumulation_steps 1 \
        --dit_last_n_blocks 0 \
        --adapter_depth 1 \
        --diffusion_loss_weight 0.1 \
        --cfg_drop_rate 0.1 \
        --diffusion_inference_steps 2 \
        --guidance_scale 1.0 \
        --dtype fp32 \
        --skip_fid \
        --log_every 1 \
        "${checkpoint_args[@]}" \
        2>&1 | tee "${OUTPUT_DIR}/${tag}.log"

    test -f "${OUTPUT_DIR}/${tag}/summary.json"
    if [ "${oracle}" != "c_current" ]; then
        test -f "${OUTPUT_DIR}/${tag}/oracle_checkpoint.pt"
    fi
    grep -q "\"oracle\": \"${oracle}\"" "${OUTPUT_DIR}/${tag}/summary.json"
    grep -q '"num_samples": 1' "${OUTPUT_DIR}/${tag}/summary.json"
    grep -q 'RESULT' "${OUTPUT_DIR}/${tag}.log"
}

echo "===== Object Bottleneck Oracle v2 SMOKE on GPU ${GPU} ====="
run_smoke c_current 4 0 0
run_smoke c_full 4 2 1
run_smoke c_gtobj 2 1 1
run_smoke c_ovt 4 1 1

"${PYTHON}" - "${OUTPUT_DIR}/c_full/summary.json" <<'PY'
import json
import sys
with open(sys.argv[1]) as handle:
    summary = json.load(handle)
assert summary["last_distill_loss"] is not None
assert summary["last_diffusion_loss"] is not None
assert summary["dit_last_n_blocks"] == 0
PY

rm -rf "${OUTPUT_DIR}"
test ! -e "${OUTPUT_DIR}"
trap - EXIT
echo "Smoke complete; smoke artifacts removed."
