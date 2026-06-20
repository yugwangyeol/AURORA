#!/bin/bash
# Run the checkpoint-only information-path diagnostic on two GPUs in parallel.
set -euo pipefail

PROJECT_ROOT="/home/jovyan/PGOT"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/outputs/information_path_diagnostic_no_swap}"
SAMPLE_INDICES="${SAMPLE_INDICES:-0,1020,1407}"
LAYERS="${LAYERS:-last4}"
TOP_REGISTERS="${TOP_REGISTERS:-5}"
DIFFUSION_INFERENCE_STEPS="${DIFFUSION_INFERENCE_STEPS:-25}"
GUIDANCE_SCALE="${GUIDANCE_SCALE:-1.0}"
DTYPE="${DTYPE:-fp32}"
PYTHON="${HOME}/.conda/envs/scale_rae/bin/python"
CONDA_LIB="${HOME}/.conda/envs/scale_rae/lib"

mkdir -p "${OUTPUT_DIR}/gpu0" "${OUTPUT_DIR}/gpu1"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export PYTHONNOUSERSITE=1
export LD_LIBRARY_PATH="${CONDA_LIB}:${LD_LIBRARY_PATH:-}"

COMMON_ARGS=(
    --sample_indices "${SAMPLE_INDICES}"
    --layers "${LAYERS}"
    --top_registers "${TOP_REGISTERS}"
    --compute_gradients
    --decode_recon
    --diffusion_inference_steps "${DIFFUSION_INFERENCE_STEPS}"
    --guidance_scale "${GUIDANCE_SCALE}"
    --dtype "${DTYPE}"
)

echo "GPU 0: V3, V8.2"
CUDA_VISIBLE_DEVICES=0 "${PYTHON}" -m pgot.eval.diagnose_information_paths \
    --model "V3|${PROJECT_ROOT}/checkpoints/pgot_main_v3/checkpoint-14000" \
    --model "V8.2|${PROJECT_ROOT}/checkpoints/pgot_main_v8_2_spatial_outside_log" \
    --output_dir "${OUTPUT_DIR}/gpu0" \
    "${COMMON_ARGS[@]}" \
    > "${OUTPUT_DIR}/gpu0/run.log" 2>&1 &
PID0=$!

echo "GPU 1: V8.3, V11"
CUDA_VISIBLE_DEVICES=1 "${PYTHON}" -m pgot.eval.diagnose_information_paths \
    --model "V8.3|${PROJECT_ROOT}/checkpoints/pgot_main_v8_3_bce1_outlog1" \
    --model "V11|${PROJECT_ROOT}/checkpoints/pgot_main_v11_bal_bce_ce03_mean" \
    --output_dir "${OUTPUT_DIR}/gpu1" \
    "${COMMON_ARGS[@]}" \
    > "${OUTPUT_DIR}/gpu1/run.log" 2>&1 &
PID1=$!

status=0
wait "${PID0}" || status=$?
wait "${PID1}" || status=$?
if [ "${status}" -ne 0 ]; then
    echo "Diagnostic failed. Check ${OUTPUT_DIR}/gpu0/run.log and gpu1/run.log." >&2
    exit "${status}"
fi

"${PYTHON}" - "${OUTPUT_DIR}" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
merged = {}
for worker in ("gpu0", "gpu1"):
    with (root / worker / "summary.json").open() as f:
        merged.update(json.load(f))
with (root / "summary.json").open("w") as f:
    json.dump(merged, f, indent=2)
print(root / "summary.json")
PY

echo "Completed: ${OUTPUT_DIR}"
