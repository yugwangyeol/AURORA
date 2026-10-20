#!/usr/bin/env bash
# Audit once, then run AR+rFID and compositional FID concurrently on two GPUs.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
IFS=, read -r -a DEVICES <<< "${CUDA_VISIBLE_DEVICES}"
if (( ${#DEVICES[@]} != 2 )); then
    echo "Set CUDA_VISIBLE_DEVICES to exactly two GPU indices (for example 0,1)." >&2
    exit 2
fi

export MODEL_PATH="${MODEL_PATH:-${PROJECT_ROOT}/checkpoints/pgot_oneshot_memory4_register32_noprior_scratch/checkpoint-${MAX_STEPS:-5000}}"
export EVAL_ROOT="${EVAL_ROOT:-${PROJECT_ROOT}/outputs/eval_pgot_oneshot_memory4_register32_noprior_scratch}"
PYTHON="${PYTHON:-/home/jovyan/.conda/envs/scale_rae/bin/python}"

test -f "${MODEL_PATH}/config.json" || {
    echo "Missing trained checkpoint: ${MODEL_PATH}" >&2
    exit 1
}
PYTHONNOUSERSITE=1 PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}" \
LD_LIBRARY_PATH="$(dirname "${PYTHON}")/../lib:${LD_LIBRARY_PATH:-}" \
"${PYTHON}" -m pgot.eval.verify_oneshot_memory_config \
    --model_path "${MODEL_PATH}" --model_size 1.5b \
    --object_memories 4 --register_memories 32 --require_scratch_init

mkdir -p "${EVAL_ROOT}/logs"

(
    CUDA_VISIBLE_DEVICES="${DEVICES[0]}" \
    MODEL_PATH="${MODEL_PATH}" OUTPUT_DIR="${EVAL_ROOT}/ar" \
    COMPUTE_RFID=True COMPUTE_KID=True \
    bash "${PROJECT_ROOT}/scripts/eval_pgot_oneshot_memory4_register32_noprior_scratch_ar.sh" \
    2>&1 | tee "${EVAL_ROOT}/logs/ar_rfid.log"
) &
pids=("$!")
names=("AR+rFID")

(
    CUDA_VISIBLE_DEVICES="${DEVICES[1]}" \
    MODEL_PATH="${MODEL_PATH}" OUTPUT_DIR="${EVAL_ROOT}/compgen" \
    bash "${PROJECT_ROOT}/scripts/eval_pgot_oneshot_memory4_register32_noprior_scratch_compgen.sh" \
    2>&1 | tee "${EVAL_ROOT}/logs/compgen.log"
) &
pids+=("$!")
names+=("Compositional")

failed=0
for index in "${!pids[@]}"; do
    if wait "${pids[$index]}"; then
        echo "PASS ${names[$index]}"
    else
        status=$?
        echo "FAIL ${names[$index]} status=${status}" >&2
        failed=1
    fi
done
(( failed == 0 )) || exit 1

for summary in "${EVAL_ROOT}/ar/summary.json" "${EVAL_ROOT}/compgen/summary.json"; do
    test -s "${summary}" || {
        echo "Missing evaluation summary: ${summary}" >&2
        exit 1
    }
done

"${PYTHON}" - "${EVAL_ROOT}" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
ar = json.loads((root / "ar/summary.json").read_text())
comp = json.loads((root / "compgen/summary.json").read_text())
print(
    "RESULT "
    f"FG-ARI={ar.get('fARI')}, mBO={ar.get('mBO')}, "
    f"mIoU={ar.get('mIoU')}, rFID={ar.get('rFID')}, "
    f"CountMAE={ar.get('ar_object_count_mae')}, "
    f"CompFID={comp.get('FID')}, CompKIDx1e3={comp.get('KID_x1e3_mean')}"
)
PY

echo "Completed two-GPU evaluation: ${EVAL_ROOT}"
