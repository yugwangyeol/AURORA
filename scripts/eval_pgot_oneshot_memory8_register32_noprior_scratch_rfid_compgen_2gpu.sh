#!/usr/bin/env bash
# Evaluate the scratch O8/R32 checkpoint: AR+rFID on one GPU and CompFID on another.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2,3}"
IFS=, read -r -a DEVICES <<< "${CUDA_VISIBLE_DEVICES}"
if (( ${#DEVICES[@]} != 2 )) || [[ "${DEVICES[0]}" == "${DEVICES[1]}" ]]; then
    echo "CUDA_VISIBLE_DEVICES must contain exactly two distinct GPUs." >&2
    exit 2
fi

export MODEL_PATH="${MODEL_PATH:-${PROJECT_ROOT}/checkpoints/pgot_oneshot_memory8_register32_noprior_scratch/checkpoint-${MAX_STEPS:-5000}}"
export EVAL_ROOT="${EVAL_ROOT:-${PROJECT_ROOT}/outputs/eval_pgot_oneshot_memory8_register32_noprior_scratch}"
PYTHON="${PGOT_PYTHON:-/home/jovyan/.conda/envs/scale_rae/bin/python}"

test -f "${MODEL_PATH}/config.json" || {
    echo "Missing trained checkpoint: ${MODEL_PATH}" >&2
    exit 1
}
if [[ "${FORCE_OVERWRITE:-0}" != 1 ]] && \
   { [[ -s "${EVAL_ROOT}/ar/summary.json" ]] || [[ -s "${EVAL_ROOT}/compgen/summary.json" ]]; }; then
    echo "Existing summary detected under ${EVAL_ROOT}; set FORCE_OVERWRITE=1 to rerun." >&2
    exit 2
fi

PYTHONNOUSERSITE=1 PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}" \
LD_LIBRARY_PATH="$(dirname "${PYTHON}")/../lib:${LD_LIBRARY_PATH:-}" \
"${PYTHON}" -m pgot.eval.verify_oneshot_memory_config \
    --model_path "${MODEL_PATH}" --model_size 1.5b \
    --object_memories 8 --register_memories 32 --require_scratch_init \
    --caption_init true --own_caption_attention true

mkdir -p "${EVAL_ROOT}/logs"

# Keep the same 4,720-image AR+rFID protocol used by the existing memory sweep.
(
    CUDA_VISIBLE_DEVICES="${DEVICES[0]}" \
    MODEL_PATH="${MODEL_PATH}" OUTPUT_DIR="${EVAL_ROOT}/ar" \
    COMPUTE_RFID=True COMPUTE_KID=True AR_BATCH_SIZE="${AR_BATCH_SIZE:-4}" \
    bash "${PROJECT_ROOT}/scripts/eval_pgot_oneshot_memory8_register32_noprior_ar.sh" \
    2>&1 | tee "${EVAL_ROOT}/logs/ar_rfid.log"
) &
pids=("$!")
names=("AR+rFID")

# Match the scratch O1/O4 compositional protocol: full 5K, random-slot mixing,
# random register source, seed 1234, guidance 1.0, and 10 diffusion steps.
(
    CUDA_VISIBLE_DEVICES="${DEVICES[1]}" \
    MODEL_PATH="${MODEL_PATH}" OUTPUT_DIR="${EVAL_ROOT}/compgen" \
    DTYPE="${DTYPE:-fp32}" BATCH_SIZE="${COMP_BATCH_SIZE:-4}" \
    EXPECTED_SAMPLES="${EXPECTED_SAMPLES:-5000}" METRIC_RESOLUTION="${METRIC_RESOLUTION:-0}" \
    MIX_MODE="${MIX_MODE:-random_slots}" REGISTER_SOURCE="${REGISTER_SOURCE:-random}" \
    SEED="${SEED:-1234}" GUIDANCE_SCALE="${GUIDANCE_SCALE:-1.0}" \
    DIFFUSION_INFERENCE_STEPS="${DIFFUSION_INFERENCE_STEPS:-10}" \
    bash "${PROJECT_ROOT}/scripts/eval_pgot_compositional_gen.sh" \
    2>&1 | tee "${EVAL_ROOT}/logs/compgen.log"
) &
pids+=("$!")
names+=("Compositional FID")

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
    "RESULT O8/R32 scratch "
    f"FG-ARI={ar.get('fARI')}, mBO={ar.get('mBO')}, "
    f"mIoU={ar.get('mIoU')}, rFID={ar.get('rFID')}, "
    f"CountMAE={ar.get('ar_object_count_mae')}, "
    f"CompFID={comp.get('FID')}, CompKIDx1e3={comp.get('KID_x1e3_mean')}"
)
PY

echo "Completed O8/R32 scratch rFID + compositional FID: ${EVAL_ROOT}"
