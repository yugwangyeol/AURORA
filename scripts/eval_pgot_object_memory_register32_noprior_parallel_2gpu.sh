#!/usr/bin/env bash
# Evaluate one non-scratch O{N}/R32 checkpoint: AR+rFID and CompFID in parallel.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
: "${OBJECT_MEMORIES:?Set OBJECT_MEMORIES to the trained object-memory count}"
: "${MODEL_PATH:?Set MODEL_PATH to the trained checkpoint}"
: "${EVAL_ROOT:?Set EVAL_ROOT for evaluation outputs}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
IFS=, read -r -a DEVICES <<< "${CUDA_VISIBLE_DEVICES}"
if (( ${#DEVICES[@]} != 2 )) || [[ "${DEVICES[0]}" == "${DEVICES[1]}" ]]; then
    echo "CUDA_VISIBLE_DEVICES must contain exactly two distinct GPU indices." >&2
    exit 2
fi
if [[ ! "${OBJECT_MEMORIES}" =~ ^[1-9][0-9]*$ ]]; then
    echo "OBJECT_MEMORIES must be a positive integer: ${OBJECT_MEMORIES}" >&2
    exit 2
fi

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
    --object_memories "${OBJECT_MEMORIES}" --register_memories 32 \
    --caption_init true --own_caption_attention true

mkdir -p "${EVAL_ROOT}/logs"

# Match the existing O8/R32 and O16/R32 4,720-image AR reconstruction protocol.
(
    CUDA_VISIBLE_DEVICES="${DEVICES[0]}" \
    MODEL_PATH="${MODEL_PATH}" OUTPUT_DIR="${EVAL_ROOT}/ar" \
    MEMORY_KEY_MODE=content COMPUTE_RFID=True COMPUTE_KID=True \
    COMPUTE_CLASS_METRICS=True BATCH_SIZE="${AR_BATCH_SIZE:-4}" \
    bash "${PROJECT_ROOT}/scripts/eval_pgot_oneshot_memory4.sh" \
        --caption_mode autoregressive \
        --ar_max_new_tokens "${AR_MAX_NEW_TOKENS:-512}" \
    2>&1 | tee "${EVAL_ROOT}/logs/ar_rfid.log"
) &
pids=("$!")
names=("AR+rFID")

# Match the existing O8/R32 and O16/R32 4,720-image compositional protocol.
(
    CUDA_VISIBLE_DEVICES="${DEVICES[1]}" \
    MODEL_PATH="${MODEL_PATH}" OUTPUT_DIR="${EVAL_ROOT}/compgen" \
    VAL_JSONL="${PROJECT_ROOT}/data/pgot_pix2cap_generated_val5k.jsonl" \
    EXPECTED_SAMPLES=4720 DTYPE="${DTYPE:-fp32}" \
    BATCH_SIZE="${COMP_BATCH_SIZE:-4}" METRIC_RESOLUTION="${METRIC_RESOLUTION:-0}" \
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

"${PYTHON}" - "${EVAL_ROOT}" "${OBJECT_MEMORIES}" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
memory = sys.argv[2]
ar = json.loads((root / "ar/summary.json").read_text())
comp = json.loads((root / "compgen/summary.json").read_text())
print(
    f"RESULT O{memory}/R32 non-scratch "
    f"rFID={ar.get('rFID')}, rKIDx1e3={ar.get('KID_x1000')}, "
    f"CompFID={comp.get('FID')}, CompKIDx1e3={comp.get('KID_x1e3_mean')}"
)
PY

echo "Completed O${OBJECT_MEMORIES}/R32 evaluation: ${EVAL_ROOT}"
