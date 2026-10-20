#!/usr/bin/env bash
# Re-evaluate the non-scratch O8/R32 Init-on/Attn-on description-path baseline.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2,3}"
IFS=, read -r -a DEVICES <<< "${CUDA_VISIBLE_DEVICES}"
if (( ${#DEVICES[@]} != 2 )) || [[ "${DEVICES[0]}" == "${DEVICES[1]}" ]]; then
    echo "CUDA_VISIBLE_DEVICES must contain exactly two distinct GPUs." >&2
    exit 2
fi

export MODEL_PATH="${MODEL_PATH:-${PROJECT_ROOT}/checkpoints/pgot_oneshot_memory8_register32_noprior/checkpoint-5000}"
export EVAL_ROOT="${EVAL_ROOT:-${PROJECT_ROOT}/outputs/eval_pgot_description_path_memory8_register32_noprior_nonscratch}"
PYTHON="${PGOT_PYTHON:-/home/jovyan/.conda/envs/scale_rae/bin/python}"

test -f "${MODEL_PATH}/config.json" || {
    echo "Missing checkpoint: ${MODEL_PATH}" >&2
    exit 1
}
if [[ "${FORCE_OVERWRITE:-0}" != 1 ]] && \
   { [[ -s "${EVAL_ROOT}/coco_ar/summary.json" ]] || \
     [[ -s "${EVAL_ROOT}/refcoco/summary.json" ]] || \
     [[ -s "${EVAL_ROOT}/refcoco_target_hit/summary.json" ]]; }; then
    echo "Existing summary detected under ${EVAL_ROOT}; set FORCE_OVERWRITE=1 to rerun." >&2
    exit 2
fi

PYTHONNOUSERSITE=1 PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}" \
LD_LIBRARY_PATH="$(dirname "${PYTHON}")/../lib:${LD_LIBRARY_PATH:-}" \
"${PYTHON}" -m pgot.eval.verify_oneshot_memory_config \
    --model_path "${MODEL_PATH}" --model_size 1.5b \
    --object_memories 8 --register_memories 32 \
    --caption_init true --own_caption_attention true

# Prevent this baseline runner from silently accepting the cold-start checkpoint.
"${PYTHON}" - "${MODEL_PATH}/config.json" <<'PY'
import json
import sys

config = json.load(open(sys.argv[1]))
if config.get("pgot_one_shot_scratch_init", False):
    raise SystemExit("Expected the non-scratch O8/R32 checkpoint, but scratch init is enabled")
print("PASS initialization audit: non-scratch checkpoint, Init=on, Attn=on")
PY

mkdir -p "${EVAL_ROOT}/logs"

# GPU 2: image-only AR on all 5,000 COCO validation images, including GT-empty images.
(
    CUDA_VISIBLE_DEVICES="${DEVICES[0]}" GPU="${DEVICES[0]}" \
    MODEL_PATH="${MODEL_PATH}" OUTPUT_ROOT="${EVAL_ROOT}" \
    BATCH_SIZE="${COCO_BATCH_SIZE:-4}" COMPUTE_RFID=False COMPUTE_KID=False \
    bash "${PROJECT_ROOT}/scripts/eval_pgot_memory8_full_coco_voc_ar.sh" coco \
    2>&1 | tee "${EVAL_ROOT}/logs/coco_ar.log"
) &
pids=("$!")
names=("COCO_AR_5000")

# GPU 3: RefCOCO expression grounding, followed by CPU-only TargetHit analysis.
(
    CUDA_VISIBLE_DEVICES="${DEVICES[1]}" GPU="${DEVICES[1]}" \
    MODEL_PATH="${MODEL_PATH}" OUTPUT_ROOT="${EVAL_ROOT}/refcoco" \
    bash "${PROJECT_ROOT}/scripts/eval_deor_refcoco_ours_only.sh" \
    2>&1 | tee "${EVAL_ROOT}/logs/refcoco.log"

    PYTHONNOUSERSITE=1 PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}" \
    LD_LIBRARY_PATH="$(dirname "${PYTHON}")/../lib:${LD_LIBRARY_PATH:-}" \
    "${PYTHON}" "${PROJECT_ROOT}/scripts/analyze_refcoco_cached_grounding.py" \
        --root "${EVAL_ROOT}/refcoco" \
        --output "${EVAL_ROOT}/refcoco_target_hit" \
        --models deor \
        2>&1 | tee "${EVAL_ROOT}/logs/refcoco_target_hit.log"
) &
pids+=("$!")
names+=("REFCOCO_MIOU_TARGET_HIT")

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

for summary in \
    "${EVAL_ROOT}/coco_ar/summary.json" \
    "${EVAL_ROOT}/refcoco/summary.json" \
    "${EVAL_ROOT}/refcoco_target_hit/summary.json"; do
    test -s "${summary}" || { echo "Missing evaluation summary: ${summary}" >&2; exit 1; }
done

"${PYTHON}" - "${EVAL_ROOT}" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
coco = json.loads((root / "coco_ar/summary.json").read_text())
ref = json.loads((root / "refcoco_target_hit/summary.json").read_text())
for subset, prefix in (("nonoverlap", "TABLE_RESULT"), ("full", "FULL_RESULT")):
    score = ref["subsets"][subset]["all"]["deor"]
    print(
        f"{prefix} subset={subset} Init=on Attn=on nonscratch "
        f"COCO_FGARI={100*coco['fARI']:.2f} "
        f"COCO_mIoUi={100*coco['mIoU']:.2f} "
        f"RefCOCO_mIoU={score['mIoU']:.2f} "
        f"RefCOCO_TargetHit@1={score['target_hit_at_1']:.2f} "
        f"SameCat_TargetHit@1={score['same_category_target_hit_at_1']:.2f} "
        f"SameCat_n={score['same_category_expressions']}"
    )
PY

echo "Completed non-scratch Init-on / Attn-on baseline: ${EVAL_ROOT}"
