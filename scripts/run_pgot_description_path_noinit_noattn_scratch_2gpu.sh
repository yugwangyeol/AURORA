#!/usr/bin/env bash
# Train Init-off/Attn-off O8/R32 from scratch, then evaluate COCO and RefCOCO.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
IFS=, read -r -a DEVICES <<< "${CUDA_VISIBLE_DEVICES}"
if (( ${#DEVICES[@]} != 2 )) || [[ "${DEVICES[0]}" == "${DEVICES[1]}" ]]; then
    echo "CUDA_VISIBLE_DEVICES must contain exactly two distinct GPUs." >&2
    exit 2
fi

export MAX_STEPS="${MAX_STEPS:-5000}"
export OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/checkpoints/pgot_oneshot_memory8_register32_noprior_scratch_noinit_noattn}"
export EVAL_ROOT="${EVAL_ROOT:-${PROJECT_ROOT}/outputs/eval_pgot_oneshot_memory8_register32_noprior_scratch_noinit_noattn}"
if [[ -e "${OUTPUT_DIR}/config.json" || \
      -n "$(find "${OUTPUT_DIR}" -maxdepth 1 -type d -name 'checkpoint-*' -print -quit 2>/dev/null || true)" || \
      -s "${EVAL_ROOT}/logs/train.log" ]]; then
    echo "Existing training output detected; choose fresh OUTPUT_DIR and EVAL_ROOT." >&2
    exit 2
fi

mkdir -p "${EVAL_ROOT}/logs"
bash "${PROJECT_ROOT}/scripts/train_pgot_oneshot_memory8_register32_noprior_scratch_noinit_noattn.sh" \
    2>&1 | tee "${EVAL_ROOT}/logs/train.log"

export MODEL_PATH="${OUTPUT_DIR}/checkpoint-${MAX_STEPS}"
test -f "${MODEL_PATH}/config.json" || {
    echo "Training finished without expected checkpoint: ${MODEL_PATH}" >&2
    exit 1
}
PYTHON="${PGOT_PYTHON:-/home/jovyan/.conda/envs/scale_rae/bin/python}"
PYTHONNOUSERSITE=1 PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}" \
LD_LIBRARY_PATH="$(dirname "${PYTHON}")/../lib:${LD_LIBRARY_PATH:-}" \
"${PYTHON}" -m pgot.eval.verify_oneshot_memory_config \
    --model_path "${MODEL_PATH}" --model_size 1.5b \
    --object_memories 8 --register_memories 32 --require_scratch_init \
    --caption_init false --own_caption_attention false

# After training, evaluate the two table branches in parallel: GPU 0 handles
# the 5k COCO AR split and GPU 1 handles RefCOCO. TargetHit is computed from
# the cached RefCOCO masks without another model forward pass.
(
    CUDA_VISIBLE_DEVICES="${DEVICES[0]}" GPU="${DEVICES[0]}" \
    MODEL_PATH="${MODEL_PATH}" OUTPUT_ROOT="${EVAL_ROOT}" \
    BATCH_SIZE="${COCO_BATCH_SIZE:-4}" \
    bash "${PROJECT_ROOT}/scripts/eval_pgot_memory8_full_coco_voc_ar.sh" coco \
    2>&1 | tee "${EVAL_ROOT}/logs/coco_ar.log"
) &
pids=("$!")
names=("COCO_AR_5000")

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
nonoverlap = ref["subsets"]["nonoverlap"]["all"]["deor"]
full = ref["subsets"]["full"]["all"]["deor"]
print(
    "TABLE_RESULT subset=nonoverlap Init=off Attn=off "
    f"COCO_FGARI={100*coco['fARI']:.2f} "
    f"COCO_mIoUi={100*coco['mIoU']:.2f} "
    f"RefCOCO_mIoU={nonoverlap['mIoU']:.2f} "
    f"RefCOCO_TargetHit@1={nonoverlap['target_hit_at_1']:.2f} "
    f"SameCat_TargetHit@1={nonoverlap['same_category_target_hit_at_1']:.2f} "
    f"SameCat_n={nonoverlap['same_category_expressions']}"
)
print(
    "FULL_RESULT subset=full Init=off Attn=off "
    f"RefCOCO_mIoU={full['mIoU']:.2f} "
    f"RefCOCO_TargetHit@1={full['target_hit_at_1']:.2f} "
    f"SameCat_TargetHit@1={full['same_category_target_hit_at_1']:.2f} "
    f"SameCat_n={full['same_category_expressions']}"
)
PY

echo "Completed Init-off / Attn-off description-path ablation: ${EVAL_ROOT}"
