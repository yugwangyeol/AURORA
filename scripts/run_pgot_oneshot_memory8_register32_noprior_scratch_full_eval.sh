#!/usr/bin/env bash
# GPU 0,1: cold-start 1.5B training, then COCO and VOC/RefCOCO evaluation.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
IFS=, read -r -a DEVICES <<< "${CUDA_VISIBLE_DEVICES}"
if (( ${#DEVICES[@]} != 2 )) || [[ "${DEVICES[0]}" == "${DEVICES[1]}" ]]; then
    echo "CUDA_VISIBLE_DEVICES must contain exactly two distinct GPUs." >&2
    exit 2
fi

export MAX_STEPS="${MAX_STEPS:-5000}"
export OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/checkpoints/pgot_oneshot_memory8_register32_noprior_scratch}"
export EVAL_ROOT="${EVAL_ROOT:-${PROJECT_ROOT}/outputs/eval_pgot_oneshot_memory8_register32_noprior_scratch}"
if [[ -e "${OUTPUT_DIR}/config.json" || \
      -n "$(find "${OUTPUT_DIR}" -maxdepth 1 -type d -name 'checkpoint-*' -print -quit 2>/dev/null || true)" || \
      -s "${EVAL_ROOT}/train.log" ]]; then
    echo "Existing training output detected; choose fresh OUTPUT_DIR and EVAL_ROOT." >&2
    exit 2
fi

mkdir -p "${EVAL_ROOT}/logs"
bash "${PROJECT_ROOT}/scripts/train_pgot_oneshot_memory8_register32_noprior_scratch.sh" \
    2>&1 | tee "${EVAL_ROOT}/train.log"

export MODEL_PATH="${OUTPUT_DIR}/checkpoint-${MAX_STEPS}"
test -f "${MODEL_PATH}/config.json" || {
    echo "Training finished without expected checkpoint: ${MODEL_PATH}" >&2
    exit 1
}
AUDIT_PYTHON="${AUDIT_PYTHON:-/home/jovyan/.conda/envs/scale_rae/bin/python}"
PYTHONNOUSERSITE=1 PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}" \
LD_LIBRARY_PATH="$(dirname "${AUDIT_PYTHON}")/../lib:${LD_LIBRARY_PATH:-}" \
"${AUDIT_PYTHON}" -m pgot.eval.verify_oneshot_o8r32_config \
    --model_path "${MODEL_PATH}" --model_size 1.5b --require_scratch_init

# A failed command in either branch is collected below; VOC must finish before
# RefCOCO starts on the same physical GPU.
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
    MODEL_PATH="${MODEL_PATH}" OUTPUT_ROOT="${EVAL_ROOT}" BATCH_SIZE=1 \
    bash "${PROJECT_ROOT}/scripts/eval_pgot_memory8_full_coco_voc_ar.sh" voc \
    2>&1 | tee "${EVAL_ROOT}/logs/voc_ar.log"
    CUDA_VISIBLE_DEVICES="${DEVICES[1]}" GPU="${DEVICES[1]}" \
    MODEL_PATH="${MODEL_PATH}" OUTPUT_ROOT="${EVAL_ROOT}/refcoco" \
    bash "${PROJECT_ROOT}/scripts/eval_deor_ctrlo_refcoco.sh" \
    2>&1 | tee "${EVAL_ROOT}/logs/refcoco.log"
) &
pids+=("$!")
names+=("VOC_AR_THEN_REFCOCO")

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
    "${EVAL_ROOT}/voc_ar/summary.json" \
    "${EVAL_ROOT}/refcoco/summary.json"; do
    test -s "${summary}" || { echo "Missing evaluation summary: ${summary}" >&2; exit 1; }
done
echo "Completed 1.5B scratch training and all evaluations: ${EVAL_ROOT}"
