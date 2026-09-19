#!/usr/bin/env bash
# Run COCO AR, VOC AR, and compositional evaluation in parallel after auditing.
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
IFS=, read -r -a DEVICES <<< "${CUDA_VISIBLE_DEVICES}"
if (( ${#DEVICES[@]} != 4 )); then
    echo 'Set CUDA_VISIBLE_DEVICES to exactly four GPUs.' >&2; exit 2
fi
export MODEL_PATH="${MODEL_PATH:-${PROJECT_ROOT}/checkpoints/pgot_oneshot_memory8_register32_noprior_7b_fp32_fast_woreader/checkpoint-${MAX_STEPS:-5000}}"
export EVAL_ROOT="${EVAL_ROOT:-${PROJECT_ROOT}/outputs/eval_pgot_oneshot_memory8_register32_noprior_7b_fp32_fast_woreader}"
test -f "${MODEL_PATH}/config.json"
AUDIT_PYTHON="${AUDIT_PYTHON:-/home/jovyan/.conda/envs/scale_rae/bin/python}"
PYTHONNOUSERSITE=1 PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}" \
LD_LIBRARY_PATH="$(dirname "${AUDIT_PYTHON}")/../lib:${LD_LIBRARY_PATH:-}" \
"${AUDIT_PYTHON}" -m pgot.eval.verify_woreader_config \
    --model_path "${MODEL_PATH}" --model_size 7b
mkdir -p "${EVAL_ROOT}/logs"

( CUDA_VISIBLE_DEVICES="${DEVICES[0]}" MODEL_PATH="${MODEL_PATH}" \
  OUTPUT_DIR="${EVAL_ROOT}/ar" \
  bash "${PROJECT_ROOT}/scripts/eval_pgot_oneshot_memory8_register32_noprior_7b_fp32_fast_woreader_ar.sh" \
  2>&1 | tee "${EVAL_ROOT}/logs/ar.log" ) &
pids=("$!"); names=("COCO_AR")
( CUDA_VISIBLE_DEVICES="${DEVICES[1]}" GPU="${DEVICES[1]}" MODEL_PATH="${MODEL_PATH}" \
  OUTPUT_ROOT="${EVAL_ROOT}" \
  bash "${PROJECT_ROOT}/scripts/eval_pgot_oneshot_memory8_register32_noprior_7b_fp32_fast_woreader_voc_ar.sh" \
  2>&1 | tee "${EVAL_ROOT}/logs/voc_ar.log" ) &
pids+=("$!"); names+=("VOC_AR")
( CUDA_VISIBLE_DEVICES="${DEVICES[2]}" MODEL_PATH="${MODEL_PATH}" \
  OUTPUT_DIR="${EVAL_ROOT}/compgen" \
  bash "${PROJECT_ROOT}/scripts/eval_pgot_oneshot_memory8_register32_noprior_7b_fp32_fast_woreader_compgen.sh" \
  2>&1 | tee "${EVAL_ROOT}/logs/compgen.log" ) &
pids+=("$!"); names+=("COMPGEN")

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
for summary in "${EVAL_ROOT}/ar/summary.json" \
               "${EVAL_ROOT}/voc_ar/summary.json" \
               "${EVAL_ROOT}/compgen/summary.json"; do
    test -s "${summary}" || { echo "Missing evaluation summary: ${summary}" >&2; exit 1; }
done
echo "Completed parallel evaluations: ${EVAL_ROOT}"
