#!/usr/bin/env bash
# GPU 2,3: matched K=7 training, then parallel COCO/VOC AR evaluation.
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2,3}"
IFS=, read -r -a DEVICES <<< "${CUDA_VISIBLE_DEVICES}"
(( ${#DEVICES[@]} == 2 )) || { echo "Exactly two GPUs are required" >&2; exit 2; }
TRAIN_DIR="${TRAIN_DIR:-${PROJECT_ROOT}/checkpoints/pgot_fixedquery_k7_memory8_register32}"
EVAL_ROOT="${EVAL_ROOT:-${PROJECT_ROOT}/outputs/eval_pgot_fixedquery_k7_memory8_register32}"
FINAL="${TRAIN_DIR}/checkpoint-${MAX_STEPS:-5000}"
mkdir -p "${EVAL_ROOT}"

if [[ ! -f "${FINAL}/config.json" ]]; then
    if [[ -e "${TRAIN_DIR}/config.json" || -n "$(find "${TRAIN_DIR}" -maxdepth 1 -name 'checkpoint-*' -print -quit 2>/dev/null || true)" ]]; then
        echo "Partial/non-final training output exists: ${TRAIN_DIR}" >&2
        echo "Resume explicitly or choose a fresh TRAIN_DIR; refusing accidental overwrite." >&2
        exit 2
    fi
    OUTPUT_DIR="${TRAIN_DIR}" NUM_GPUS=2 MAX_STEPS="${MAX_STEPS:-5000}" \
    MASTER_PORT="${MASTER_PORT:-29772}" \
        bash "${PROJECT_ROOT}/scripts/train_pgot_fixedquery_k7_memory8_register32.sh" \
        2>&1 | tee "${EVAL_ROOT}/train.log"
fi
"/home/jovyan/.conda/envs/scale_rae/bin/python" \
    "${PROJECT_ROOT}/scripts/verify_fixedquery_checkpoint.py" "${FINAL}" --count 7

pids=(); names=()
if [[ ! -f "${EVAL_ROOT}/coco_ar/summary.json" ]]; then
    GPU="${DEVICES[0]}" STAGE=coco MODEL_PATH="${FINAL}" OUTPUT_ROOT="${EVAL_ROOT}" \
        bash "${PROJECT_ROOT}/scripts/eval_pgot_fixedquery_k7.sh" \
        > "${EVAL_ROOT}/coco_ar.console.log" 2>&1 &
    pids+=("$!"); names+=("COCO_AR")
fi
if [[ ! -f "${EVAL_ROOT}/voc_ar/summary.json" ]]; then
    GPU="${DEVICES[1]}" STAGE=voc MODEL_PATH="${FINAL}" OUTPUT_ROOT="${EVAL_ROOT}" \
        bash "${PROJECT_ROOT}/scripts/eval_pgot_fixedquery_k7.sh" \
        > "${EVAL_ROOT}/voc_ar.console.log" 2>&1 &
    pids+=("$!"); names+=("VOC_AR")
fi
status=0
for index in "${!pids[@]}"; do
    if wait "${pids[$index]}"; then echo "PASS: ${names[$index]}"; else echo "FAIL: ${names[$index]}" >&2; status=1; fi
done
(( status == 0 )) || exit "${status}"
echo "PASS: FixedQuery K=7 training and matched AR evaluations complete."
