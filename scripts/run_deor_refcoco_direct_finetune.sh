#!/usr/bin/env bash
# Train, direct-evaluate, and build full/non-overlap comparison tables.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

export OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/checkpoints/pgot_oneshot_memory8_register32_noprior_refcoco_direct}"
export MAX_STEPS="${MAX_STEPS:-5000}"
TRAIN_LOG="${TRAIN_LOG:-${OUTPUT_DIR}/train_pipeline.log}"
mkdir -p "${OUTPUT_DIR}"
bash scripts/train_deor_refcoco_direct.sh 2>&1 | tee "${TRAIN_LOG}"

CHECKPOINT="${OUTPUT_DIR}/checkpoint-${MAX_STEPS}"
test -f "${CHECKPOINT}/config.json" || {
    echo "Expected final checkpoint is missing: ${CHECKPOINT}" >&2
    exit 3
}

EVAL_ROOT="${EVAL_ROOT:-${PROJECT_ROOT}/outputs/eval_deor_refcoco_direct_finetuned}"
MODEL_PATH="${CHECKPOINT}" GPU="${EVAL_GPU:-0}" EVAL_GPUS="${EVAL_GPUS:-}" OUTPUT_ROOT="${EVAL_ROOT}" \
    BATCH_SIZE="${EVAL_BATCH_SIZE:-1}" \
    SUPERVISION_LABEL="fine-tuned on official RefCOCO train expressions only; evaluated checkpoint frozen" \
    SPLITS="val testA testB" bash scripts/eval_deor_refcoco_ours_only.sh

BASELINES="${BASELINES:-${PROJECT_ROOT}/outputs/eval_lisa_glamm_refcoco_nonoverlap}"
ZERO_SHOT_ROOT="${ZERO_SHOT_ROOT:-${PROJECT_ROOT}/outputs/eval_deor_ctrlo_refcoco_direct}"
TRAIN_SOURCE="${TRAIN_SOURCE:-${PROJECT_ROOT}/data/pgot_pix2cap_thing_train.jsonl}"
models=(--model "deor_refcoco_ft=${EVAL_ROOT}/ours")
[[ -d "${ZERO_SHOT_ROOT}/ours" ]] && models+=(--model "deor_zeroshot=${ZERO_SHOT_ROOT}/ours")
[[ -d "${ZERO_SHOT_ROOT}/ctrlo" ]] && models+=(--model "ctrlo=${ZERO_SHOT_ROOT}/ctrlo")

PYTHON="${PYTHON:-/home/jovyan/.conda/envs/scale_rae/bin/python}"
"${PYTHON}" -u scripts/score_mllm_refcoco_baselines.py \
    --manifest "${EVAL_ROOT}/manifest.json" --train "${TRAIN_SOURCE}" \
    --subset full --output "${EVAL_ROOT}/comparison_full" \
    --model "deor_refcoco_ft=${EVAL_ROOT}/ours"

nonoverlap_models=("${models[@]}")
[[ -d "${BASELINES}/lisa" ]] && nonoverlap_models+=(--model "lisa=${BASELINES}/lisa")
[[ -d "${BASELINES}/glamm" ]] && nonoverlap_models+=(--model "glamm=${BASELINES}/glamm")
"${PYTHON}" -u scripts/score_mllm_refcoco_baselines.py \
    --manifest "${EVAL_ROOT}/manifest.json" --train "${TRAIN_SOURCE}" \
    --subset nonoverlap --output "${EVAL_ROOT}/comparison_nonoverlap" \
    "${nonoverlap_models[@]}"

echo "Done: ${EVAL_ROOT}/comparison_nonoverlap/table.md"
