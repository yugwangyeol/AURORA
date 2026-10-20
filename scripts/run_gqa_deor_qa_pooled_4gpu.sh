#!/usr/bin/env bash
# One-epoch four-GPU GQA training with semantic + pooled visual DeOR tokens.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
IFS=, read -r -a DEVICES <<< "${CUDA_VISIBLE_DEVICES}"
if (( ${#DEVICES[@]} != 4 )); then
    echo 'CUDA_VISIBLE_DEVICES must contain exactly four GPUs.' >&2
    exit 2
fi
declare -A SEEN=()
for device in "${DEVICES[@]}"; do
    if [[ -n "${SEEN[$device]:-}" ]]; then
        echo 'CUDA_VISIBLE_DEVICES must contain four distinct GPUs.' >&2
        exit 2
    fi
    SEEN[$device]=1
done

PYTHON="${PYTHON:-/home/jovyan/.conda/envs/scale_rae/bin/python}"
MODEL_PATH="${MODEL_PATH:-/home/jovyan/data/vicuna-7b-v1.5}"
PREPARED_ROOT="${PREPARED_ROOT:-/home/jovyan/data/gqa/prepared_balanced}"
CACHE_ROOT="${CACHE_ROOT:-/home/jovyan/data/gqa/deor_object_cache_m8_r32_bs8}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/checkpoints/gqa_deor_semantic_visual_pooled_vicuna7b}"
EVAL_ROOT="${EVAL_ROOT:-${PROJECT_ROOT}/outputs/eval_gqa_deor_semantic_visual_pooled_vicuna7b}"
LOG_PATH="${LOG_PATH:-${PROJECT_ROOT}/outputs/gqa_deor_semantic_visual_pooled_vicuna7b.log}"
PER_GPU_BATCH="${PER_GPU_BATCH:-16}"
TRAIN_STEPS_OVERRIDE="${TRAIN_STEPS_OVERRIDE:-0}"
EVAL_MAX_SAMPLES="${EVAL_MAX_SAMPLES:-0}"
GENERATION_SAMPLES="${GENERATION_SAMPLES:-999999}"
WORLD_SIZE=4

test -x "${PYTHON}" || { echo "Missing Python: ${PYTHON}" >&2; exit 1; }
test -s "${MODEL_PATH}/config.json" || { echo "Missing Vicuna model: ${MODEL_PATH}" >&2; exit 1; }
TRAIN_MANIFEST="${PREPARED_ROOT}/manifests/questions_train.jsonl"
VAL_MANIFEST="${PREPARED_ROOT}/manifests/questions_val.jsonl"
TESTDEV_MANIFEST="${PREPARED_ROOT}/manifests/questions_testdev.jsonl"
for manifest in "${TRAIN_MANIFEST}" "${VAL_MANIFEST}" "${TESTDEV_MANIFEST}"; do
    test -s "${manifest}" || { echo "Missing manifest: ${manifest}" >&2; exit 1; }
done
if [[ -e "${OUTPUT_DIR}" ]] && find "${OUTPUT_DIR}" -mindepth 1 -print -quit | grep -q .; then
    echo "Refusing to overwrite non-empty checkpoint: ${OUTPUT_DIR}" >&2
    exit 1
fi
if [[ -e "${EVAL_ROOT}" ]] && find "${EVAL_ROOT}" -mindepth 1 -print -quit | grep -q .; then
    echo "Refusing to overwrite non-empty evaluation output: ${EVAL_ROOT}" >&2
    exit 1
fi

NUM_QUESTIONS="$(wc -l < "${TRAIN_MANIFEST}")"
GLOBAL_BATCH=$(( PER_GPU_BATCH * WORLD_SIZE ))
MAX_STEPS=$(( (NUM_QUESTIONS + GLOBAL_BATCH - 1) / GLOBAL_BATCH ))
if (( TRAIN_STEPS_OVERRIDE > 0 )); then
    MAX_STEPS="${TRAIN_STEPS_OVERRIDE}"
fi
mkdir -p "$(dirname "${LOG_PATH}")" "${EVAL_ROOT}"

export PYTHONNOUSERSITE=1
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="$(dirname "${PYTHON}")/../lib:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-3}"

echo "Training semantic_visual_pooled for one epoch: questions=${NUM_QUESTIONS}, global_batch=${GLOBAL_BATCH}, steps=${MAX_STEPS}"
"${PYTHON}" -m torch.distributed.run --standalone --nproc_per_node=4 \
  -m pgot.eval.gqa_deor_qa train \
  --model_path "${MODEL_PATH}" --questions "${TRAIN_MANIFEST}" \
  --cache_root "${CACHE_ROOT}" --split train --output_dir "${OUTPUT_DIR}" \
  --representation semantic_visual_pooled --batch_size "${PER_GPU_BATCH}" \
  --max_steps "${MAX_STEPS}" --learning_rate 2e-4 --warmup_ratio 0.03 \
  --num_workers 0 2>&1 | tee "${LOG_PATH}"

echo 'Training complete. Running val and testdev evaluations in parallel.'
pids=()
CUDA_VISIBLE_DEVICES="${DEVICES[0]}" "${PYTHON}" -m pgot.eval.gqa_deor_qa eval \
  --questions "${VAL_MANIFEST}" --cache_root "${CACHE_ROOT}" --split val \
  --output_dir "${OUTPUT_DIR}" --result_dir "${EVAL_ROOT}/val" \
  --max_samples "${EVAL_MAX_SAMPLES}" --batch_size 16 \
  --generation_samples "${GENERATION_SAMPLES}" --max_new_tokens 8 \
  >"${EVAL_ROOT}/val.log" 2>&1 &
pids+=("$!")
CUDA_VISIBLE_DEVICES="${DEVICES[1]}" "${PYTHON}" -m pgot.eval.gqa_deor_qa eval \
  --questions "${TESTDEV_MANIFEST}" --cache_root "${CACHE_ROOT}" --split testdev \
  --output_dir "${OUTPUT_DIR}" --result_dir "${EVAL_ROOT}/testdev" \
  --max_samples "${EVAL_MAX_SAMPLES}" --batch_size 16 \
  --generation_samples "${GENERATION_SAMPLES}" --max_new_tokens 8 \
  >"${EVAL_ROOT}/testdev.log" 2>&1 &
pids+=("$!")

stop_children() {
    local pid
    for pid in "${pids[@]:-}"; do kill "${pid}" 2>/dev/null || true; done
    for pid in "${pids[@]:-}"; do wait "${pid}" 2>/dev/null || true; done
}
trap 'stop_children; exit 130' INT TERM

failed=0
for index in 0 1; do
    if wait "${pids[$index]}"; then
        echo "PASS evaluation index=${index}"
    else
        status=$?
        echo "FAIL evaluation index=${index} status=${status}" >&2
        failed=1
    fi
done
pids=()
(( failed == 0 )) || exit 1
test -s "${EVAL_ROOT}/val/eval_summary.json"
test -s "${EVAL_ROOT}/testdev/eval_summary.json"
echo "Completed training and both evaluations: ${OUTPUT_DIR}"
