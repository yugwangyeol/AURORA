#!/usr/bin/env bash
# Frozen DeOR caches -> RefCOCO-trained lightweight selector -> common scoring.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

PYTHON="${PGOT_PYTHON:-/home/jovyan/.conda/envs/scale_rae/bin/python}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/outputs/refcoco_frozen_deor_selector}"
MODEL_PATH="${MODEL_PATH:-${PROJECT_ROOT}/checkpoints/pgot_oneshot_memory8_register32_noprior/checkpoint-5000}"
EVAL_MANIFEST="${EVAL_MANIFEST:-${PROJECT_ROOT}/outputs/eval_deor_ctrlo_refcoco_direct/manifest.json}"
TRAIN_OVERLAP_SOURCE="${TRAIN_OVERLAP_SOURCE:-${PROJECT_ROOT}/data/pgot_pix2cap_thing_train.jsonl}"
GPU_LIST="${GPU_LIST:-0,1}"
IFS=',' read -r -a GPUS <<< "${GPU_LIST}"
NUM_SHARDS="${#GPUS[@]}"

mkdir -p "${OUTPUT_ROOT}/logs"
export PYTHONNOUSERSITE=1
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${CPU_THREADS:-3}"
export MKL_NUM_THREADS="${CPU_THREADS:-3}"
export TOKENIZERS_PARALLELISM=false

COMMON=(--output "${OUTPUT_ROOT}" --model-path "${MODEL_PATH}" --eval-manifest "${EVAL_MANIFEST}"
  --train-overlap-source "${TRAIN_OVERLAP_SOURCE}"
  --dtype "${DTYPE:-fp32}" --seed "${SEED:-0}" --cpu-threads "${CPU_THREADS:-3}"
  --max-train-expressions "${MAX_TRAIN_EXPRESSIONS:-0}"
  --max-eval-expressions "${MAX_EVAL_EXPRESSIONS:-0}"
  --min-label-iou "${MIN_LABEL_IOU:-0.1}" --epochs "${EPOCHS:-5}"
  --batch-size "${BATCH_SIZE:-256}" --query-batch-size "${QUERY_BATCH_SIZE:-32}")
COMMON+=(--extract-batch-size "${EXTRACT_BATCH_SIZE:-8}")

run_cpu() {
  local stage="$1"
  "${PYTHON}" -u scripts/train_refcoco_frozen_deor_selector.py --stage "${stage}" "${COMMON[@]}" \
    2>&1 | tee "${OUTPUT_ROOT}/logs/${stage}.log"
}

run_shards() {
  local stage="$1" pids=() status=0
  for shard in "${!GPUS[@]}"; do
    CUDA_VISIBLE_DEVICES="${GPUS[$shard]}" \
      "${PYTHON}" -u scripts/train_refcoco_frozen_deor_selector.py --stage "${stage}" \
      "${COMMON[@]}" --num-shards "${NUM_SHARDS}" --shard-index "${shard}" \
      > >(tee "${OUTPUT_ROOT}/logs/${stage}_shard${shard}.log") 2>&1 &
    pids+=("$!")
  done
  for pid in "${pids[@]}"; do wait "${pid}" || status=1; done
  if [[ "${status}" != 0 ]]; then
    echo "${stage} failed; completed shard caches are resumable." >&2
    exit 1
  fi
}

run_cpu prepare
run_shards extract
run_shards encode
run_cpu verify

CUDA_VISIBLE_DEVICES="${GPUS[0]}" "${PYTHON}" -u scripts/train_refcoco_frozen_deor_selector.py \
  --stage train "${COMMON[@]}" 2>&1 | tee "${OUTPUT_ROOT}/logs/train.log"
CUDA_VISIBLE_DEVICES="${GPUS[0]}" "${PYTHON}" -u scripts/train_refcoco_frozen_deor_selector.py \
  --stage evaluate "${COMMON[@]}" 2>&1 | tee "${OUTPUT_ROOT}/logs/evaluate.log"

models=(--model "deor+selector=${OUTPUT_ROOT}/predictions")
BASELINES="${PROJECT_ROOT}/outputs/eval_lisa_glamm_refcoco_nonoverlap"
DIRECT="${PROJECT_ROOT}/outputs/eval_deor_ctrlo_refcoco_direct"
[[ -d "${BASELINES}/lisa" ]] && models+=(--model "lisa=${BASELINES}/lisa")
[[ -d "${BASELINES}/glamm" ]] && models+=(--model "glamm=${BASELINES}/glamm")
[[ -d "${DIRECT}/ours" ]] && models+=(--model "deor-zero-shot=${DIRECT}/ours")
[[ -d "${DIRECT}/ctrlo" ]] && models+=(--model "ctrlo=${DIRECT}/ctrlo")

"${PYTHON}" -u scripts/score_mllm_refcoco_baselines.py \
  --manifest "${EVAL_MANIFEST}" --train "${TRAIN_OVERLAP_SOURCE}" \
  --subset nonoverlap --max-samples "${MAX_EVAL_EXPRESSIONS:-0}" \
  --output "${OUTPUT_ROOT}/scores" "${models[@]}" \
  2>&1 | tee "${OUTPUT_ROOT}/logs/score.log"

echo "Done: ${OUTPUT_ROOT}/scores/table.md"
