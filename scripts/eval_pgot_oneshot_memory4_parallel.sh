#!/usr/bin/env bash
# Independent full-set TF and AR evals, one GPU each. Never average shard FIDs.
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export MEMORY_KEY_MODE="${MEMORY_KEY_MODE:-content}"
export MODEL_PATH="${MODEL_PATH:-${PROJECT_ROOT}/checkpoints/pgot_oneshot_memory4_${MEMORY_KEY_MODE}/checkpoint-${MAX_STEPS:-5000}}"
EVAL_ROOT="${EVAL_ROOT:-${PROJECT_ROOT}/outputs/eval_pgot_oneshot_memory4_${MEMORY_KEY_MODE}}"
IFS=, read -r -a EVAL_DEVICES <<< "${CUDA_VISIBLE_DEVICES:-0,1}"
if (( ${#EVAL_DEVICES[@]} != 2 )); then echo "Set CUDA_VISIBLE_DEVICES to exactly two GPUs" >&2; exit 2; fi
test -f "${MODEL_PATH}/config.json"
mkdir -p "${EVAL_ROOT}"
pids=()
stop_children() { for pid in "${pids[@]}"; do kill "${pid}" 2>/dev/null || true; done; }
trap stop_children INT TERM
CUDA_VISIBLE_DEVICES="${EVAL_DEVICES[0]}" OUTPUT_DIR="${EVAL_ROOT}/tf" \
    bash "${PROJECT_ROOT}/scripts/eval_pgot_oneshot_memory4.sh" --caption_mode teacher_forced \
    > "${EVAL_ROOT}/tf.log" 2>&1 &
pids+=("$!")
CUDA_VISIBLE_DEVICES="${EVAL_DEVICES[1]}" OUTPUT_DIR="${EVAL_ROOT}/ar" BATCH_SIZE="${AR_BATCH_SIZE:-4}" \
    bash "${PROJECT_ROOT}/scripts/eval_pgot_oneshot_memory4.sh" \
    --caption_mode autoregressive --ar_max_new_tokens "${AR_MAX_NEW_TOKENS:-512}" \
    > "${EVAL_ROOT}/ar.log" 2>&1 &
pids+=("$!")
echo "TF and AR evaluations started: ${EVAL_ROOT}/{tf,ar}.log"
status=0
for pid in "${pids[@]}"; do wait "${pid}" || status=1; done
if (( status )); then echo "Evaluation failed; inspect ${EVAL_ROOT}/{tf,ar}.log" >&2; exit 1; fi
test -f "${EVAL_ROOT}/tf/summary.json"
test -f "${EVAL_ROOT}/ar/summary.json"
echo "TF + AR complete: ${EVAL_ROOT}/{tf,ar}/summary.json"
