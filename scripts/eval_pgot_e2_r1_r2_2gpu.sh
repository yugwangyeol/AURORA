#!/usr/bin/env bash
# Run E2-R1 oracle routing and E2-R2 predicted-OVT routing concurrently.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_PATH="${MODEL_PATH:-${PROJECT_ROOT}/checkpoints/pgot_lviscap_hardreg_e2/checkpoint-10000}"
VAL_JSONL="${VAL_JSONL:-${PROJECT_ROOT}/data/pgot_coco_instance_val.jsonl}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/outputs}"
R1_OUTPUT_DIR="${R1_OUTPUT_DIR:-${OUTPUT_ROOT}/eval_e2_r1_oracle_gt_coda512}"
R2_OUTPUT_DIR="${R2_OUTPUT_DIR:-${OUTPUT_ROOT}/eval_e2_r2_predicted_ovt_coda512}"

run_route() {
    local gpu="$1"
    local route="$2"
    local output_dir="$3"
    CUDA_VISIBLE_DEVICES="${gpu}" \
    MODEL_PATH="${MODEL_PATH}" \
    VAL_JSONL="${VAL_JSONL}" \
    OUTPUT_DIR="${output_dir}" \
    COMPUTE_RFID="${COMPUTE_RFID:-True}" \
    BATCH_SIZE="${BATCH_SIZE:-2}" \
    NUM_WORKERS="${NUM_WORKERS:-2}" \
    DTYPE="${DTYPE:-fp32}" \
    GUIDANCE_SCALE="${GUIDANCE_SCALE:-2.5}" \
    DIFFUSION_INFERENCE_STEPS="${DIFFUSION_INFERENCE_STEPS:-10}" \
    MAX_SAMPLES="${MAX_SAMPLES:-}" \
    REGISTER_EVAL_ROUTE="${route}" \
    REGISTER_EVAL_GT_THRESHOLD="${REGISTER_EVAL_GT_THRESHOLD:-0.0}" \
    REGISTER_EVAL_PRED_MERGE="${REGISTER_EVAL_PRED_MERGE:-mean}" \
    REGISTER_EVAL_PRED_DILATION="${REGISTER_EVAL_PRED_DILATION:-0}" \
    bash "${PROJECT_ROOT}/scripts/eval_pgot_lviscap_hardreg_e2.sh" \
        2>&1 | tee "${output_dir}.log"
}

echo "E2 R1/R2 parallel evaluation"
echo "  model: ${MODEL_PATH}"
echo "  val:   ${VAL_JSONL}"
echo "  R1:    GPU 0 -> ${R1_OUTPUT_DIR}"
echo "  R2:    GPU 1 -> ${R2_OUTPUT_DIR}"

run_route 0 oracle_gt "${R1_OUTPUT_DIR}" &
r1_pid=$!
run_route 1 predicted_ovt "${R2_OUTPUT_DIR}" &
r2_pid=$!

status=0
if ! wait "${r1_pid}"; then
    echo "R1 oracle evaluation failed" >&2
    status=1
fi
if ! wait "${r2_pid}"; then
    echo "R2 predicted evaluation failed" >&2
    status=1
fi
if (( status != 0 )); then
    exit "${status}"
fi

echo "Both evaluations completed."
echo "  R1 summary: ${R1_OUTPUT_DIR}/summary.json"
echo "  R2 summary: ${R2_OUTPUT_DIR}/summary.json"
