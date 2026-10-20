#!/usr/bin/env bash
# Four matched 100-step GQA pilots, one DeOR representation per GPU.
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
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/outputs/gqa_deor_qa_pilot_100step}"
PILOT_STEPS="${PILOT_STEPS:-100}"
TRAIN_SAMPLES="${TRAIN_SAMPLES:-4096}"
VAL_SAMPLES="${VAL_SAMPLES:-128}"
GENERATION_SAMPLES="${GENERATION_SAMPLES:-32}"
BATCH_SIZE="${BATCH_SIZE:-4}"

test -x "${PYTHON}" || { echo "Missing Python: ${PYTHON}" >&2; exit 1; }
test -s "${MODEL_PATH}/config.json" || { echo "Missing Vicuna model: ${MODEL_PATH}" >&2; exit 1; }
test -s "${PREPARED_ROOT}/manifests/questions_train.jsonl" || exit 1
test -s "${PREPARED_ROOT}/manifests/questions_val.jsonl" || exit 1
if [[ -e "${OUTPUT_ROOT}" ]] && find "${OUTPUT_ROOT}" -mindepth 1 -print -quit | grep -q .; then
    echo "Refusing to overwrite non-empty pilot output: ${OUTPUT_ROOT}" >&2
    exit 1
fi
mkdir -p "${OUTPUT_ROOT}/logs"

export PYTHONNOUSERSITE=1
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="$(dirname "${PYTHON}")/../lib:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-3}"

MODES=(semantic visual_all semantic_visual_pooled semantic_visual_all)
pids=()

stop_children() {
    local pid
    for pid in "${pids[@]:-}"; do kill "${pid}" 2>/dev/null || true; done
    for pid in "${pids[@]:-}"; do wait "${pid}" 2>/dev/null || true; done
}
trap 'stop_children; exit 130' INT TERM

for index in 0 1 2 3; do
    mode="${MODES[$index]}"
    gpu="${DEVICES[$index]}"
    output="${OUTPUT_ROOT}/${mode}"
    log="${OUTPUT_ROOT}/logs/${mode}.log"
    echo "START gpu=${gpu} mode=${mode} log=${log}"
    (
        CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON}" -m pgot.eval.gqa_deor_qa train \
          --model_path "${MODEL_PATH}" \
          --questions "${PREPARED_ROOT}/manifests/questions_train.jsonl" \
          --cache_root "${CACHE_ROOT}" --split train \
          --output_dir "${output}" --representation "${mode}" \
          --max_samples "${TRAIN_SAMPLES}" --batch_size "${BATCH_SIZE}" \
          --max_steps "${PILOT_STEPS}" --num_workers 0

        CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON}" -m pgot.eval.gqa_deor_qa eval \
          --questions "${PREPARED_ROOT}/manifests/questions_val.jsonl" \
          --cache_root "${CACHE_ROOT}" --split val \
          --output_dir "${output}" --max_samples "${VAL_SAMPLES}" \
          --batch_size "${BATCH_SIZE}" --generation_samples "${GENERATION_SAMPLES}" \
          --max_new_tokens 8
    ) >"${log}" 2>&1 &
    pids+=("$!")
done

failed=0
for index in 0 1 2 3; do
    if wait "${pids[$index]}"; then
        echo "PASS gpu=${DEVICES[$index]} mode=${MODES[$index]}"
    else
        status=$?
        echo "FAIL gpu=${DEVICES[$index]} mode=${MODES[$index]} status=${status}" >&2
        tail -80 "${OUTPUT_ROOT}/logs/${MODES[$index]}.log" >&2 || true
        failed=1
    fi
done
pids=()
(( failed == 0 )) || exit 1

"${PYTHON}" - "${OUTPUT_ROOT}" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
modes = ("semantic", "visual_all", "semantic_visual_pooled", "semantic_visual_all")
rows = []
for mode in modes:
    train = json.loads((root / mode / "train_summary.json").read_text())
    evaluation = json.loads((root / mode / "eval_summary.json").read_text())
    rows.append({
        "representation": mode,
        "initial_loss": train["initial_loss"],
        "final_loss": train["final_loss"],
        "mean_last_10_train_loss": train["mean_last_10_loss"],
        "validation_loss": evaluation["validation_loss"],
        "exact_match": evaluation["exact_match"],
        "mean_visual_input_tokens": evaluation["mean_visual_input_tokens"],
        "steps_per_second": train["steps_per_second"],
        "peak_gpu_memory_gib": train["peak_gpu_memory_gib"],
    })
(root / "pilot_summary.json").write_text(json.dumps({"rows": rows}, indent=2) + "\n")
header = (
    "representation\ttrain-final\tval-loss\tEM\tvisual-tokens\tstep/s\tpeak-GiB"
)
print(header)
for row in rows:
    print(
        f"{row['representation']}\t{row['final_loss']:.4f}\t"
        f"{row['validation_loss']:.4f}\t{100*row['exact_match']:.2f}%\t"
        f"{row['mean_visual_input_tokens']:.2f}\t{row['steps_per_second']:.3f}\t"
        f"{row['peak_gpu_memory_gib']:.2f}"
    )
PY

echo "Completed all four pilots: ${OUTPUT_ROOT}"
