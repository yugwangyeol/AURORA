#!/usr/bin/env bash
# CTRL-O-disclosed VQAv2 setup: T5-base, T-2/64 heads, CE, batch 32, 300k steps.
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
VQAV2_ROOT="${VQAV2_ROOT:-/home/jovyan/data/vqav2}"
CACHE_ROOT="${CACHE_ROOT:-${VQAV2_ROOT}/deor_object_cache_m8_r32}"
WORK_ROOT="${WORK_ROOT:-${VQAV2_ROOT}/deor_vqa_ctrl_o_style}"
TRAIN_QUESTIONS="${WORK_ROOT}/questions_train_aligned.jsonl"
VAL_QUESTIONS="${WORK_ROOT}/questions_val_aligned.jsonl"
ANSWER_VOCAB="${WORK_ROOT}/answer_vocab_top3000.json"
MODEL_PATH="${MODEL_PATH:-google-t5/t5-base}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-${PROJECT_ROOT}/checkpoints/vqav2_deor_vqa_ctrlo300k}"
EVAL_ROOT="${EVAL_ROOT:-${PROJECT_ROOT}/outputs/eval_vqav2_deor_vqa_ctrlo300k}"
MAX_STEPS="${MAX_STEPS:-300000}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-32}"
LEARNING_RATE="${LEARNING_RATE:-1e-4}"
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-10000}"
CPU_THREADS="${CPU_THREADS:-3}"
EVAL_MAX_SAMPLES="${EVAL_MAX_SAMPLES:-0}"

test -x "${PYTHON}" || { echo "Missing Python: ${PYTHON}" >&2; exit 1; }
for path in "${TRAIN_QUESTIONS}" "${VAL_QUESTIONS}" "${ANSWER_VOCAB}"; do
  test -s "${path}" || { echo "Missing input: ${path}" >&2; exit 1; }
done
for split in train val; do
  test -s "${CACHE_ROOT}/verification_${split}.json" || {
    echo "Missing verified DeOR ${split} cache: ${CACHE_ROOT}" >&2; exit 1;
  }
done
if (( GLOBAL_BATCH_SIZE != 32 )); then
  echo 'CTRL-O comparison requires GLOBAL_BATCH_SIZE=32.' >&2
  exit 2
fi
if (( MAX_STEPS != 300000 )); then
  echo "WARNING: MAX_STEPS=${MAX_STEPS}; CTRL-O-aligned full setting is 300000." >&2
fi

AVAILABLE_KIB="$(df -Pk "${PROJECT_ROOT}" | awk 'NR == 2 {print $4}')"
MINIMUM_KIB=$(( 20 * 1024 * 1024 ))
if (( AVAILABLE_KIB < MINIMUM_KIB )); then
  echo "At least 20 GiB free disk is required; available KiB=${AVAILABLE_KIB}." >&2
  exit 1
fi

mkdir -p "${CHECKPOINT_ROOT}" "${EVAL_ROOT}/logs"
export PYTHONNOUSERSITE=1
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="$(dirname "${PYTHON}")/../lib:${LD_LIBRARY_PATH:-}"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${CPU_THREADS}"
export MKL_NUM_THREADS="${CPU_THREADS}"
export MAX_STEPS EVAL_MAX_SAMPLES

MODES=(no_coupling coupling)
GPU_PAIRS=("${DEVICES[0]},${DEVICES[1]}" "${DEVICES[2]},${DEVICES[3]}")
PORTS=(29770 29771)
pids=()

stop_children() {
  local pid
  for pid in "${pids[@]:-}"; do kill "${pid}" 2>/dev/null || true; done
  for pid in "${pids[@]:-}"; do wait "${pid}" 2>/dev/null || true; done
}
trap 'stop_children; exit 130' INT TERM

for index in 0 1; do
  mode="${MODES[$index]}"
  output="${CHECKPOINT_ROOT}/${mode}"
  log="${EVAL_ROOT}/logs/train_${mode}.log"
  if [[ -s "${output}/train_summary.json" && -s "${output}/model.pt" ]]; then
    echo "Training already complete: ${mode}"
    continue
  fi
  echo "START mode=${mode} GPUs=${GPU_PAIRS[$index]} global_batch=32 steps=${MAX_STEPS}"
  (
    CUDA_VISIBLE_DEVICES="${GPU_PAIRS[$index]}" "${PYTHON}" -u \
      -m torch.distributed.run --standalone --nproc_per_node=2 \
      --master_port="${PORTS[$index]}" -m pgot.eval.vqav2_deor_vqa train \
      --questions "${TRAIN_QUESTIONS}" --cache_root "${CACHE_ROOT}" --split train \
      --answer_vocab "${ANSWER_VOCAB}" --output_dir "${output}" \
      --model_path "${MODEL_PATH}" --mode "${mode}" \
      --batch_size "${GLOBAL_BATCH_SIZE}" --max_steps "${MAX_STEPS}" \
      --learning_rate "${LEARNING_RATE}" --output_layers 2 --output_heads 64 \
      --output_dim 128 --projected_dim 126 --checkpoint_every "${CHECKPOINT_EVERY}" \
      --log_every 100 --resume
  ) >"${log}" 2>&1 &
  pids+=("$!")
done

failed=0
for index in "${!pids[@]}"; do
  if wait "${pids[$index]}"; then
    echo "PASS training process=${index}"
  else
    status=$?
    echo "FAIL training process=${index} status=${status}" >&2
    failed=1
  fi
done
pids=()
if (( failed != 0 )); then
  for log in "${EVAL_ROOT}"/logs/train_*.log; do
    echo "--- ${log}" >&2
    tail -100 "${log}" >&2 || true
  done
  exit 1
fi

echo 'Both 300k-step models completed. Starting full 214,354-question val evaluations.'
EVAL_GPUS=("${DEVICES[0]}" "${DEVICES[2]}")
for index in 0 1; do
  mode="${MODES[$index]}"
  output="${CHECKPOINT_ROOT}/${mode}"
  result="${EVAL_ROOT}/${mode}"
  log="${EVAL_ROOT}/logs/eval_${mode}.log"
  if [[ -s "${result}/summary.json" ]]; then
    echo "Evaluation already complete: ${mode}"
    continue
  fi
  echo "START full val mode=${mode} GPU=${EVAL_GPUS[$index]}"
  (
    CUDA_VISIBLE_DEVICES="${EVAL_GPUS[$index]}" "${PYTHON}" -u \
      -m pgot.eval.vqav2_deor_vqa eval \
      --questions "${VAL_QUESTIONS}" --cache_root "${CACHE_ROOT}" --split val \
      --answer_vocab "${ANSWER_VOCAB}" --output_dir "${output}" \
      --result_dir "${result}" --batch_size 64 --max_samples "${EVAL_MAX_SAMPLES}"
  ) >"${log}" 2>&1 &
  pids+=("$!")
done

failed=0
for index in "${!pids[@]}"; do
  if wait "${pids[$index]}"; then
    echo "PASS evaluation process=${index}"
  else
    status=$?
    echo "FAIL evaluation process=${index} status=${status}" >&2
    failed=1
  fi
done
pids=()
(( failed == 0 )) || {
  for log in "${EVAL_ROOT}"/logs/eval_*.log; do tail -100 "${log}" >&2 || true; done
  exit 1
}

"${PYTHON}" - "${EVAL_ROOT}" <<'PY'
import json
import os
import sys
from pathlib import Path

root = Path(sys.argv[1])
requested_eval = int(os.environ["EVAL_MAX_SAMPLES"])
expected_eval = requested_eval if requested_eval > 0 else 214354
steps = int(os.environ["MAX_STEPS"])
rows = []
for mode in ("no_coupling", "coupling"):
    value = json.loads((root / mode / "summary.json").read_text())
    if value["num_questions"] != expected_eval:
        raise SystemExit(f"Incomplete {mode} evaluation: {value['num_questions']}")
    rows.append({
        "mode": mode,
        "num_questions": value["num_questions"],
        "vqa_official_accuracy": value["vqa_official_accuracy"],
        "classification_accuracy": value["classification_accuracy"],
        "mean_selected_object_tokens": value["mean_selected_object_tokens"],
    })
summary = {
    "protocol": "CTRL-O-disclosed VQAv2 setup adapted to frozen DeOR representations",
    "t5": "google-t5/t5-base, fine-tuned",
    "output_transformer": "2 layers, 64 heads, d_model=128, feedforward=128",
    "objective": "cross_entropy over top-3000 answers",
    "global_batch_size": 32,
    "learning_rate": 1e-4,
    "steps": steps,
    "rows": rows,
    "coupling_delta_vqa": rows[1]["vqa_official_accuracy"] - rows[0]["vqa_official_accuracy"],
    "coupling_delta_classification": rows[1]["classification_accuracy"] - rows[0]["classification_accuracy"],
}
(root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
for row in rows:
    print(
        f"{row['mode']}: VQA={100*row['vqa_official_accuracy']:.2f}% "
        f"classification={100*row['classification_accuracy']:.2f}%"
    )
PY

echo "Completed CTRL-O-aligned VQAv2 training and evaluation: ${EVAL_ROOT}/summary.json"
