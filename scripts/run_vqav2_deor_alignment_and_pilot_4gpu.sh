#!/usr/bin/env bash
# Full VQAv2 noun/object alignment on four GPUs, then matched T5-base pilots.
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
PREPARED_ROOT="${PREPARED_ROOT:-${VQAV2_ROOT}/prepared}"
CACHE_ROOT="${CACHE_ROOT:-${VQAV2_ROOT}/deor_object_cache_m8_r32}"
WORK_ROOT="${WORK_ROOT:-${VQAV2_ROOT}/deor_vqa_ctrl_o_style}"
PILOT_ROOT="${PILOT_ROOT:-${PROJECT_ROOT}/outputs/vqav2_deor_vqa_t5base_pilot}"
MODEL_PATH="${MODEL_PATH:-google-t5/t5-base}"
MATCHER_MODEL="${MATCHER_MODEL:-openai/clip-vit-base-patch32}"
CPU_THREADS="${CPU_THREADS:-3}"
PILOT_STEPS="${PILOT_STEPS:-100}"
PILOT_TRAIN_SAMPLES="${PILOT_TRAIN_SAMPLES:-4096}"
PILOT_VAL_SAMPLES="${PILOT_VAL_SAMPLES:-512}"
PILOT_BATCH_SIZE="${PILOT_BATCH_SIZE:-32}"

TRAIN_QUESTIONS="${PREPARED_ROOT}/manifests/questions_train.jsonl"
VAL_QUESTIONS="${PREPARED_ROOT}/manifests/questions_val.jsonl"
ANSWER_VOCAB="${WORK_ROOT}/answer_vocab_top3000.json"
ALIGNMENT_ROOT="${WORK_ROOT}/alignments"
TRAIN_ALIGNED="${WORK_ROOT}/questions_train_aligned.jsonl"
VAL_ALIGNED="${WORK_ROOT}/questions_val_aligned.jsonl"

test -x "${PYTHON}" || { echo "Missing Python: ${PYTHON}" >&2; exit 1; }
test -s "${TRAIN_QUESTIONS}" || { echo "Missing train questions: ${TRAIN_QUESTIONS}" >&2; exit 1; }
test -s "${VAL_QUESTIONS}" || { echo "Missing val questions: ${VAL_QUESTIONS}" >&2; exit 1; }
for split in train val; do
  test -s "${CACHE_ROOT}/verification_${split}.json" || {
    echo "Missing verified DeOR cache for ${split}: ${CACHE_ROOT}" >&2; exit 1;
  }
done

export PYTHONNOUSERSITE=1
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="$(dirname "${PYTHON}")/../lib:${LD_LIBRARY_PATH:-}"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${CPU_THREADS}"
export MKL_NUM_THREADS="${CPU_THREADS}"

mkdir -p "${WORK_ROOT}/logs" "${ALIGNMENT_ROOT}" "${PILOT_ROOT}/logs"

if [[ ! -s "${ANSWER_VOCAB}" ]]; then
  echo 'Building the top-3000 VQAv2 answer vocabulary.'
  "${PYTHON}" -m pgot.eval.vqav2_deor_vqa vocab \
    --train_questions "${TRAIN_QUESTIONS}" --output "${ANSWER_VOCAB}" --size 3000
else
  echo "Using existing answer vocabulary: ${ANSWER_VOCAB}"
fi

pids=()
stop_children() {
  local pid
  for pid in "${pids[@]:-}"; do kill "${pid}" 2>/dev/null || true; done
  for pid in "${pids[@]:-}"; do wait "${pid}" 2>/dev/null || true; done
}
trap 'stop_children; exit 130' INT TERM

run_alignment_split() {
  local split="$1"
  local questions="$2"
  local expected="$3"
  local shard output log failed status
  echo "Starting full ${split} noun/object alignment on four GPUs."
  pids=()
  for shard in 0 1 2 3; do
    output="${ALIGNMENT_ROOT}/${split}.shard$(printf '%03d' "${shard}")-of-004.jsonl"
    log="${WORK_ROOT}/logs/align_${split}_shard${shard}.log"
    (
      CUDA_VISIBLE_DEVICES="${DEVICES[$shard]}" "${PYTHON}" -u \
        -m pgot.eval.vqav2_deor_vqa align \
        --questions "${questions}" --cache_root "${CACHE_ROOT}" --split "${split}" \
        --output "${output}" --num_shards 4 --shard_index "${shard}" \
        --matcher_model "${MATCHER_MODEL}" --device cuda --log_every 1000
    ) >"${log}" 2>&1 &
    pids+=("$!")
    echo "START ${split} shard=${shard} gpu=${DEVICES[$shard]} log=${log}"
  done
  failed=0
  for shard in 0 1 2 3; do
    if wait "${pids[$shard]}"; then
      echo "PASS ${split} alignment shard=${shard}"
    else
      status=$?
      echo "FAIL ${split} alignment shard=${shard} status=${status}" >&2
      tail -80 "${WORK_ROOT}/logs/align_${split}_shard${shard}.log" >&2 || true
      failed=1
    fi
  done
  pids=()
  (( failed == 0 )) || exit 1
  local inputs=()
  for shard in 0 1 2 3; do
    inputs+=("${ALIGNMENT_ROOT}/${split}.shard$(printf '%03d' "${shard}")-of-004.jsonl")
  done
  local merged
  if [[ "${split}" == train ]]; then merged="${TRAIN_ALIGNED}"; else merged="${VAL_ALIGNED}"; fi
  "${PYTHON}" -m pgot.eval.vqav2_deor_vqa merge \
    --inputs "${inputs[@]}" --output "${merged}" --expected_questions "${expected}"
}

run_alignment_split train "${TRAIN_QUESTIONS}" 443757
run_alignment_split val "${VAL_QUESTIONS}" 214354

echo "Full alignment complete: ${TRAIN_ALIGNED} and ${VAL_ALIGNED}"
echo "Preloading ${MODEL_PATH} once before launching concurrent pilots."
"${PYTHON}" - "${MODEL_PATH}" <<'PY'
import sys
from transformers import AutoTokenizer, T5EncoderModel

model = sys.argv[1]
AutoTokenizer.from_pretrained(model, use_fast=True)
T5EncoderModel.from_pretrained(model, low_cpu_mem_usage=True)
print(f"Model is locally available: {model}")
PY

MODES=(no_coupling coupling)
pids=()
for index in 0 1; do
  mode="${MODES[$index]}"
  output="${PILOT_ROOT}/${mode}"
  log="${PILOT_ROOT}/logs/${mode}.log"
  if [[ -s "${output}/train_summary.json" && -s "${output}/eval/summary.json" ]]; then
    echo "Pilot already complete: ${mode}"
    continue
  fi
  if [[ ! -s "${output}/train_summary.json" && -e "${output}" ]] && \
      find "${output}" -mindepth 1 -print -quit | grep -q .; then
    echo "Refusing to overwrite incomplete non-empty pilot output: ${output}" >&2
    exit 1
  fi
  echo "START T5-base pilot mode=${mode} gpu=${DEVICES[$index]} log=${log}"
  (
    if [[ ! -s "${output}/train_summary.json" ]]; then
      CUDA_VISIBLE_DEVICES="${DEVICES[$index]}" "${PYTHON}" -u \
        -m pgot.eval.vqav2_deor_vqa train \
        --questions "${TRAIN_ALIGNED}" --cache_root "${CACHE_ROOT}" --split train \
        --answer_vocab "${ANSWER_VOCAB}" --output_dir "${output}" \
        --model_path "${MODEL_PATH}" --mode "${mode}" \
        --max_samples "${PILOT_TRAIN_SAMPLES}" --batch_size "${PILOT_BATCH_SIZE}" \
        --max_steps "${PILOT_STEPS}" --learning_rate 1e-4
    else
      echo "Using completed training checkpoint for mode=${mode}"
      test -s "${output}/model.pt"
      test -s "${output}/run_config.json"
    fi
    if [[ ! -s "${output}/eval/summary.json" ]]; then
      CUDA_VISIBLE_DEVICES="${DEVICES[$index]}" "${PYTHON}" -u \
        -m pgot.eval.vqav2_deor_vqa eval \
        --questions "${VAL_ALIGNED}" --cache_root "${CACHE_ROOT}" --split val \
        --answer_vocab "${ANSWER_VOCAB}" --output_dir "${output}" \
        --result_dir "${output}/eval" --max_samples "${PILOT_VAL_SAMPLES}" \
        --batch_size "${PILOT_BATCH_SIZE}"
    fi
  ) >"${log}" 2>&1 &
  pids+=("$!")
done

failed=0
for index in "${!pids[@]}"; do
  if wait "${pids[$index]}"; then
    echo "PASS T5-base pilot process=${index}"
  else
    status=$?
    echo "FAIL T5-base pilot process=${index} status=${status}" >&2
    failed=1
  fi
done
pids=()
(( failed == 0 )) || {
  for log in "${PILOT_ROOT}"/logs/*.log; do tail -80 "${log}" >&2 || true; done
  exit 1
}

"${PYTHON}" - "${PILOT_ROOT}" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
rows = []
for mode in ("no_coupling", "coupling"):
    train = json.loads((root / mode / "train_summary.json").read_text())
    evaluation = json.loads((root / mode / "eval" / "summary.json").read_text())
    rows.append({
        "mode": mode,
        "steps": train["steps"],
        "initial_loss": train["initial_loss"],
        "final_loss": train["final_loss"],
        "steps_per_second": train["steps_per_second"],
        "peak_gpu_memory_gib": train["peak_gpu_memory_gib"],
        "num_eval_questions": evaluation["num_questions"],
        "vqa_soft_accuracy": evaluation["vqa_soft_accuracy"],
        "normalized_exact_match": evaluation["normalized_exact_match"],
        "mean_selected_object_tokens": evaluation["mean_selected_object_tokens"],
    })
(root / "pilot_summary.json").write_text(json.dumps({"rows": rows}, indent=2) + "\n")
for row in rows:
    print(
        f"{row['mode']}: loss {row['initial_loss']:.3f}->{row['final_loss']:.3f}, "
        f"soft={100*row['vqa_soft_accuracy']:.2f}%, "
        f"EM={100*row['normalized_exact_match']:.2f}%, "
        f"step/s={row['steps_per_second']:.3f}, peak={row['peak_gpu_memory_gib']:.2f} GiB"
    )
PY

echo "Completed full VQAv2 alignment and both T5-base pilots: ${PILOT_ROOT}"
