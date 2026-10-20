#!/usr/bin/env bash
# End-to-end smoke: vocab, frozen noun/object alignment, both VQA modes, reload/eval.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

GPU="${GPU:-0}"
if [[ "${GPU}" == *,* ]]; then
  echo 'GPU must be one CUDA index.' >&2
  exit 2
fi
PYTHON="${PYTHON:-/home/jovyan/.conda/envs/scale_rae/bin/python}"
VQAV2_ROOT="${VQAV2_ROOT:-/home/jovyan/data/vqav2}"
PREPARED_ROOT="${PREPARED_ROOT:-${VQAV2_ROOT}/prepared}"
CACHE_ROOT="${CACHE_ROOT:-${VQAV2_ROOT}/deor_object_cache_m8_r32}"
MODEL_PATH="${MODEL_PATH:-google-t5/t5-small}"
MATCHER_MODEL="${MATCHER_MODEL:-openai/clip-vit-base-patch32}"
TRAIN_SAMPLES="${TRAIN_SAMPLES:-32}"
VAL_SAMPLES="${VAL_SAMPLES:-8}"
TRAIN_STEPS="${TRAIN_STEPS:-2}"
BATCH_SIZE="${BATCH_SIZE:-2}"
SMOKE_ROOT="${SMOKE_ROOT:-$(mktemp -d "${PROJECT_ROOT}/.smoke_vqav2_vqa.XXXXXX")}"

test -x "${PYTHON}" || { echo "Missing Python: ${PYTHON}" >&2; exit 1; }
for split in train val; do
  test -s "${PREPARED_ROOT}/manifests/questions_${split}.jsonl" || {
    echo "Missing VQAv2 manifest: ${split}" >&2; exit 1;
  }
  test -s "${CACHE_ROOT}/verification_${split}.json" || {
    echo "Missing verified DeOR cache: ${split}" >&2; exit 1;
  }
done

cleanup_on_error() {
  echo "Smoke failed; artifacts retained at ${SMOKE_ROOT}" >&2
}
trap cleanup_on_error ERR

export PYTHONNOUSERSITE=1
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="$(dirname "${PYTHON}")/../lib:${LD_LIBRARY_PATH:-}"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-3}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-3}"
export CUDA_VISIBLE_DEVICES="${GPU}"

echo "Smoke root: ${SMOKE_ROOT}"
"${PYTHON}" -m pgot.eval.vqav2_deor_vqa vocab \
  --train_questions "${PREPARED_ROOT}/manifests/questions_train.jsonl" \
  --output "${SMOKE_ROOT}/answer_vocab.json" --size 3000

"${PYTHON}" -m pgot.eval.vqav2_deor_vqa align \
  --questions "${PREPARED_ROOT}/manifests/questions_train.jsonl" \
  --cache_root "${CACHE_ROOT}" --split train \
  --output "${SMOKE_ROOT}/train_aligned.jsonl" --max_samples "${TRAIN_SAMPLES}" \
  --matcher_model "${MATCHER_MODEL}" --device cuda --log_every 8

"${PYTHON}" -m pgot.eval.vqav2_deor_vqa align \
  --questions "${PREPARED_ROOT}/manifests/questions_val.jsonl" \
  --cache_root "${CACHE_ROOT}" --split val \
  --output "${SMOKE_ROOT}/val_aligned.jsonl" --max_samples "${VAL_SAMPLES}" \
  --matcher_model "${MATCHER_MODEL}" --device cuda --log_every 8

for mode in no_coupling coupling; do
  echo "Training smoke mode=${mode}"
  "${PYTHON}" -m pgot.eval.vqav2_deor_vqa train \
    --questions "${SMOKE_ROOT}/train_aligned.jsonl" \
    --cache_root "${CACHE_ROOT}" --split train \
    --answer_vocab "${SMOKE_ROOT}/answer_vocab.json" \
    --output_dir "${SMOKE_ROOT}/${mode}" --model_path "${MODEL_PATH}" \
    --mode "${mode}" --max_samples "${TRAIN_SAMPLES}" \
    --batch_size "${BATCH_SIZE}" --max_steps "${TRAIN_STEPS}"

  echo "Reloading and evaluating smoke mode=${mode}"
  "${PYTHON}" -m pgot.eval.vqav2_deor_vqa eval \
    --questions "${SMOKE_ROOT}/val_aligned.jsonl" \
    --cache_root "${CACHE_ROOT}" --split val \
    --answer_vocab "${SMOKE_ROOT}/answer_vocab.json" \
    --output_dir "${SMOKE_ROOT}/${mode}" --result_dir "${SMOKE_ROOT}/${mode}/eval" \
    --max_samples "${VAL_SAMPLES}" --batch_size "${BATCH_SIZE}"

  test -s "${SMOKE_ROOT}/${mode}/model.pt"
  test -s "${SMOKE_ROOT}/${mode}/train_summary.json"
  test -s "${SMOKE_ROOT}/${mode}/eval/summary.json"
done

"${PYTHON}" - "${SMOKE_ROOT}" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
for mode in ("no_coupling", "coupling"):
    train = json.loads((root / mode / "train_summary.json").read_text())
    evaluation = json.loads((root / mode / "eval" / "summary.json").read_text())
    assert train["status"] == evaluation["status"] == "complete"
    assert evaluation["checkpoint_reload_verified"] is True
    print(
        f"PASS {mode}: steps={train['steps']} "
        f"soft_acc={100*evaluation['vqa_soft_accuracy']:.2f}% "
        f"EM={100*evaluation['normalized_exact_match']:.2f}% "
        f"peak={train['peak_gpu_memory_gib']:.2f} GiB"
    )
PY

trap - ERR
if [[ "${KEEP_SMOKE:-0}" == 1 ]]; then
  echo "PASS: smoke retained at ${SMOKE_ROOT}"
else
  case "${SMOKE_ROOT}" in
    "${PROJECT_ROOT}/.smoke_vqav2_vqa."*)
      find "${SMOKE_ROOT}" -depth -delete
      echo 'PASS: full VQAv2 DeOR VQA smoke completed; smoke artifacts removed.'
      ;;
    *) echo "PASS: custom smoke root retained at ${SMOKE_ROOT}" ;;
  esac
fi
