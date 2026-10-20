#!/usr/bin/env bash
# End-to-end smoke for semantic-only DeOR -> projector -> Vicuna-7B LoRA GQA.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2,3}"
IFS=, read -r -a DEVICES <<< "${CUDA_VISIBLE_DEVICES}"
if (( ${#DEVICES[@]} != 2 )) || [[ "${DEVICES[0]}" == "${DEVICES[1]}" ]]; then
    echo 'CUDA_VISIBLE_DEVICES must contain exactly two distinct GPUs.' >&2
    exit 2
fi

PYTHON="${PYTHON:-/home/jovyan/.conda/envs/scale_rae/bin/python}"
MODEL_PATH="${MODEL_PATH:-/home/jovyan/data/vicuna-7b-v1.5}"
PREPARED_ROOT="${PREPARED_ROOT:-/home/jovyan/data/gqa/prepared_balanced}"
CACHE_ROOT="${CACHE_ROOT:-/home/jovyan/data/gqa/deor_object_cache_m8_r32_bs8}"
SMOKE_ROOT="${SMOKE_ROOT:-$(mktemp -d "${PROJECT_ROOT}/.smoke_gqa_qa.XXXXXX")}"
MASTER_PORT="${MASTER_PORT:-29637}"

test -s "${MODEL_PATH}/config.json" || { echo "Missing Vicuna model: ${MODEL_PATH}" >&2; exit 1; }
test -s "${PREPARED_ROOT}/manifests/questions_train.jsonl" || exit 1
test -s "${PREPARED_ROOT}/manifests/questions_val.jsonl" || exit 1

export PYTHONNOUSERSITE=1
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="$(dirname "${PYTHON}")/../lib:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"

echo "Smoke output: ${SMOKE_ROOT}"
"${PYTHON}" -m torch.distributed.run \
  --standalone --nproc_per_node=2 --master_port="${MASTER_PORT}" \
  -m pgot.eval.gqa_deor_qa train \
  --model_path "${MODEL_PATH}" \
  --questions "${PREPARED_ROOT}/manifests/questions_train.jsonl" \
  --cache_root "${CACHE_ROOT}" --split train \
  --output_dir "${SMOKE_ROOT}" \
  --max_samples 32 --batch_size 1 --max_steps 2 --num_workers 0

# Reload the written LoRA and projector in a fresh process, then generate answers.
CUDA_VISIBLE_DEVICES="${DEVICES[0]}" "${PYTHON}" -m pgot.eval.gqa_deor_qa eval \
  --questions "${PREPARED_ROOT}/manifests/questions_val.jsonl" \
  --cache_root "${CACHE_ROOT}" --split val \
  --output_dir "${SMOKE_ROOT}" --max_samples 4 --max_new_tokens 8

test -s "${SMOKE_ROOT}/adapter/adapter_model.safetensors"
test -s "${SMOKE_ROOT}/deor_projector.pt"
test -s "${SMOKE_ROOT}/train_summary.json"
test -s "${SMOKE_ROOT}/eval_summary.json"
echo "PASS: two-GPU training, checkpoint save/reload, and generation all completed."
echo "Removing smoke artifacts: ${SMOKE_ROOT}"
rm -rf -- "${SMOKE_ROOT}"
