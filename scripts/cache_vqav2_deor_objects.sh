#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

GPU="${GPU:-0}"
if [[ "${GPU}" == *,* ]]; then
  echo 'GPU must be one CUDA index or MIG UUID.' >&2
  exit 2
fi
PYTHON="${PYTHON:-/home/jovyan/.conda/envs/scale_rae/bin/python}"
PREPARED_ROOT="${PREPARED_ROOT:-/home/jovyan/data/vqav2/prepared}"
CACHE_ROOT="${CACHE_ROOT:-/home/jovyan/data/vqav2/deor_object_cache_m8_r32}"
MODEL_PATH="${MODEL_PATH:-${PROJECT_ROOT}/checkpoints/pgot_oneshot_memory8_register32_noprior/checkpoint-5000}"
SPLIT="${SPLIT:-train}"
MANIFEST="${IMAGES_MANIFEST:-${PREPARED_ROOT}/manifests/images_${SPLIT}.jsonl}"

args=(extract --images_manifest "${MANIFEST}" --cache_root "${CACHE_ROOT}"
  --model_path "${MODEL_PATH}" --dtype "${DTYPE:-fp32}"
  --cache_dtype "${CACHE_DTYPE:-float16}" --eval_size 512
  --max_objects 50 --max_caption_tokens 4096
  --ar_max_new_tokens "${AR_MAX_NEW_TOKENS:-2048}"
  --cpu_threads "${CPU_THREADS:-3}" --seed "${SEED:-0}"
  --batch_size "${BATCH_SIZE:-8}"
  --num_shards "${NUM_SHARDS:-1}" --shard_index "${SHARD_INDEX:-0}")
if [[ "${MAX_IMAGES:-0}" != 0 ]]; then
  args+=(--max_images "${MAX_IMAGES}")
fi

mkdir -p "${CACHE_ROOT}/logs"
log="${CACHE_ROOT}/logs/${SPLIT}_${NUM_SHARDS:-1}way_shard${SHARD_INDEX:-0}_bs${BATCH_SIZE:-8}.log"
CUDA_VISIBLE_DEVICES="${GPU}" PYTHONNOUSERSITE=1 \
  PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}" \
  LD_LIBRARY_PATH="$(dirname "${PYTHON}")/../lib:${LD_LIBRARY_PATH:-}" \
  OMP_NUM_THREADS="${CPU_THREADS:-3}" MKL_NUM_THREADS="${CPU_THREADS:-3}" \
  TOKENIZERS_PARALLELISM=false \
  "${PYTHON}" -u -m pgot.eval.vqav2_object_cache "${args[@]}" 2>&1 | tee -a "${log}"
