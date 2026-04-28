#!/bin/bash
# Fast 2-GPU eval for a CaptionSlot checkpoint — rFID / SSIM / PSNR / MSE only.
#
# Usage:
#   bash scripts/run_eval_fast.sh
# Override defaults via env vars:
#   MODEL_PATH=... DTYPE=bf16 BATCH_SIZE=24 bash scripts/run_eval_fast.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${PY:-/home/jovyan/.conda/envs/scale_rae/bin/python}"

export MODEL_PATH="${MODEL_PATH:-/home/jovyan/AURORA/checkpoints/captionslot_firstslot_headprior_s1.0_stage1}"
export IMAGE_DIR="${IMAGE_DIR:-/home/jovyan/data/coco/val2017}"
export CAPTIONS_JSONL="${CAPTIONS_JSONL:-/home/jovyan/Scale-RAE/outputs/caption_to_image_recon_concise/captions.jsonl}"
export OUTPUT_DIR="${OUTPUT_DIR:-/home/jovyan/AURORA/outputs/captionslot_headprior_s1.0_stage1_eval_fast}"

export GPU0="${GPU0:-0}"
export GPU1="${GPU1:-1}"
export DTYPE="${DTYPE:-fp32}"            # fp32 | bf16 | fp16
export ALLOW_TF32="${ALLOW_TF32:-1}"     # 1 keeps fp32 weights/acts and enables TF32 GEMMs for speed
export TORCH_COMPILE="${TORCH_COMPILE:-1}"
export BATCH_SIZE="${BATCH_SIZE:-16}"
export NUM_WORKERS="${NUM_WORKERS:-4}"
export DIFFUSION_STEPS="${DIFFUSION_STEPS:-20}"
export GUIDANCE_LEVEL="${GUIDANCE_LEVEL:-1.0}"
export MAX_CAPTION_TOKENS="${MAX_CAPTION_TOKENS:-64}"
export MAX_SAMPLES="${MAX_SAMPLES:-}"      # empty = full 5k
export SAVE_IMAGES="${SAVE_IMAGES:-100}"   # total triptychs across all shards
export SPACY_MODEL="${SPACY_MODEL:-en_core_web_sm}"

# Split the total evenly across 2 shards (ceil so the first shard gets the spare).
SAVE_PER_SHARD_0=$(( (SAVE_IMAGES + 1) / 2 ))
SAVE_PER_SHARD_1=$(( SAVE_IMAGES / 2 ))

LOG_DIR="${OUTPUT_DIR}/logs"
SHARD0_DIR="${OUTPUT_DIR}/shard_0_of_2"
SHARD1_DIR="${OUTPUT_DIR}/shard_1_of_2"
mkdir -p "${LOG_DIR}"

run_shard () {
  local gpu="$1"
  local shard_idx="$2"
  local shard_out="$3"
  local log_file="$4"
  local save_images="$5"

  CUDA_VISIBLE_DEVICES="${gpu}" \
  PYTHONNOUSERSITE="${PYTHONNOUSERSITE:-1}" \
  "${PY}" "${SCRIPT_DIR}/eval_fast.py" \
    --action eval \
    --model-path "${MODEL_PATH}" \
    --image-dir "${IMAGE_DIR}" \
    --captions-jsonl "${CAPTIONS_JSONL}" \
    --output-dir "${shard_out}" \
    --device cuda \
    --dtype "${DTYPE}" \
    --allow-tf32 "${ALLOW_TF32}" \
    --torch-compile "${TORCH_COMPILE}" \
    --batch-size "${BATCH_SIZE}" \
    --num-workers "${NUM_WORKERS}" \
    --max-caption-tokens "${MAX_CAPTION_TOKENS}" \
    --guidance-level "${GUIDANCE_LEVEL}" \
    --diffusion-steps "${DIFFUSION_STEPS}" \
    --save-images "${save_images}" \
    --spacy-model "${SPACY_MODEL}" \
    --num-shards 2 \
    --shard-index "${shard_idx}" \
    ${MAX_SAMPLES:+--max-samples "${MAX_SAMPLES}"} \
    > "${log_file}" 2>&1
}

echo "[run_eval_fast] GPU ${GPU0} -> shard 0 | GPU ${GPU1} -> shard 1"
echo "[run_eval_fast] dtype=${DTYPE} allow_tf32=${ALLOW_TF32} torch_compile=${TORCH_COMPILE} batch=${BATCH_SIZE} steps=${DIFFUSION_STEPS} save_images=${SAVE_IMAGES}"
echo "[run_eval_fast] logs -> ${LOG_DIR}"

run_shard "${GPU0}" 0 "${SHARD0_DIR}" "${LOG_DIR}/shard_0_of_2.log" "${SAVE_PER_SHARD_0}" &
PID0=$!
run_shard "${GPU1}" 1 "${SHARD1_DIR}" "${LOG_DIR}/shard_1_of_2.log" "${SAVE_PER_SHARD_1}" &
PID1=$!

FAIL=0
wait "${PID0}" || FAIL=1
wait "${PID1}" || FAIL=1
if [[ "${FAIL}" -ne 0 ]]; then
  echo "[run_eval_fast] a shard failed — inspect ${LOG_DIR}/*.log" >&2
  exit 1
fi

echo "[run_eval_fast] merging shard outputs"
PYTHONNOUSERSITE="${PYTHONNOUSERSITE:-1}" \
"${PY}" "${SCRIPT_DIR}/eval_fast.py" \
  --action merge \
  --output-dir "${OUTPUT_DIR}" \
  --shard-dir "${SHARD0_DIR}" \
  --shard-dir "${SHARD1_DIR}"

echo "[run_eval_fast] done. summary -> ${OUTPUT_DIR}/summary.json"
