#!/usr/bin/env bash
# Inference-only latent-splice oracles: decompose the rFID gap into background
# vs object condition information. No training — only the DiT-sampled SigLIP
# latents are partially replaced with GT encoder-B features before pixel
# decoding.
#
# Usage:
#   SPLICE_MODE=background_gt bash scripts/eval_pgot_latent_splice_oracle.sh
#   SPLICE_MODE=object_gt     bash scripts/eval_pgot_latent_splice_oracle.sh
#   SPLICE_MODE=full_gt       bash scripts/eval_pgot_latent_splice_oracle.sh
#
# Env overrides: GPU, MODEL_PATH, OUTPUT_DIR, MAX_SAMPLES, BATCH_SIZE,
#                FG_THRESHOLD, DTYPE, GUIDANCE_SCALE, DIFFUSION_INFERENCE_STEPS
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SPLICE_MODE="${SPLICE_MODE:?set SPLICE_MODE to background_gt, object_gt, or full_gt}"
case "${SPLICE_MODE}" in
    background_gt|object_gt|full_gt) ;;
    *) echo "Invalid SPLICE_MODE=${SPLICE_MODE}" >&2; exit 1 ;;
esac

MODEL_PATH="${MODEL_PATH:-${PROJECT_ROOT}/checkpoints/pgot_e10_r_raw_value/checkpoint-10000}"
VAL_JSONL="${VAL_JSONL:-${PROJECT_ROOT}/data/pgot_pix2cap_generated_val5k.jsonl}"
MODEL_TAG="$(basename "$(dirname "${MODEL_PATH}")")"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/outputs/eval_${MODEL_TAG}_latent_splice_${SPLICE_MODE}}"
COCO_MASK_CACHE="${COCO_MASK_CACHE:-${PROJECT_ROOT}/data/coco_inst_mask_cache_coda512}"
PYTHON="${PYTHON:-/home/jovyan/.conda/envs/scale_rae/bin/python}"

for path in "${MODEL_PATH}" "${VAL_JSONL}" "${COCO_MASK_CACHE}/meta.json"; do
    test -e "${path}" || { echo "Missing required path: ${path}" >&2; exit 1; }
done
export CUDA_VISIBLE_DEVICES="${GPU:-0}"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export PYTHONNOUSERSITE=1
export LD_LIBRARY_PATH="$(dirname "${PYTHON}")/../lib:${LD_LIBRARY_PATH:-}"

EXTRA_ARGS=()
if [[ -n "${MAX_SAMPLES:-}" ]]; then EXTRA_ARGS+=(--max_samples "${MAX_SAMPLES}"); fi

"${PYTHON}" -m pgot.eval.run_eval \
    --model_path "${MODEL_PATH}" --val_jsonl "${VAL_JSONL}" --output_dir "${OUTPUT_DIR}" \
    --batch_size "${BATCH_SIZE:-4}" --num_workers "${NUM_WORKERS:-4}" \
    --grid_size 32 --max_caption_tokens 1024 --n_ovt_per_object 1 --max_objects 50 \
    --eval_size 512 --readout ovt_owner --eval_merge mean \
    --gt_source coco_instance --coco_mask_cache "${COCO_MASK_CACHE}" \
    --image_preprocess_mode coda_center_crop --coda_crop_size 512 \
    --register_eval_route unrestricted \
    --dtype "${DTYPE:-fp32}" --guidance_scale "${GUIDANCE_SCALE:-1.0}" \
    --diffusion_inference_steps "${DIFFUSION_INFERENCE_STEPS:-10}" \
    --compute_rfid \
    --latent_splice "${SPLICE_MODE}" \
    --latent_splice_fg_threshold "${FG_THRESHOLD:-0.5}" \
    "${EXTRA_ARGS[@]}"
