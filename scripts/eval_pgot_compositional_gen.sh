#!/usr/bin/env bash
# CODA-style compositional generation: randomly mix owner slots within a batch,
# decode the composed slot sets, and report FID / KID x1e3 against real images.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_PATH="${MODEL_PATH:-${PROJECT_ROOT}/checkpoints/pgot_oneshot_memory4_content/checkpoint-5000}"
VAL_JSONL="${VAL_JSONL:-${PROJECT_ROOT}/data/pgot_pix2cap_generated_val5k.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/outputs/compgen_pgot_oneshot_memory4_content}"
PYTHON="${PYTHON:-/home/jovyan/.conda/envs/scale_rae/bin/python}"

for path in "${MODEL_PATH}/config.json" "${VAL_JSONL}"; do
    test -e "${path}" || { echo "Missing required path: ${path}" >&2; exit 1; }
done
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export PYTHONNOUSERSITE=1
export LD_LIBRARY_PATH="$(dirname "${PYTHON}")/../lib:${LD_LIBRARY_PATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

EXTRA_ARGS=()
if [[ -n "${MAX_SAMPLES:-}" ]]; then EXTRA_ARGS+=(--max_samples "${MAX_SAMPLES}"); fi

"${PYTHON}" -m pgot.eval.run_compositional_gen \
    --model_path "${MODEL_PATH}" --val_jsonl "${VAL_JSONL}" --output_dir "${OUTPUT_DIR}" \
    --batch_size "${BATCH_SIZE:-4}" --num_workers "${NUM_WORKERS:-4}" \
    --grid_size 32 --max_caption_tokens 1024 --n_ovt_per_object 1 --max_objects 50 \
    --image_preprocess_mode coda_center_crop --coda_crop_size 512 \
    --dtype "${DTYPE:-fp32}" --guidance_scale "${GUIDANCE_SCALE:-1.0}" \
    --diffusion_inference_steps "${DIFFUSION_INFERENCE_STEPS:-10}" \
    --mix_mode "${MIX_MODE:-random_slots}" --register_source "${REGISTER_SOURCE:-random}" \
    --seed "${SEED:-1234}" \
    --kid_subsets "${KID_SUBSETS:-100}" --kid_subset_size "${KID_SUBSET_SIZE:-1000}" \
    --save_examples "${SAVE_EXAMPLES:-8}" \
    "${EXTRA_ARGS[@]}" "$@"
