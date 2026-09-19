#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEOR_PYTHON="${DEOR_PYTHON:-/home/jovyan/.conda/envs/scale_rae/bin/python}"
SELECTOR_PYTHON="${SELECTOR_PYTHON:-/home/jovyan/.conda/envs/metaquery/bin/python}"
SELECTOR_SITE_PACKAGES="${SELECTOR_SITE_PACKAGES-/home/jovyan/.local/lib/python3.11/site-packages}"
GPU="${GPU:-1}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/outputs/language_object_editing_gpu1}"
if [[ "${GPU}" == *,* ]]; then echo "Use one free device; stages execute sequentially." >&2; exit 2; fi
mkdir -p "${OUTPUT_DIR}/logs"
COMMON=(--output_dir "${OUTPUT_DIR}" --requests "${REQUESTS:-${PROJECT_ROOT}/configs/language_object_editing_requests.jsonl}"
    --model_path "${MODEL_PATH:-${PROJECT_ROOT}/checkpoints/pgot_oneshot_memory8_register32_noprior/checkpoint-5000}"
    --selector_model "${SELECTOR_MODEL:-/home/jovyan/.cache/huggingface/hub/models--Qwen--Qwen3-VL-4B-Instruct/snapshots/ebb281ec70b05090aa6165b016eac8ec08e71b17}"
    --seed "${SEED:-1234}" --diffusion_inference_steps "${STEPS:-10}"
    --guidance_scale "${GUIDANCE_SCALE:-1}" --limit "${LIMIT:-0}"
    --replacement_mode "${REPLACEMENT_MODE:-visual_only}")
if [[ "${COMPARE_REPLACEMENT_MODES:-0}" == 1 ]]; then COMMON+=(--compare_replacement_modes); fi
run_stage() {
    local stage=$1 interpreter=$2 stage_pythonpath="${PROJECT_ROOT}:${PYTHONPATH:-}"
    if [[ "${stage}" == select && -n "${SELECTOR_SITE_PACKAGES}" ]]; then
        stage_pythonpath="${SELECTOR_SITE_PACKAGES}:${stage_pythonpath}"
    fi
    CUDA_VISIBLE_DEVICES="${GPU}" PYTHONNOUSERSITE=1 PYTHONPATH="${stage_pythonpath}" \
    LD_LIBRARY_PATH="$(dirname "${interpreter}")/../lib:${LD_LIBRARY_PATH:-}" OMP_NUM_THREADS=3 MKL_NUM_THREADS=3 \
    "${interpreter}" -u -m pgot.eval.language_object_editing --stage "${stage}" "${COMMON[@]}" \
        2>&1 | tee -a "${OUTPUT_DIR}/logs/${stage}.log"
}
run_stage prepare "${DEOR_PYTHON}"
run_stage select "${SELECTOR_PYTHON}"
run_stage render "${DEOR_PYTHON}"
run_stage audit "${DEOR_PYTHON}"
if [[ "${LIMIT:-0}" == 0 ]]; then
    CUDA_VISIBLE_DEVICES="${GPU}" PYTHONNOUSERSITE=1 PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}" \
    LD_LIBRARY_PATH="$(dirname "${DEOR_PYTHON}")/../lib:${LD_LIBRARY_PATH:-}" \
    "${DEOR_PYTHON}" -m pgot.eval.render_language_object_editing --output_dir "${OUTPUT_DIR}" \
        2>&1 | tee -a "${OUTPUT_DIR}/logs/figure.log"
fi
echo "Completed actual language-selected editing: ${OUTPUT_DIR}/validation.json"
