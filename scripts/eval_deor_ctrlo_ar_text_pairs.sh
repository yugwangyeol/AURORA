#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"
export PYTHONNOUSERSITE=1
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="/home/jovyan/.conda/envs/scale_rae/lib:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS=3
export MKL_NUM_THREADS=3
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES="${GPU:?Set GPU to an available CUDA index or MIG UUID}"
PGOT_PYTHON="${PGOT_PYTHON:-/home/jovyan/.conda/envs/scale_rae/bin/python}"
CTRLO_PYTHON="${CTRLO_PYTHON:-/home/jovyan/.venvs/ctrlo_eval/bin/python}"
TEXT_MODE="${TEXT_MODE:-description}"
if [[ "${SMOKE:-0}" == "1" ]]; then
    IMAGE_IDS="${IMAGE_IDS:-coco:139 coco:450686 voc:2010_004165}"
    OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/outputs/smoke_deor_ctrlo_ar_text_pairs_${TEXT_MODE}}"
else
    OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/outputs/eval_deor_ctrlo_ar_text_pairs_${TEXT_MODE}}"
fi
mkdir -p "${OUTPUT_ROOT}"
read -r -a dataset_args <<< "${DATASETS:-coco voc}"
args=(--output_dir "${OUTPUT_ROOT}" --datasets "${dataset_args[@]}"
      --text_mode "${TEXT_MODE}" --max_samples "${MAX_SAMPLES:-0}"
      --model_path "${MODEL_PATH:-${PROJECT_ROOT}/checkpoints/pgot_oneshot_memory8_register32_noprior/checkpoint-5000}"
      --caption_root "${CAPTION_ROOT:-${PROJECT_ROOT}/outputs/segeval_pgot_memory8_full_coco_voc_ar}"
      --dtype "${DTYPE:-fp32}" --seed "${SEED:-0}"
      --queries_per_pass "${QUERIES_PER_PASS:-6}"
      --embedding_batch_size "${EMBEDDING_BATCH_SIZE:-4}"
      --embedding_max_length "${EMBEDDING_MAX_LENGTH:-1024}")
if [[ -n "${IMAGE_IDS:-}" ]]; then
    read -r -a smoke_ids <<< "${IMAGE_IDS}"
    args+=(--image_ids "${smoke_ids[@]}")
fi
if [[ -n "${LLM2VEC_BASE:-}" ]]; then args+=(--llm2vec_base "${LLM2VEC_BASE}"); fi
if [[ -n "${LLM2VEC_ADAPTER:-}" ]]; then args+=(--llm2vec_adapter "${LLM2VEC_ADAPTER}"); fi
run_stage() {
    local python_bin="$1" stage="$2"
    printf '\n[%s] stage=%s text_mode=%s\n' "$(date -u +%FT%TZ)" "${stage}" "${TEXT_MODE}" | tee -a "${OUTPUT_ROOT}/run.log"
    "${python_bin}" -u -m pgot.eval.run_ar_text_pair_eval --stage "${stage}" "${args[@]}" 2>&1 | tee -a "${OUTPUT_ROOT}/run.log"
}
trap 'printf "Evaluation stopped at line %s; see %s/run.log. Completed predictions can be resumed.\n" "${LINENO}" "${OUTPUT_ROOT}" >&2' ERR
run_stage "${CTRLO_PYTHON}" prepare
# Fail before costly PGOT inference if the official full-text encoder is gated.
run_stage "${CTRLO_PYTHON}" preflight
if [[ "${PREFLIGHT_ONLY:-0}" == "1" ]]; then
    printf 'Preflight passed; no model inference was started.\n'
    exit 0
fi
run_stage "${CTRLO_PYTHON}" encode
run_stage "${PGOT_PYTHON}" ours
run_stage "${CTRLO_PYTHON}" ctrlo
run_stage "${CTRLO_PYTHON}" score
printf '\nDone: %s/summary.json\n' "${OUTPUT_ROOT}"
