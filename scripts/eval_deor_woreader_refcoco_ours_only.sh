#!/usr/bin/env bash
# RefCOCO language-region alignment for the 1.5B O8/R32 checkpoint trained
# without Reader supervision losses. Evaluates DeOR only; CTRL-O is not loaded.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

export PYTHONNOUSERSITE=1
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export OMP_NUM_THREADS=3
export MKL_NUM_THREADS=3
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES="${GPU:?Set GPU to an available CUDA index or MIG UUID}"

PYTHON="${PGOT_PYTHON:-/home/jovyan/.conda/envs/scale_rae/bin/python}"
MODEL_PATH="${MODEL_PATH:-${PROJECT_ROOT}/checkpoints/pgot_oneshot_memory8_register32_noprior_woreader/checkpoint-5000}"
REF_DATASET="${REF_DATASET:-refcoco}"

if [[ "${SMOKE:-0}" == 1 ]]; then
    SPLITS="${SPLITS:-val}"
    MAX_REFS="${MAX_REFS:-3}"
    default_output="${PROJECT_ROOT}/outputs/smoke_deor_woreader_refcoco_ours_only"
else
    default_output="${PROJECT_ROOT}/outputs/eval_deor_woreader_refcoco_ours_only"
fi
OUTPUT_ROOT="${OUTPUT_ROOT:-${default_output}}"
mkdir -p "${OUTPUT_ROOT}"

"${PYTHON}" -m pgot.eval.verify_woreader_config --model_path "${MODEL_PATH}"

args=(--output_dir "${OUTPUT_ROOT}" --dataset "${REF_DATASET}"
      --ours_only
      --data_root "${REF_DATA_ROOT:-/home/jovyan/data/coco}"
      --image_root "${REF_IMAGE_ROOT:-/home/jovyan/data/coco}"
      --model_path "${MODEL_PATH}"
      --ctrlo_repo "${CTRLO_REPO:-/home/jovyan/baselines/CTRL-O}"
      --checkpoint_dir "${CTRLO_CHECKPOINT_DIR:-/home/jovyan/baselines/ckpt/ctrlo}"
      --max_refs "${MAX_REFS:-0}" --max_samples "${MAX_SAMPLES:-0}"
      --eval_size "${EVAL_SIZE:-512}" --dtype "${DTYPE:-fp32}" --seed "${SEED:-0}")

if [[ -n "${SPLITS:-}" ]]; then
    read -r -a split_args <<< "${SPLITS}"
    args+=(--splits "${split_args[@]}")
fi
if [[ -n "${SPLIT_BY:-}" ]]; then args+=(--split_by "${SPLIT_BY}"); fi

run_stage() {
    local stage="$1"
    printf '\n[%s] stage=%s dataset=%s\n' "$(date -u +%FT%TZ)" "${stage}" "${REF_DATASET}" | tee -a "${OUTPUT_ROOT}/run.log"
    "${PYTHON}" -u -m pgot.eval.run_refcoco_eval --stage "${stage}" "${args[@]}" 2>&1 | tee -a "${OUTPUT_ROOT}/run.log"
}

trap 'printf "Stopped at line %s; see %s/run.log. Completed queries can be resumed.\n" "${LINENO}" "${OUTPUT_ROOT}" >&2' ERR
run_stage prepare
run_stage ours
run_stage score_ours
printf '\nDone: %s/summary.json\n' "${OUTPUT_ROOT}"
