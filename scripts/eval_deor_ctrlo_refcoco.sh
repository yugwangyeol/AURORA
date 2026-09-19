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
REF_DATASET="${REF_DATASET:-refcoco}"
if [[ "${SMOKE:-0}" == "1" ]]; then
    SPLITS="${SPLITS:-val}"
    MAX_REFS="${MAX_REFS:-3}"
    OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/outputs/smoke_deor_ctrlo_${REF_DATASET}_direct}"
else
    OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/outputs/eval_deor_ctrlo_${REF_DATASET}_direct}"
fi
mkdir -p "${OUTPUT_ROOT}"
args=(--output_dir "${OUTPUT_ROOT}" --dataset "${REF_DATASET}"
      --data_root "${REF_DATA_ROOT:-/home/jovyan/data/coco}"
      --image_root "${REF_IMAGE_ROOT:-/home/jovyan/data/coco}"
      --model_path "${MODEL_PATH:-${PROJECT_ROOT}/checkpoints/pgot_oneshot_memory8_register32_noprior/checkpoint-5000}"
      --ctrlo_repo "${CTRLO_REPO:-/home/jovyan/baselines/CTRL-O}"
      --checkpoint_dir "${CTRLO_CHECKPOINT_DIR:-/home/jovyan/baselines/ckpt/ctrlo}"
      --max_refs "${MAX_REFS:-0}" --max_samples "${MAX_SAMPLES:-0}"
      --eval_size "${EVAL_SIZE:-512}" --dtype "${DTYPE:-fp32}" --seed "${SEED:-0}"
      --embedding_batch_size "${EMBEDDING_BATCH_SIZE:-4}"
      --embedding_max_length "${EMBEDDING_MAX_LENGTH:-1024}")
if [[ -n "${SPLITS:-}" ]]; then
    read -r -a split_args <<< "${SPLITS}"
    args+=(--splits "${split_args[@]}")
fi
if [[ -n "${SPLIT_BY:-}" ]]; then args+=(--split_by "${SPLIT_BY}"); fi
if [[ -n "${LLM2VEC_BASE:-}" ]]; then args+=(--llm2vec_base "${LLM2VEC_BASE}"); fi
if [[ -n "${LLM2VEC_ADAPTER:-}" ]]; then args+=(--llm2vec_adapter "${LLM2VEC_ADAPTER}"); fi
run_stage() {
    local python_bin="$1" stage="$2"
    printf '\n[%s] stage=%s dataset=%s\n' "$(date -u +%FT%TZ)" "${stage}" "${REF_DATASET}" | tee -a "${OUTPUT_ROOT}/run.log"
    "${python_bin}" -u -m pgot.eval.run_refcoco_eval --stage "${stage}" "${args[@]}" 2>&1 | tee -a "${OUTPUT_ROOT}/run.log"
}
trap 'printf "Stopped at line %s; see %s/run.log. Completed queries can be resumed.\n" "${LINENO}" "${OUTPUT_ROOT}" >&2' ERR
run_stage "${CTRLO_PYTHON}" prepare
run_stage "${CTRLO_PYTHON}" preflight
if [[ "${PREFLIGHT_ONLY:-0}" == "1" ]]; then exit 0; fi
run_stage "${CTRLO_PYTHON}" encode
run_stage "${PGOT_PYTHON}" ours
run_stage "${CTRLO_PYTHON}" ctrlo
run_stage "${CTRLO_PYTHON}" score
printf '\nDone: %s/summary.json\n' "${OUTPUT_ROOT}"
