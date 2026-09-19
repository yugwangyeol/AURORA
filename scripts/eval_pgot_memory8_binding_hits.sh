#!/usr/bin/env bash
# COCO annotation-only class/point query grounding, NOT free-running AR.
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-/home/jovyan/.conda/envs/scale_rae/bin/python}"
MODEL_PATH="${MODEL_PATH:-${PROJECT_ROOT}/checkpoints/pgot_oneshot_memory8_register32_noprior/checkpoint-5000}"
QUERY_MODE="${QUERY_MODE:-class_point}"
BINDING_ROUTE="${BINDING_ROUTE:-direct}"
default_output="${PROJECT_ROOT}/outputs/binding_hits_pgot_memory8_register32_noprior/${QUERY_MODE}"
if [[ "${SMOKE:-0}" == 1 ]]; then default_output="${PROJECT_ROOT}/outputs/_smoke_binding_hits_pgot/${QUERY_MODE}"; fi
if [[ "${BINDING_ROUTE}" == ar_router ]]; then
    default_output="${PROJECT_ROOT}/outputs/binding_hits_pgot_memory8_register32_noprior/ar_router_${QUERY_MODE}"
    if [[ "${SMOKE:-0}" == 1 ]]; then default_output="${PROJECT_ROOT}/outputs/_smoke_binding_hits_pgot/ar_router_${QUERY_MODE}"; fi
fi
OUTPUT_DIR="${OUTPUT_DIR:-${default_output}}"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export PYTHONNOUSERSITE=1
export LD_LIBRARY_PATH="$(dirname "${PYTHON}")/../lib:${LD_LIBRARY_PATH:-}"
export CUDA_VISIBLE_DEVICES="${GPU:-0}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-3}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-3}"
extra=()
if [[ "${SMOKE:-0}" == 1 ]]; then extra+=(--image_ids 139 25593); fi
if [[ -n "${MAX_SAMPLES:-}" ]]; then
    [[ "${SMOKE:-0}" != 1 ]] || { echo "SMOKE and MAX_SAMPLES cannot be combined" >&2; exit 2; }
    extra+=(--max_samples "${MAX_SAMPLES}")
fi
if [[ "${IGNORE_OVERLAPS:-0}" == 1 ]]; then extra+=(--ignore_overlaps); fi
"${PYTHON}" -m pgot.eval.run_binding_hits \
    --model_path "${MODEL_PATH}" --data_root "${DATA_ROOT:-/home/jovyan/data/coco}" \
    --output_dir "${OUTPUT_DIR}" --query_mode "${QUERY_MODE}" \
    --binding_route "${BINDING_ROUTE}" --ar_max_new_tokens "${AR_MAX_NEW_TOKENS:-512}" \
    --center_source "${CENTER_SOURCE:-mask}" --max_bindings "${MAX_BINDINGS:-7}" \
    --max_objects "${MAX_OBJECTS:-64}" --batch_size "${BATCH_SIZE:-1}" \
    --num_workers "${NUM_WORKERS:-2}" --eval_size 512 --seed "${SEED:-0}" \
    --dtype "${DTYPE:-fp32}" "${extra[@]}" "$@"
