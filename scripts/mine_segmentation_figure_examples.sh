#!/usr/bin/env bash
# Four COCO-checkpoint models; resumable cache, no existing benchmark overwritten.
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"
PYTHON="${PYTHON:-/home/jovyan/.conda/envs/scale_rae/bin/python}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/outputs/figure_segmentation_over_under}"
export CUDA_VISIBLE_DEVICES="${GPU:-0}"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export PYTHONNOUSERSITE=1
export LD_LIBRARY_PATH="$(dirname "${PYTHON}")/../lib:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-3}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-3}"
common=(--output_dir "${OUTPUT_ROOT}" --examples_per_type "${EXAMPLES_PER_TYPE:-10}")
if [[ -n "${LIMIT:-}" ]]; then common+=(--limit "${LIMIT}"); fi
mkdir -p "${OUTPUT_ROOT}"
if [[ ! -f "${OUTPUT_ROOT}/candidates.json" ]]; then
    "${PYTHON}" -m pgot.eval.figure_segmentation_examples --stage prepare \
        "${common[@]}" --candidates_per_type "${CANDIDATES_PER_TYPE:-80}"
fi
if (( $# == 0 )); then stages=(coda metaslot adaslot ours select); else stages=("$@"); fi
for stage in "${stages[@]}"; do
    case "${stage}" in
        metaslot) python_bin="/home/jovyan/.venvs/metaslot_eval/bin/python" ;;
        adaslot) python_bin="/home/jovyan/.venvs/adaslot_eval/bin/python" ;;
        coda|ours|select) python_bin="${PYTHON}" ;;
        *) echo "Unknown stage: ${stage}" >&2; exit 2 ;;
    esac
    args=(--stage infer --model "${stage}")
    if [[ "${stage}" == select ]]; then args=(--stage select); fi
    "${python_bin}" -m pgot.eval.figure_segmentation_examples "${common[@]}" "${args[@]}" \
        2>&1 | tee "${OUTPUT_ROOT}/${stage}.log"
done
