#!/usr/bin/env bash
# COCO val2017 per-image active-count extraction for AdaSlot, MetaSlot, CODA.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/outputs/baselines_coco_object_counts}"
SCALE_RAE_PYTHON="${SCALE_RAE_PYTHON:-/home/jovyan/.conda/envs/scale_rae/bin/python}"
ADASLOT_PYTHON="${ADASLOT_PYTHON:-/home/jovyan/.venvs/adaslot_eval/bin/python}"
METASLOT_PYTHON="${METASLOT_PYTHON:-/home/jovyan/.venvs/metaslot_eval/bin/python}"

if (( $# == 0 )); then
    stages=(adaslot metaslot coda)
else
    stages=("$@")
fi

mkdir -p "${OUTPUT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export PYTHONNOUSERSITE=1
export CUDA_VISIBLE_DEVICES="${GPU:-0}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-3}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-3}"

extra_args=()
if [[ -n "${MAX_SAMPLES:-}" ]]; then extra_args+=(--max_samples "${MAX_SAMPLES}"); fi

for stage in "${stages[@]}"; do
    case "${stage}" in
        adaslot)  python_bin="${ADASLOT_PYTHON}"; batch_size="${ADASLOT_BATCH_SIZE:-4}" ;;
        metaslot) python_bin="${METASLOT_PYTHON}"; batch_size="${METASLOT_BATCH_SIZE:-16}" ;;
        coda)     python_bin="${SCALE_RAE_PYTHON}"; batch_size="${CODA_BATCH_SIZE:-32}" ;;
        *) echo "Unknown stage: ${stage} (use adaslot|metaslot|coda)" >&2; exit 2 ;;
    esac
    test -x "${python_bin}" || { echo "Missing Python environment: ${python_bin}" >&2; exit 1; }
    mkdir -p "${OUTPUT_ROOT}/${stage}"
    echo "===== ${stage}: COCO val2017 per-image counts ====="
    LD_LIBRARY_PATH="$(dirname "${python_bin}")/../lib:${LD_LIBRARY_PATH:-}" \
    "${python_bin}" -u -m pgot.eval.run_coco_baseline_counts \
        --model "${stage}" \
        --output_dir "${OUTPUT_ROOT}/${stage}" \
        --batch_size "${batch_size}" \
        --num_workers "${NUM_WORKERS:-2}" \
        --seed "${SEED:-0}" \
        "${extra_args[@]}" \
        2>&1 | tee "${OUTPUT_ROOT}/${stage}.log"
done

echo "Done: ${OUTPUT_ROOT}/{adaslot,metaslot,coda}/{summary.json,per_image.jsonl}"
