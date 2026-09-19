#!/usr/bin/env bash
# All stages use COCO checkpoints. Default: SPOT COCO + SlotDiffusion COCO/VOC.
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCALE_RAE_PYTHON="${SCALE_RAE_PYTHON:-/home/jovyan/.conda/envs/scale_rae/bin/python}"
SLOTDIFFUSION_PYTHON="${SLOTDIFFUSION_PYTHON:-/home/jovyan/.venvs/slotdiffusion_eval/bin/python}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/outputs/baselines_coco_checkpoint}"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export PYTHONNOUSERSITE=1
export LD_LIBRARY_PATH="$(dirname "${SCALE_RAE_PYTHON}")/../lib:${LD_LIBRARY_PATH:-}"
export CUDA_VISIBLE_DEVICES="${GPU:-0}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-3}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-3}"
export OPENCV_FOR_THREADS_NUM="${OPENCV_FOR_THREADS_NUM:-3}"
if (( $# == 0 )); then stages=(spot_coco slotdiffusion_coco slotdiffusion_voc); else stages=("$@"); fi
extra_args=()
if [[ -n "${MAX_SAMPLES:-}" ]]; then extra_args+=(--max_samples "${MAX_SAMPLES}"); fi
mkdir -p "${OUTPUT_ROOT}"
for stage in "${stages[@]}"; do
    case "${stage}" in
        spot_coco) model=spot; dataset=coco; python_bin="${SCALE_RAE_PYTHON}" ;;
        slotdiffusion_coco) model=slotdiffusion; dataset=coco; python_bin="${SLOTDIFFUSION_PYTHON}" ;;
        slotdiffusion_voc) model=slotdiffusion; dataset=voc; python_bin="${SLOTDIFFUSION_PYTHON}" ;;
        *) echo "Unknown stage: ${stage}" >&2; exit 2 ;;
    esac
    test -x "${python_bin}"
    "${python_bin}" -m pgot.eval.eval_baselines_voc_zeroshot \
        --model "${model}" --dataset "${dataset}" \
        --coco_root "${COCO_ROOT:-/home/jovyan/data/coco}" \
        --voc_root "${VOC_ROOT:-/home/jovyan/data/voc}" \
        --output_dir "${OUTPUT_ROOT}/${stage}" \
        --batch_size "${BATCH_SIZE:-8}" --num_workers "${NUM_WORKERS:-2}" --seed "${SEED:-0}" \
        "${extra_args[@]}" 2>&1 | tee "${OUTPUT_ROOT}/${stage}.log"
done
if [[ -z "${MAX_SAMPLES:-}" && -f "${OUTPUT_ROOT}/spot_coco/summary.json" && \
      -f "${OUTPUT_ROOT}/slotdiffusion_coco/summary.json" && -f "${OUTPUT_ROOT}/slotdiffusion_voc/summary.json" ]]; then
    "${SCALE_RAE_PYTHON}" -m pgot.eval.summarize_coco_checkpoint_baselines --output_root "${OUTPUT_ROOT}"
fi
