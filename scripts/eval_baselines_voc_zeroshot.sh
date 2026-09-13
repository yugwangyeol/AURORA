#!/usr/bin/env bash
# Zero-shot VOC segmentation of COCO-trained CODA / SPOT / MetaSlot, scored with
# PGOT's VOC protocol so the rows sit next to PGOT's own zero-shot VOC result.
#
#   bash scripts/eval_baselines_voc_zeroshot.sh              # coda spot metaslot, then table
#   bash scripts/eval_baselines_voc_zeroshot.sh spot         # one model
#   bash scripts/eval_baselines_voc_zeroshot.sh summary      # table only
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCALE_RAE_PYTHON="${SCALE_RAE_PYTHON:-/home/jovyan/.conda/envs/scale_rae/bin/python}"
# MetaSlot needs `astor`; the venv inherits scale_rae's packages and adds only that.
METASLOT_PYTHON="${METASLOT_PYTHON:-/home/jovyan/.venvs/metaslot_eval/bin/python}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/outputs/baselines_voc_zeroshot}"
VOC_ROOT="${VOC_ROOT:-/home/jovyan/data/voc}"
GPU="${GPU:-0}"

if (( $# == 0 )); then
    stages=(coda spot metaslot)
else
    stages=("$@")
fi
mkdir -p "${OUTPUT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export PYTHONNOUSERSITE=1
export LD_LIBRARY_PATH="$(dirname "${SCALE_RAE_PYTHON}")/../lib:${LD_LIBRARY_PATH:-}"
export CUDA_VISIBLE_DEVICES="${GPU}"
# The container's CPU quota is 3 cores (cgroup cpu.max) while torch/OpenCV see
# 288 host cores; uncapped pools make CPU tensor ops ~100x slower.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-3}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-3}"
export OPENCV_FOR_THREADS_NUM="${OPENCV_FOR_THREADS_NUM:-3}"

extra_args=()
if [[ -n "${MAX_SAMPLES:-}" ]]; then extra_args+=(--max_samples "${MAX_SAMPLES}"); fi

for stage in "${stages[@]}"; do
    case "${stage}" in
        coda)     python_bin="${SCALE_RAE_PYTHON}"; stage_args=(--coda_metric_cross_check) ;;
        spot)     python_bin="${SCALE_RAE_PYTHON}"; stage_args=() ;;
        metaslot) python_bin="${METASLOT_PYTHON}";  stage_args=() ;;
        summary)  continue ;;
        *) echo "Unknown stage: ${stage} (use coda|spot|metaslot|summary)" >&2; exit 2 ;;
    esac
    test -x "${python_bin}" || { echo "Missing python for ${stage}: ${python_bin}" >&2; exit 1; }
    echo "===== [${stage}] zero-shot VOC (COCO checkpoint) ====="
    "${python_bin}" -m pgot.eval.eval_baselines_voc_zeroshot \
        --model "${stage}" \
        --voc_root "${VOC_ROOT}" \
        --output_dir "${OUTPUT_ROOT}/${stage}" \
        --batch_size "${BATCH_SIZE:-16}" \
        --num_workers "${NUM_WORKERS:-2}" \
        "${stage_args[@]}" "${extra_args[@]}" \
        2>&1 | tee "${OUTPUT_ROOT}/${stage}.log"
done

echo
echo "===== VOC zero-shot (all models trained on COCO) ====="
OUTPUT_ROOT="${OUTPUT_ROOT}" PROJECT_ROOT="${PROJECT_ROOT}" "${SCALE_RAE_PYTHON}" - <<'EOF'
import json, os

root, project = os.environ["OUTPUT_ROOT"], os.environ["PROJECT_ROOT"]
rows = [
    ("CODA", f"{root}/coda/summary.json"),
    ("SPOT", f"{root}/spot/summary.json"),
    ("MetaSlot", f"{root}/metaslot/summary.json"),
    ("PGOT memory4 noprior (AR)",
     f"{project}/outputs/segeval_pgot_oneshot_memory4_noprior/external_voc_ar/summary.json"),
    ("PGOT memory8 reg32 noprior (AR)",
     f"{project}/outputs/segeval_pgot_oneshot_memory8_register32_noprior/external_voc_ar/summary.json"),
]


def pct(d, *keys):
    for key in keys:
        value = d.get(key)
        if isinstance(value, (int, float)):
            return f"{100 * value:>9.2f}"
    return f"{'-':>9}"


print(f"{'model':<33}{'n':>6}{'FG-ARI':>9}{'mBO^i':>9}{'mBO^c':>9}{'mIoU^i':>9}{'mIoU^c':>9}")
print("-" * 84)
for name, path in rows:
    if not os.path.exists(path):
        print(f"{name:<33}{'(not run)':>6}")
        continue
    d = json.load(open(path))
    print(f"{name:<33}{d.get('num_samples', '-'):>6}{pct(d, 'fARI')}{pct(d, 'mBO_i', 'mBO')}"
          f"{pct(d, 'sMBO')}{pct(d, 'mIoU_i', 'mIoU')}{pct(d, 'sMIOU')}")
    check = d.get("coda_metric_cross_check")
    if check:
        print(f"{'  (CODA metric code)':<33}{'':>6}{pct(check, 'fARI')}{pct(check, 'mBO')}"
              f"{pct(check, 'sMBO')}{pct(check, 'mIoU')}{pct(check, 'sMIOU')}")
print("\nBaselines: COCO checkpoint, VOC2012 val 1,449, 512 center crop, PGOT metric code.")
print("mBO^c / mIoU^c = VOC SegmentationClass (sMBO / sMIOU keys).")
EOF
echo "outputs under: ${OUTPUT_ROOT}"
