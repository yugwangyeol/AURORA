#!/usr/bin/env bash
# One command for the full segmentation table of a PGOT checkpoint, all
# autoregressive (no GT captions anywhere):
#   1) COCO  : instance (fARI / mBO^i / mIoU^i) AND class (mBO^c / mIoU^c)
#   2) VOC   : instance AND class (class arrives as sMBO / sMIOU)
#   3) MOVi-C: instance only -- MOVi has no category labels, so no class metric
#   4) MOVi-E: instance only
# Reconstruction metrics are OFF (no diffusion sampling); rFID/KID come from
# the separate reconstruction eval.
#
#   MODEL_PATH=... OUTPUT_ROOT=... bash scripts/eval_pgot_segmentation_suite.sh
#   bash scripts/eval_pgot_segmentation_suite.sh coco voc      # subset of stages
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-/home/jovyan/.conda/envs/scale_rae/bin/python}"
MODEL_PATH="${MODEL_PATH:-${PROJECT_ROOT}/checkpoints/pgot_oneshot_memory4_noprior/checkpoint-5000}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/outputs/segeval_pgot_oneshot_memory4_noprior}"
DATA_ROOT="${DATA_ROOT:-/home/jovyan/data}"
GPU="${GPU:-0}"

if (( $# == 0 )); then
    stages=(coco voc movi-c movi-e)
else
    stages=("$@")
fi
test -f "${MODEL_PATH}/config.json" || { echo "Missing model: ${MODEL_PATH}" >&2; exit 1; }
mkdir -p "${OUTPUT_ROOT}"
export CUDA_VISIBLE_DEVICES="${GPU}"

for stage in "${stages[@]}"; do
    case "${stage}" in
        coco)
            echo "===== [coco] autoregressive | instance + class metrics ====="
            MODEL_PATH="${MODEL_PATH}" \
            OUTPUT_DIR="${OUTPUT_ROOT}/coco_ar" \
            COMPUTE_RFID=False COMPUTE_KID=False COMPUTE_CLASS_METRICS=True \
            BATCH_SIZE="${AR_BATCH_SIZE:-4}" NUM_WORKERS="${NUM_WORKERS:-4}" \
                bash "${PROJECT_ROOT}/scripts/eval_pgot_e8_visual_memory.sh" \
                --caption_mode autoregressive \
                --ar_max_new_tokens "${AR_MAX_NEW_TOKENS:-512}" \
                2>&1 | tee "${OUTPUT_ROOT}/coco_ar.log"
            ;;
        voc|movi-c|movi-e)
            echo "===== [${stage}] zero-shot autoregressive ====="
            MODEL_PATH="${MODEL_PATH}" \
            OUTPUT_ROOT="${OUTPUT_ROOT}/external" \
            DATA_ROOT="${DATA_ROOT}" GPU="${GPU}" \
            BATCH_SIZE="${EXTERNAL_AR_BATCH_SIZE:-1}" NUM_WORKERS="${NUM_WORKERS:-2}" \
            AR_MAX_NEW_TOKENS="${AR_MAX_NEW_TOKENS:-512}" \
                bash "${PROJECT_ROOT}/scripts/eval_pgot_e11_capacity_dit16_coda_datasets.sh" "${stage}"
            ;;
        summary) ;;  # re-print the table from summaries already on disk
        *) echo "Unknown stage: ${stage} (use coco|voc|movi-c|movi-e|summary)" >&2; exit 2 ;;
    esac
done

echo
echo "===== summary ====="
OUTPUT_ROOT="${OUTPUT_ROOT}" "${PYTHON}" - <<'EOF'
import json, os

root = os.environ["OUTPUT_ROOT"]
# COCO_AR_SUMMARY points the COCO row at an AR eval produced elsewhere (e.g. the
# training run's own eval_*/ar) without re-running or symlinking into it.
rows = [("COCO (AR)", os.environ.get("COCO_AR_SUMMARY") or f"{root}/coco_ar/summary.json"),
        ("VOC (AR)", f"{root}/external_voc_ar/summary.json"),
        ("MOVi-C (AR)", f"{root}/external_movi-c_ar/summary.json"),
        ("MOVi-E (AR)", f"{root}/external_movi-e_ar/summary.json")]


def cell(value):
    return f"{value:>10.4f}" if isinstance(value, float) else f"{'-':>10}"


hdr = f"{'dataset':<14}{'n':>7}{'fARI':>10}{'mBO^i':>10}{'mIoU^i':>10}{'mBO^c':>10}{'mIoU^c':>10}  class GT"
print(hdr)
print("-" * (len(hdr) + 8))
for name, path in rows:
    if not os.path.exists(path):
        print(f"{name:<14}{'(not run)':>7}")
        continue
    d = json.load(open(path))
    # instance metrics: mBO_i/mIoU_i on COCO, plain mBO/mIoU elsewhere
    mbo_i = d.get("mBO_i", d.get("mBO"))
    miou_i = d.get("mIoU_i", d.get("mIoU"))
    # class metrics: COCO uses the category cache, VOC its SegmentationClass map
    if d.get("mBO_c") is not None:
        mbo_c, miou_c, src = d["mBO_c"], d.get("mIoU_c"), d.get("class_gt_source", "category cache")
    elif d.get("sMBO") is not None:
        mbo_c, miou_c, src = d["sMBO"], d.get("sMIOU"), "dataset semantic mask"
    else:
        mbo_c = miou_c = None
        src = "none (dataset has no class labels)"
    print(f"{name:<14}{d.get('num_samples', '-'):>7}{cell(d.get('fARI'))}{cell(mbo_i)}{cell(miou_i)}"
          f"{cell(mbo_c)}{cell(miou_c)}  {src}")
print("\nAll rows are autoregressive (model-generated captions), no GT captions.")
print("MOVi has no category annotation, so mBO^c/mIoU^c do not exist for it.")
EOF
echo "outputs under: ${OUTPUT_ROOT}"
