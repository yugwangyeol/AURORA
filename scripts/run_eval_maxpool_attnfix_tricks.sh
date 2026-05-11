#!/usr/bin/env bash
# ============================================================
# maxpool_lora_attnfix/best-checkpoint — Tricks eval (최초)
# 선택 이유: maxpool 계열 중 fARI=42.78 (최고) + rFID=15.73 (최고)
#
# 적용 tricks:
#   1. attn-temperature sweep: 0.5, 0.3
#      - sigmoid 독립 맵 → softmax 경쟁 할당으로 변환
#   2. --coco-instances 추가 → gt-min-area-pct 실제 작동
#   3. diffusion-steps: 20 유지
#   4. slot-merge-mode: mean (fullstyle_eval과 동일 조건)
# ============================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

MODEL_PATH="${MODEL_PATH:-/home/jovyan/AURORA/checkpoints/aurora_refcoco_singlephase_maxpool_lora_attnfix_lowlr_fp32/best-checkpoint}"
OUTPUT_BASE="${OUTPUT_BASE:-/home/jovyan/AURORA/outputs}"
CAPTIONS_JSONL="${CAPTIONS_JSONL:-/home/jovyan/AURORA/outputs/stagea_object_captions_val2017_refexp/predictions.jsonl}"
IMAGE_DIR="${IMAGE_DIR:-/home/jovyan/data/coco/val2017}"
COCO_INSTANCES="${COCO_INSTANCES:-/home/jovyan/data/coco/annotations/instances_val2017.json}"
CUDA_DEVICE="${CUDA_DEVICE:-0}"

# Temperature 후보: 0.5, 0.3  (maxpool은 tricks eval 처음)
TEMPERATURES="${TEMPERATURES:-0.5 0.3}"

for TEMP in ${TEMPERATURES}; do
    TAG="temp${TEMP/./}"     # 0.5 → temp05, 0.3 → temp03
    OUTPUT_DIR="${OUTPUT_BASE}/maxpool_attnfix_tricks_${TAG}_eval"

    echo ""
    echo "============================================================"
    echo "[maxpool_attnfix] attn-temperature=${TEMP}"
    echo "  model : ${MODEL_PATH}"
    echo "  output: ${OUTPUT_DIR}"
    echo "============================================================"

    PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}" \
    /home/jovyan/.conda/envs/scale_rae/bin/python \
        "${SCRIPT_DIR}/eval_captionslot_checkpoint.py" \
        --model-path        "${MODEL_PATH}" \
        --image-dir         "${IMAGE_DIR}" \
        --captions-jsonl    "${CAPTIONS_JSONL}" \
        --output-dir        "${OUTPUT_DIR}" \
        --device            cuda \
        --dtype             fp32 \
        --per-device-eval-batch-size 8 \
        --dataloader-num-workers 4 \
        --max-caption-tokens 192 \
        --guidance-level    1.0 \
        --diffusion-steps   20 \
        --eval-slot-attention 1 \
        --attn-threshold    0.5 \
        --attn-thresholds   "0.2,0.3" \
        --attn-temperature  "${TEMP}" \
        --segmentation-bg-threshold 0.5 \
        --segmentation-bg-thresholds "0.05,0.1,0.15,0.2,0.3" \
        --coco-instances    "${COCO_INSTANCES}" \
        --gt-min-area-pct   0.005 \
        --raw-slot-argmax   0 \
        --slot-merge-mode   mean \
        --save-images       1 \
        --save-limit        50 \
        --save-fixed-first-n 30 \
        --save-attn-maps    1 \
        --save-attn-limit   50 \
        --report-losses     1 \
        --torch-compile     0 \
        ${MAX_SAMPLES:+--max-samples "${MAX_SAMPLES}"}

    echo "[maxpool_attnfix | temp=${TEMP}] 완료 → ${OUTPUT_DIR}/summary.json"
done

echo ""
echo "=== 모든 maxpool_attnfix tricks eval 완료 ==="
echo "결과 비교:"
for TEMP in ${TEMPERATURES}; do
    TAG="temp${TEMP/./}"
    F="${OUTPUT_BASE}/maxpool_attnfix_tricks_${TAG}_eval/summary.json"
    if [ -f "${F}" ]; then
        python3 -c "
import json
with open('${F}') as fp:
    d = json.load(fp)
s = d.get('segmentation_metrics_strict', {})
r = d.get('reconstruction_metrics', {})
print(f'temp=${TEMP}: fARI={s.get(\"fARI\",0)*100:.2f}  mBO={s.get(\"MBO\",0)*100:.2f}  mIoU={s.get(\"mIoU\",0)*100:.2f}  rFID={r.get(\"rFID\",0):.2f}')
"
    fi
done
echo "(참고: CODA → fARI=47.5  mBO=36.3  mIoU=36.4  rFID=10.65)"
echo "(참고: maxpool_attnfix 기존(no tricks) → fARI=42.78  mBO=23.17  mIoU=26.07  rFID=16.21)"
