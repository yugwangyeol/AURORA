#!/usr/bin/env bash
# Frozen DeOR owner extraction -> identical text-only selector ablations -> score.
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEOR_PYTHON="${DEOR_PYTHON:-/home/jovyan/.conda/envs/scale_rae/bin/python}"
SELECTOR_PYTHON="${SELECTOR_PYTHON:-/home/jovyan/.conda/envs/metaquery/bin/python}"
# Qwen3-VL support is installed here, not in metaquery's older Transformers.
# Select stages only: never expose these packages to native DeOR evaluation.
SELECTOR_SITE_PACKAGES="${SELECTOR_SITE_PACKAGES-/home/jovyan/.local/lib/python3.11/site-packages}"
GPU="${GPU:-0}"
SELECTOR_GPUS="${SELECTOR_GPUS:-${GPU}}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/outputs/natural_object_selection_memory8_register32_noprior}"
MODEL_PATH="${MODEL_PATH:-${PROJECT_ROOT}/checkpoints/pgot_oneshot_memory8_register32_noprior/checkpoint-5000}"
SELECTOR_MODEL="${SELECTOR_MODEL:-/home/jovyan/.cache/huggingface/hub/models--Qwen--Qwen3-VL-4B-Instruct/snapshots/ebb281ec70b05090aa6165b016eac8ec08e71b17}"
IFS=, read -r -a SELECTOR_DEVICES <<< "${SELECTOR_GPUS}"
if [[ "${GPU}" == *,* ]] || (( ${#SELECTOR_DEVICES[@]} < 1 || ${#SELECTOR_DEVICES[@]} > 2 )); then
    echo "GPU must be one device; SELECTOR_GPUS must be one or two devices." >&2; exit 2
fi
if (( ${#SELECTOR_DEVICES[@]} == 2 )) && [[ "${SELECTOR_DEVICES[0]}" == "${SELECTOR_DEVICES[1]}" ]]; then
    echo "Use one selector device for sequential execution, or two distinct devices." >&2; exit 2
fi
for path in "${DEOR_PYTHON}" "${SELECTOR_PYTHON}" "${MODEL_PATH}/config.json" "${SELECTOR_MODEL}/config.json"; do
    test -e "${path}" || { echo "Missing required path: ${path}" >&2; exit 2; }
done
mkdir -p "${OUTPUT_DIR}/logs"
COMMON=(--output_dir "${OUTPUT_DIR}" --model_path "${MODEL_PATH}"
    --annotations "${ANNOTATIONS:-${PROJECT_ROOT}/data/pgot_pix2cap_generated_val5k.jsonl}"
    --train_annotations "${TRAIN_ANNOTATIONS:-${PROJECT_ROOT}/data/pgot_pix2cap_thing_train.jsonl}"
    --captions "${CAPTIONS:-${PROJECT_ROOT}/outputs/segeval_pgot_memory8_full_coco_voc_ar/coco_ar/generated_captions.jsonl}"
    --caption_summary "${CAPTION_SUMMARY:-${PROJECT_ROOT}/outputs/segeval_pgot_memory8_full_coco_voc_ar/coco_ar/summary.json}"
    --selector_model "${SELECTOR_MODEL}" --seed "${SEED:-0}"
    --presence_iou "${PRESENCE_IOU:-0.1}" --eval_size 512
    --cpu_threads "${CPU_THREADS:-3}" --max_caption_tokens "${MAX_CAPTION_TOKENS:-4096}"
    --selector_max_new_tokens "${SELECTOR_MAX_NEW_TOKENS:-64}"
    --selector_max_input_tokens "${SELECTOR_MAX_INPUT_TOKENS:-8192}"
    --save_examples "${SAVE_EXAMPLES:-24}")
if [[ -n "${QUERIES:-}" ]]; then COMMON+=(--queries "${QUERIES}"); fi
if [[ -n "${IMAGE_IDS:-}" ]]; then COMMON+=(--image_ids "${IMAGE_IDS}"); fi
if [[ -n "${MAX_IMAGES:-}" ]]; then COMMON+=(--max_images "${MAX_IMAGES}"); fi
if [[ "${CACHE_BUNDLES:-0}" == 1 ]]; then COMMON+=(--cache_bundles); fi
run_stage() {
    local stage=$1 interpreter=$2 device=$3
    shift 3
    local tag="${stage}"
    local stage_pythonpath="${PROJECT_ROOT}:${PYTHONPATH:-}"
    if [[ "${stage}" == select ]]; then
        tag="select_${2}"
        if [[ -n "${SELECTOR_SITE_PACKAGES}" ]]; then
            stage_pythonpath="${SELECTOR_SITE_PACKAGES}:${stage_pythonpath}"
        fi
    fi
    CUDA_VISIBLE_DEVICES="${device}" \
    LD_LIBRARY_PATH="$(dirname "${interpreter}")/../lib:${LD_LIBRARY_PATH:-}" \
    PYTHONPATH="${stage_pythonpath}" PYTHONNOUSERSITE=1 \
    "${interpreter}" -m pgot.eval.run_natural_object_selection \
        --stage "${stage}" "${COMMON[@]}" "$@" \
        2>&1 | tee -a "${OUTPUT_DIR}/logs/${tag}.log"
}
run_stage prepare "${DEOR_PYTHON}" "${GPU}"
run_stage extract "${DEOR_PYTHON}" "${GPU}"
if (( ${#SELECTOR_DEVICES[@]} == 1 )); then
    run_stage select "${SELECTOR_PYTHON}" "${SELECTOR_DEVICES[0]}" --condition category
    run_stage select "${SELECTOR_PYTHON}" "${SELECTOR_DEVICES[0]}" --condition description
else
    run_stage select "${SELECTOR_PYTHON}" "${SELECTOR_DEVICES[0]}" --condition category &
    category_pid=$!
    run_stage select "${SELECTOR_PYTHON}" "${SELECTOR_DEVICES[1]}" --condition description &
    description_pid=$!
    status=0
    wait "${category_pid}" || status=1
    wait "${description_pid}" || status=1
    if (( status != 0 )); then echo "Selector failed; completed query caches preserved." >&2; exit 1; fi
fi
run_stage score "${DEOR_PYTHON}" "${GPU}"
echo "Completed: ${OUTPUT_DIR}/results.md and comparison.csv"
