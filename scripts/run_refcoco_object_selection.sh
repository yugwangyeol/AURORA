#!/usr/bin/env bash
# Image-only DeOR discovery -> text selector ablations -> CTRL-O -> exact-target score.
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"
GPU="${GPU:-1}"
if [[ "${GPU}" == *,* ]]; then echo 'GPU must be one CUDA index or MIG UUID.' >&2; exit 2; fi
DEOR_PYTHON="${DEOR_PYTHON:-/home/jovyan/.conda/envs/scale_rae/bin/python}"
SELECTOR_PYTHON="${SELECTOR_PYTHON:-/home/jovyan/.conda/envs/metaquery/bin/python}"
CTRLO_PYTHON="${CTRLO_PYTHON:-/home/jovyan/.venvs/ctrlo_eval/bin/python}"
SELECTOR_SITE_PACKAGES="${SELECTOR_SITE_PACKAGES-/home/jovyan/.local/lib/python3.11/site-packages}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/outputs/refcoco_object_selection_memory8_register32}"
WITH_CTRLO="${WITH_CTRLO:-1}"
if [[ "${WITH_CTRLO}" != 0 && "${WITH_CTRLO}" != 1 ]]; then echo 'WITH_CTRLO must be 0 or 1.' >&2; exit 2; fi
mkdir -p "${OUTPUT_DIR}/logs"
read -r -a split_args <<< "${SPLITS:-val}"
COMMON=(--output_dir "${OUTPUT_DIR}" --splits "${split_args[@]}"
    --dataset "${REF_DATASET:-refcoco}" --split_by "${SPLIT_BY:-unc}"
    --data_root "${REF_DATA_ROOT:-/home/jovyan/data/coco}"
    --image_root "${REF_IMAGE_ROOT:-/home/jovyan/data/coco}"
    --model_path "${MODEL_PATH:-${PROJECT_ROOT}/checkpoints/pgot_oneshot_memory8_register32_noprior/checkpoint-5000}"
    --train_annotations "${TRAIN_ANNOTATIONS:-${PROJECT_ROOT}/data/pgot_pix2cap_thing_train.jsonl}"
    --selector_model "${SELECTOR_MODEL:-/home/jovyan/.cache/huggingface/hub/models--Qwen--Qwen3-VL-4B-Instruct/snapshots/ebb281ec70b05090aa6165b016eac8ec08e71b17}"
    --max_refs "${MAX_REFS:-0}" --max_samples "${MAX_SAMPLES:-0}"
    --seed "${SEED:-0}" --dtype fp32 --eval_size 512
    --ar_max_new_tokens "${AR_MAX_NEW_TOKENS:-2048}"
    --max_caption_tokens "${MAX_CAPTION_TOKENS:-4096}"
    --selector_max_new_tokens "${SELECTOR_MAX_NEW_TOKENS:-64}"
    --selector_max_input_tokens "${SELECTOR_MAX_INPUT_TOKENS:-8192}"
    --save_examples "${SAVE_EXAMPLES:-24}" --cpu_threads "${CPU_THREADS:-3}"
    --presence_iou "${PRESENCE_IOU:-0.1}"
    --ctrlo_repo "${CTRLO_REPO:-/home/jovyan/baselines/CTRL-O}"
    --checkpoint_dir "${CTRLO_CHECKPOINT_DIR:-/home/jovyan/baselines/ckpt/ctrlo}"
    --llm2vec_base "${LLM2VEC_BASE:-McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp}"
    --llm2vec_adapter "${LLM2VEC_ADAPTER:-McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp-unsup-simcse}"
    --embedding_batch_size "${EMBEDDING_BATCH_SIZE:-4}"
    --embedding_max_length "${EMBEDDING_MAX_LENGTH:-1024}")
if [[ "${WITH_CTRLO}" == 1 ]]; then COMMON+=(--with_ctrlo); fi
if [[ "${CACHE_BUNDLES:-0}" == 1 ]]; then COMMON+=(--cache_bundles); fi
run_stage() {
    local stage=$1 interpreter=$2
    shift 2
    local tag="${stage}" stage_pythonpath="${PROJECT_ROOT}:${PYTHONPATH:-}"
    if [[ "${stage}" == select ]]; then
        tag="select_${2}"
        stage_pythonpath="${SELECTOR_SITE_PACKAGES}:${stage_pythonpath}"
    fi
    CUDA_VISIBLE_DEVICES="${GPU}" PYTHONNOUSERSITE=1 \
    PYTHONPATH="${stage_pythonpath}" \
    LD_LIBRARY_PATH="$(dirname "${interpreter}")/../lib:${LD_LIBRARY_PATH:-}" \
    OMP_NUM_THREADS=3 MKL_NUM_THREADS=3 TOKENIZERS_PARALLELISM=false \
    "${interpreter}" -u -m pgot.eval.run_refcoco_object_selection \
        --stage "${stage}" "${COMMON[@]}" "$@" 2>&1 | tee -a "${OUTPUT_DIR}/logs/${tag}.log"
}
trap 'echo "Failed; see ${OUTPUT_DIR}/logs. Completed caches are resumable." >&2' ERR
run_stage prepare "${DEOR_PYTHON}"
if [[ "${WITH_CTRLO}" == 1 ]]; then run_stage preflight "${CTRLO_PYTHON}"; fi
if [[ "${PREFLIGHT_ONLY:-0}" == 1 ]]; then exit 0; fi
run_stage extract "${DEOR_PYTHON}"
run_stage select "${SELECTOR_PYTHON}" --condition category
run_stage select "${SELECTOR_PYTHON}" --condition description
if [[ "${WITH_CTRLO}" == 1 ]]; then
    run_stage encode "${CTRLO_PYTHON}"
    run_stage ctrlo "${CTRLO_PYTHON}"
fi
run_stage score "${DEOR_PYTHON}"
echo "Completed: ${OUTPUT_DIR}/results.md and comparison.csv"
