#!/usr/bin/env bash
# Run LISA and GLaMM-RefSeg concurrently, then score both with DeOR/CTRL-O.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

PYTHON="${BASELINE_PYTHON:-/home/jovyan/.conda/envs/scale_rae/bin/python}"
LISA_REPO="${LISA_REPO:-/home/jovyan/baselines/LISA}"
GLAMM_REPO="${GLAMM_REPO:-/home/jovyan/baselines/groundingLMM}"
SAM_CHECKPOINT="${SAM_CHECKPOINT:-/home/jovyan/baselines/checkpoints/sam_vit_h_4b8939.pth}"
MANIFEST="${MANIFEST:-${PROJECT_ROOT}/outputs/eval_deor_ctrlo_refcoco_direct/manifest.json}"
TRAIN_JSONL="${TRAIN_JSONL:-${PROJECT_ROOT}/data/pgot_pix2cap_thing_train.jsonl}"
SOURCE_ROOT="${SOURCE_ROOT:-${PROJECT_ROOT}/outputs/eval_deor_ctrlo_refcoco_direct}"
SUBSET="${SUBSET:-nonoverlap}"
MAX_SAMPLES="${MAX_SAMPLES:-0}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/outputs/eval_lisa_glamm_refcoco_${SUBSET}}"
GPU_LISA="${GPU_LISA:-0}"
GPU_GLAMM="${GPU_GLAMM:-1}"
QUERIES_PER_PASS="${QUERIES_PER_PASS:-4}"

for required in "${PYTHON}" "${LISA_REPO}/model/LISA.py" "${GLAMM_REPO}/model/GLaMM.py" \
                "${SAM_CHECKPOINT}" "${MANIFEST}" "${TRAIN_JSONL}"; do
    if [[ ! -e "${required}" ]]; then
        echo "Missing required path: ${required}" >&2
        exit 2
    fi
done
mkdir -p "${OUTPUT_ROOT}/logs"

export PYTHONNOUSERSITE=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-3}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-3}"

common=(--manifest "${MANIFEST}" --train "${TRAIN_JSONL}" --subset "${SUBSET}"
        --max-samples "${MAX_SAMPLES}" --queries-per-pass "${QUERIES_PER_PASS}"
        --sam-checkpoint "${SAM_CHECKPOINT}")

echo "[$(date -u +%FT%TZ)] LISA GPU=${GPU_LISA}; GLaMM GPU=${GPU_GLAMM}; subset=${SUBSET}"
CUDA_VISIBLE_DEVICES="${GPU_LISA}" PYTHONPATH="${LISA_REPO}:${PROJECT_ROOT}" \
    "${PYTHON}" -u scripts/run_lisa_refcoco_manifest.py \
    "${common[@]}" --repo "${LISA_REPO}" --output "${OUTPUT_ROOT}/lisa" \
    > >(tee "${OUTPUT_ROOT}/logs/lisa.log") 2>&1 &
lisa_pid=$!

CUDA_VISIBLE_DEVICES="${GPU_GLAMM}" PYTHONPATH="${GLAMM_REPO}:${PROJECT_ROOT}" \
    "${PYTHON}" -u scripts/run_glamm_refcoco_manifest.py \
    "${common[@]}" --repo "${GLAMM_REPO}" --output "${OUTPUT_ROOT}/glamm" \
    > >(tee "${OUTPUT_ROOT}/logs/glamm.log") 2>&1 &
glamm_pid=$!

status=0
wait "${lisa_pid}" || { echo "LISA failed; see ${OUTPUT_ROOT}/logs/lisa.log" >&2; status=1; }
wait "${glamm_pid}" || { echo "GLaMM failed; see ${OUTPUT_ROOT}/logs/glamm.log" >&2; status=1; }
if [[ "${status}" -ne 0 ]]; then exit "${status}"; fi

models=(--model "lisa=${OUTPUT_ROOT}/lisa" --model "glamm=${OUTPUT_ROOT}/glamm")
if [[ -d "${SOURCE_ROOT}/ours" ]]; then models+=(--model "deor=${SOURCE_ROOT}/ours"); fi
if [[ -d "${SOURCE_ROOT}/ctrlo" ]]; then models+=(--model "ctrlo=${SOURCE_ROOT}/ctrlo"); fi

"${PYTHON}" -u scripts/score_mllm_refcoco_baselines.py \
    --manifest "${MANIFEST}" --train "${TRAIN_JSONL}" --subset "${SUBSET}" \
    --max-samples "${MAX_SAMPLES}" --output "${OUTPUT_ROOT}/scores" "${models[@]}" \
    2>&1 | tee "${OUTPUT_ROOT}/logs/score.log"

echo "Done: ${OUTPUT_ROOT}/scores/table.md"
