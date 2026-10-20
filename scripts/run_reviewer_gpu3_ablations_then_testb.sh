#!/usr/bin/env bash
# GPU 3: w/o Reader-supervision transfer -> w/o grounding locality -> testB.
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GPU="${GPU:-3}"
WOREADER="${PROJECT_ROOT}/checkpoints/pgot_oneshot_memory8_register32_noprior_woreader/checkpoint-5000"
WOGROUND="${PROJECT_ROOT}/checkpoints/pgot_oneshot_memory8_register32_noprior_wogrounding/checkpoint-5000"
EDIT="${PROJECT_ROOT}/outputs/reviewer_object_transfer_woreader"
LOCAL="${PROJECT_ROOT}/outputs/reviewer_writer_locality_wogrounding"
E2E="${PROJECT_ROOT}/outputs/refcoco_object_selection_memory8_register32_nonoverlap_testB"

if [[ ! -f "${EDIT}/summary.json" ]]; then
    CUDA_VISIBLE_DEVICES="${GPU}" MODEL_PATH="${WOREADER}" OUTPUT_DIR="${EDIT}" \
        bash "${PROJECT_ROOT}/scripts/eval_object_memory_transfer.sh" \
        2>&1 | tee "${EDIT}.log"
fi
if [[ ! -f "${LOCAL}/summary.json" ]]; then
    CUDA_VISIBLE_DEVICES="${GPU}" MODEL_PATH="${WOGROUND}" OUTPUT_DIR="${LOCAL}" \
        bash "${PROJECT_ROOT}/scripts/eval_writer_locality.sh" \
        2>&1 | tee "${LOCAL}.log"
fi
if [[ ! -f "${E2E}/summary.json" ]]; then
    GPU="${GPU}" SPLIT=testB OUTPUT_DIR="${E2E}" WITH_CTRLO=0 \
        bash "${PROJECT_ROOT}/scripts/run_refcoco_e2e_nonoverlap_split.sh" \
        2>&1 | tee "${E2E}.log"
fi
"/home/jovyan/.conda/envs/scale_rae/bin/python" \
    "${PROJECT_ROOT}/scripts/analyze_refcoco_e2e_breakdown.py" --root "${E2E}"
echo "PASS: GPU ${GPU} reviewer chain complete."
