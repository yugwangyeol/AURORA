#!/bin/bash
# Smoke test for the V15 readout sweep. Runs a tiny sweep and removes artifacts
# after validating the output JSON/log.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
OUTPUT_DIR="${OUTPUT_DIR:-/home/jovyan/PGOT/outputs/smoke_sweep_readout_pgot_v15}"

rm -rf "${OUTPUT_DIR}"

OUTPUT_DIR="${OUTPUT_DIR}" \
MAX_SAMPLES=2 \
BATCH_SIZE=1 \
NUM_WORKERS=1 \
BG_THRESHOLDS="0.03,0.05" \
TEMPS="0.5" \
MERGES="mean" \
COMPETITIONS="sigmoid" \
USE_BGS="1,0" \
TOPK_FULL_METRIC=2 \
bash "${PROJECT_ROOT}/scripts/run_sweep_readout_pgot_v15.sh"

test -f "${OUTPUT_DIR}/sweep_readout.json"
test -f "${OUTPUT_DIR}/sweep.log"
grep -q "LoRA re-loaded" "${OUTPUT_DIR}/sweep.log"
grep -q "BEST fARI" "${OUTPUT_DIR}/sweep.log"
grep -q '"n_samples": 2' "${OUTPUT_DIR}/sweep_readout.json"
grep -q '"image_preprocess_mode": "coda_center_crop"' "${OUTPUT_DIR}/sweep_readout.json"
grep -q '"results"' "${OUTPUT_DIR}/sweep_readout.json"

rm -rf "${OUTPUT_DIR}"
test ! -e "${OUTPUT_DIR}"
echo "V15 readout sweep smoke complete; smoke artifacts removed."
