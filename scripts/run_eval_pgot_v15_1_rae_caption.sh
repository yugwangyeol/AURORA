#!/bin/bash
# PGOT v15.1 RAE-caption ablation eval. Uses the same V15.1 threshold readout.
set -euo pipefail

export MODEL_PATH="${MODEL_PATH:-/home/jovyan/PGOT/checkpoints/pgot_main_v15_1_rae_caption}"
export OUTPUT_DIR="${OUTPUT_DIR:-/home/jovyan/PGOT/outputs/eval_pgot_v15_1_rae_caption}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "${SCRIPT_DIR}/run_eval_pgot_v15_1.sh" "$@"
