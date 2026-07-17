#!/bin/bash
# PGOT v15.1 ablation: V15.1 router bottleneck, but RAE queries can attend
# the full valid caption instead of only OVT positions.
set -euo pipefail

export OUTPUT_DIR="${OUTPUT_DIR:-/home/jovyan/PGOT/checkpoints/pgot_main_v15_1_rae_caption}"
export WANDB_NAME="${WANDB_NAME:-pgot_main_v15_1_rae_caption}"
export PGOT_RAE_ATTENDS_CAPTION="${PGOT_RAE_ATTENDS_CAPTION:-True}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "${SCRIPT_DIR}/train_pgot_v15_1.sh" "$@"
