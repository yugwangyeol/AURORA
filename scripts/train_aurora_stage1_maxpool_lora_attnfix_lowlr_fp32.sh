#!/bin/bash
# Experiment D: max-pool LoRA low-LR fine-tune with attention-mask fixes.
#   - RAE latent queries attend bidirectionally to one another.
#   - Slots assigned to the same object attend bidirectionally within the group.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export OUTPUT_DIR="${OUTPUT_DIR:-/home/jovyan/AURORA/checkpoints/aurora_refcoco_singlephase_maxpool_lora_attnfix_lowlr_fp32}"
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-aurora_refcoco_singlephase_maxpool_lora_attnfix_lowlr_fp32}"
export MASTER_PORT="${MASTER_PORT:-29506}"
export CAPTIONSLOT_RAE_BIDIRECTIONAL="${CAPTIONSLOT_RAE_BIDIRECTIONAL:-True}"
export CAPTIONSLOT_SAME_OBJECT_SLOT_ATTENTION="${CAPTIONSLOT_SAME_OBJECT_SLOT_ATTENTION:-True}"

exec bash "${SCRIPT_DIR}/train_aurora_stage1_maxpool_lora_continue_lowlr_fp32.sh"
