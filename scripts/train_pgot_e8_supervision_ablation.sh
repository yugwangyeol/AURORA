#!/usr/bin/env bash
# Train one E8.1 Writer/Reader supervision ablation (A-E).
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EXPERIMENT="${EXPERIMENT:-${1:-}}"

case "${EXPERIMENT^^}" in
    A)
        EXPERIMENT_NAME="pgot_e8_ablation_a_writer_gt_reader_writer"
        E8_OWNER_WEIGHT=1.0
        E8_READER_SUPERVISION_MODE=writer
        E8_READER_OBJECT_WEIGHT=0.5
        E8_READER_BACKGROUND_WEIGHT=0.25
        ;;
    B)
        EXPERIMENT_NAME="pgot_e8_ablation_b_writer_gt_reader_none"
        E8_OWNER_WEIGHT=1.0
        E8_READER_SUPERVISION_MODE=none
        E8_READER_OBJECT_WEIGHT=0.0
        E8_READER_BACKGROUND_WEIGHT=0.0
        ;;
    C)
        EXPERIMENT_NAME="pgot_e8_ablation_c_writer_none_reader_gt"
        E8_OWNER_WEIGHT=0.0
        E8_READER_SUPERVISION_MODE=gt
        E8_READER_OBJECT_WEIGHT=0.5
        E8_READER_BACKGROUND_WEIGHT=0.25
        ;;
    D)
        EXPERIMENT_NAME="pgot_e8_ablation_d_writer_none_reader_writer"
        E8_OWNER_WEIGHT=0.0
        E8_READER_SUPERVISION_MODE=writer
        E8_READER_OBJECT_WEIGHT=0.5
        E8_READER_BACKGROUND_WEIGHT=0.25
        ;;
    E)
        EXPERIMENT_NAME="pgot_e8_ablation_e_writer_none_reader_none"
        E8_OWNER_WEIGHT=0.0
        E8_READER_SUPERVISION_MODE=none
        E8_READER_OBJECT_WEIGHT=0.0
        E8_READER_BACKGROUND_WEIGHT=0.0
        ;;
    *)
        echo "Usage: EXPERIMENT=A|B|C|D|E bash $0" >&2
        exit 2
        ;;
esac

export E8_OWNER_WEIGHT E8_READER_SUPERVISION_MODE
export E8_READER_OBJECT_WEIGHT E8_READER_BACKGROUND_WEIGHT
export E8_CAUSAL_ENABLE=False
export OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/checkpoints/${EXPERIMENT_NAME}}"
export WANDB_NAME="${WANDB_NAME:-${EXPERIMENT_NAME}}"

echo "E8 supervision ablation ${EXPERIMENT^^}: ${EXPERIMENT_NAME}"
exec bash "${PROJECT_ROOT}/scripts/train_pgot_e8_visual_memory.sh"
