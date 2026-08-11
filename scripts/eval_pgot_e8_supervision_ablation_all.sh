#!/usr/bin/env bash
# Evaluate all five E8.1 supervision ablations sequentially.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CHECKPOINT_STEP="${CHECKPOINT_STEP:-10000}"

experiment_names=(
    pgot_e8_ablation_a_writer_gt_reader_writer
    pgot_e8_ablation_b_writer_gt_reader_none
    pgot_e8_ablation_c_writer_none_reader_gt
    pgot_e8_ablation_d_writer_none_reader_writer
    pgot_e8_ablation_e_writer_none_reader_none
)

for experiment_name in "${experiment_names[@]}"; do
    checkpoint="${PROJECT_ROOT}/checkpoints/${experiment_name}/checkpoint-${CHECKPOINT_STEP}"
    output="${PROJECT_ROOT}/outputs/eval_${experiment_name}"
    echo "Evaluating ${experiment_name} from checkpoint-${CHECKPOINT_STEP}"
    MODEL_PATH="${checkpoint}" OUTPUT_DIR="${output}" bash \
        "${PROJECT_ROOT}/scripts/eval_pgot_e8_visual_memory.sh"
done

echo "All five E8 supervision ablation evaluations completed."
