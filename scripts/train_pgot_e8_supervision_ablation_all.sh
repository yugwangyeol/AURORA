#!/usr/bin/env bash
# Train all five E8.1 supervision ablations sequentially; each run uses two GPUs.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export NUM_GPUS="${NUM_GPUS:-2}"

for experiment in A B C D E; do
    echo "Starting E8 supervision ablation ${experiment}/E"
    EXPERIMENT="${experiment}" bash \
        "${PROJECT_ROOT}/scripts/train_pgot_e8_supervision_ablation.sh"
done

echo "All five E8 supervision ablation training runs completed."
