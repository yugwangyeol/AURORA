#!/usr/bin/env bash
# Sequential full runs: Capacity train/eval, then Query-Separation train/eval.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p "${PROJECT_ROOT}/outputs"

echo "[1/4] E11 Capacity Only training"
bash "${PROJECT_ROOT}/scripts/train_pgot_e11_capacity.sh" \
    2>&1 | tee "${PROJECT_ROOT}/outputs/train_pgot_e11_capacity.log"

echo "[2/4] E11 Capacity Only evaluation"
bash "${PROJECT_ROOT}/scripts/eval_pgot_e11_capacity.sh" \
    2>&1 | tee "${PROJECT_ROOT}/outputs/eval_pgot_e11_capacity.log"

echo "[3/4] E11 Capacity + Query Separation training"
bash "${PROJECT_ROOT}/scripts/train_pgot_e11_capacity_query_separation.sh" \
    2>&1 | tee "${PROJECT_ROOT}/outputs/train_pgot_e11_capacity_query_separation.log"

echo "[4/4] E11 Capacity + Query Separation evaluation"
bash "${PROJECT_ROOT}/scripts/eval_pgot_e11_capacity_query_separation.sh" \
    2>&1 | tee "${PROJECT_ROOT}/outputs/eval_pgot_e11_capacity_query_separation.log"

echo "All E11 capacity experiments completed."
