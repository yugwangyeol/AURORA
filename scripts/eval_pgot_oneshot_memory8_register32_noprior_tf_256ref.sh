#!/usr/bin/env bash
# Evaluate native 224px reconstructions after bicubic upsampling against
# independently loaded CODA-256 references. This is a 256-reference metric,
# not native 256 generation.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export MODEL_PATH="${MODEL_PATH:-${PROJECT_ROOT}/checkpoints/pgot_oneshot_memory8_register32_noprior/checkpoint-5000}"
export OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/outputs/eval_pgot_oneshot_memory8_register32_noprior/tf_recon_256ref}"
export COMPUTE_RFID="${COMPUTE_RFID:-True}"
export COMPUTE_KID="${COMPUTE_KID:-True}"

exec bash "${PROJECT_ROOT}/scripts/eval_pgot_oneshot_memory8_register32_noprior_tf.sh" \
    --recon_metric_resolution 256 "$@"
