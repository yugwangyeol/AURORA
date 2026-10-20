#!/usr/bin/env bash
# O16/R32 compositional FID under the same 4,720-image protocol as O4/O8.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export MODEL_PATH="${MODEL_PATH:-${PROJECT_ROOT}/checkpoints/pgot_oneshot_memory16_register32_noprior/checkpoint-5000}"
export OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/outputs/compgen_pgot_oneshot_memory16_register32_noprior}"
export VAL_JSONL="${VAL_JSONL:-${PROJECT_ROOT}/data/pgot_pix2cap_generated_val5k.jsonl}"
export DTYPE="${DTYPE:-fp32}"
export BATCH_SIZE="${COMP_BATCH_SIZE:-4}"
export EXPECTED_SAMPLES="${EXPECTED_SAMPLES:-4720}"
export METRIC_RESOLUTION="${METRIC_RESOLUTION:-0}"
export MIX_MODE="${MIX_MODE:-random_slots}"
export REGISTER_SOURCE="${REGISTER_SOURCE:-random}"
export SEED="${SEED:-1234}"
export GUIDANCE_SCALE="${GUIDANCE_SCALE:-1.0}"
export DIFFUSION_INFERENCE_STEPS="${DIFFUSION_INFERENCE_STEPS:-10}"

PYTHON="${PYTHON:-/home/jovyan/.conda/envs/scale_rae/bin/python}"
test -f "${MODEL_PATH}/config.json" || {
    echo "Missing O16/R32 checkpoint: ${MODEL_PATH}" >&2
    exit 1
}
PYTHONNOUSERSITE=1 PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}" \
LD_LIBRARY_PATH="$(dirname "${PYTHON}")/../lib:${LD_LIBRARY_PATH:-}" \
"${PYTHON}" -m pgot.eval.verify_oneshot_memory_config \
    --model_path "${MODEL_PATH}" --model_size 1.5b \
    --object_memories 16 --register_memories 32

exec bash "${PROJECT_ROOT}/scripts/eval_pgot_compositional_gen.sh" "$@"
