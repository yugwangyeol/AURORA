#!/usr/bin/env bash
# Train O1/R32 on GPUs 1,0 and O4/R32 on GPUs 2,3 concurrently.
# After each training job completes, split its two GPUs across rFID and CompFID.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MAX_STEPS="${MAX_STEPS:-5000}"
BASE_MODEL_PATH="${BASE_MODEL_PATH:-${PROJECT_ROOT}/checkpoints/pgot_e11_capacity_dit16/checkpoint-10000}"

run_experiment() {
    local memory="$1"
    local devices="$2"
    local master_port="$3"
    local train_script="${PROJECT_ROOT}/scripts/train_pgot_oneshot_memory${memory}_register32_noprior.sh"
    local train_root="${PROJECT_ROOT}/checkpoints/pgot_oneshot_memory${memory}_register32_noprior"
    local eval_root="${PROJECT_ROOT}/outputs/eval_pgot_oneshot_memory${memory}_register32_noprior"
    local trained_model

    test -f "${train_script}" || {
        echo "Missing training script: ${train_script}" >&2
        return 1
    }
    if [[ -e "${train_root}/config.json" || \
          -n "$(find "${train_root}" -maxdepth 1 -name 'checkpoint-*' -print -quit 2>/dev/null || true)" ]]; then
        echo "Training output already contains a model: ${train_root}" >&2
        echo "Choose a fresh output path or move the existing run before retrying." >&2
        return 2
    fi

    mkdir -p "${eval_root}/logs"
    echo "START O${memory}/R32 training on physical GPUs ${devices}"
    CUDA_VISIBLE_DEVICES="${devices}" \
    MODEL_PATH="${BASE_MODEL_PATH}" OUTPUT_DIR="${train_root}" \
    MAX_STEPS="${MAX_STEPS}" MASTER_PORT="${master_port}" \
    bash "${train_script}" 2>&1 | tee "${eval_root}/logs/train.log"

    trained_model="${train_root}/checkpoint-${MAX_STEPS}"
    if [[ ! -f "${trained_model}/config.json" ]]; then
        trained_model="${train_root}"
    fi
    test -f "${trained_model}/config.json" || {
        echo "Training finished without a loadable checkpoint under ${train_root}" >&2
        return 1
    }

    echo "START O${memory}/R32 parallel evaluation on physical GPUs ${devices}"
    CUDA_VISIBLE_DEVICES="${devices}" OBJECT_MEMORIES="${memory}" \
    MODEL_PATH="${trained_model}" EVAL_ROOT="${eval_root}" \
    bash "${PROJECT_ROOT}/scripts/eval_pgot_object_memory_register32_noprior_parallel_2gpu.sh"
}

run_experiment 1 "${MEMORY1_GPUS:-1,0}" "${MEMORY1_MASTER_PORT:-29711}" &
pids=("$!")
names=("O1/R32")

run_experiment 4 "${MEMORY4_GPUS:-2,3}" "${MEMORY4_MASTER_PORT:-29714}" &
pids+=("$!")
names+=("O4/R32")

failed=0
for index in "${!pids[@]}"; do
    if wait "${pids[$index]}"; then
        echo "PASS ${names[$index]} train -> rFID/CompFID"
    else
        status=$?
        echo "FAIL ${names[$index]} status=${status}" >&2
        failed=1
    fi
done
(( failed == 0 )) || exit 1

echo "Completed both object-memory experiments."
