#!/usr/bin/env bash
# Resume/complete the frozen DeOR object cache on four GPUs.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
IFS=, read -r -a DEVICES <<< "${CUDA_VISIBLE_DEVICES}"
if (( ${#DEVICES[@]} != 4 )); then
    echo 'Set CUDA_VISIBLE_DEVICES to exactly four GPUs.' >&2
    exit 2
fi
if [[ "${DEVICES[0]}" == "${DEVICES[1]}" || "${DEVICES[0]}" == "${DEVICES[2]}" || \
      "${DEVICES[0]}" == "${DEVICES[3]}" || "${DEVICES[1]}" == "${DEVICES[2]}" || \
      "${DEVICES[1]}" == "${DEVICES[3]}" || "${DEVICES[2]}" == "${DEVICES[3]}" ]]; then
    echo 'CUDA_VISIBLE_DEVICES must contain four distinct GPUs.' >&2
    exit 2
fi

PREPARED_ROOT="${PREPARED_ROOT:-/home/jovyan/data/gqa/prepared_balanced}"
CACHE_ROOT="${CACHE_ROOT:-/home/jovyan/data/gqa/deor_object_cache_m8_r32_bs8}"
MODEL_PATH="${MODEL_PATH:-${PROJECT_ROOT}/checkpoints/pgot_oneshot_memory8_register32_noprior/checkpoint-5000}"
DEOR_PYTHON="${DEOR_PYTHON:-/home/jovyan/.conda/envs/scale_rae/bin/python}"
BATCH_SIZE="${BATCH_SIZE:-8}"
CPU_THREADS="${CPU_THREADS:-3}"
SPLITS="${SPLITS:-train val testdev}"

test -f "${MODEL_PATH}/config.json" || { echo "Missing model: ${MODEL_PATH}" >&2; exit 1; }
read -r -a SPLIT_LIST <<< "${SPLITS}"
(( ${#SPLIT_LIST[@]} > 0 )) || { echo 'SPLITS cannot be empty.' >&2; exit 2; }
for split in "${SPLIT_LIST[@]}"; do
    case "${split}" in train|val|testdev) ;; *) echo "Unknown GQA split: ${split}" >&2; exit 2 ;; esac
    test -s "${PREPARED_ROOT}/manifests/images_${split}.jsonl" || {
        echo "Missing prepared manifest for ${split}." >&2; exit 1;
    }
done

mkdir -p "${CACHE_ROOT}/logs"
pids=()
stop_children() {
    local pid
    for pid in "${pids[@]:-}"; do kill "${pid}" 2>/dev/null || true; done
    for pid in "${pids[@]:-}"; do wait "${pid}" 2>/dev/null || true; done
}
trap 'stop_children; exit 130' INT TERM

for split in "${SPLIT_LIST[@]}"; do
    echo "Starting GQA ${split}: four resumable shards, batch=${BATCH_SIZE}"
    pids=()
    for shard in 0 1 2 3; do
        GPU="${DEVICES[$shard]}" SPLIT="${split}" NUM_SHARDS=4 SHARD_INDEX="${shard}" \
        PREPARED_ROOT="${PREPARED_ROOT}" CACHE_ROOT="${CACHE_ROOT}" MODEL_PATH="${MODEL_PATH}" \
        BATCH_SIZE="${BATCH_SIZE}" CPU_THREADS="${CPU_THREADS}" DEOR_PYTHON="${DEOR_PYTHON}" \
          bash "${PROJECT_ROOT}/scripts/cache_gqa_deor_objects.sh" &
        pids+=("$!")
    done

    failed=0
    for shard in 0 1 2 3; do
        if wait "${pids[$shard]}"; then
            echo "PASS ${split} shard=${shard} gpu=${DEVICES[$shard]}"
        else
            status=$?
            echo "FAIL ${split} shard=${shard} gpu=${DEVICES[$shard]} status=${status}" >&2
            failed=1
        fi
    done
    pids=()
    (( failed == 0 )) || exit 1

    PYTHONNOUSERSITE=1 PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}" \
    LD_LIBRARY_PATH="$(dirname "${DEOR_PYTHON}")/../lib:${LD_LIBRARY_PATH:-}" \
      "${DEOR_PYTHON}" -m pgot.eval.gqa_object_cache verify \
        --images_manifest "${PREPARED_ROOT}/manifests/images_${split}.jsonl" \
        --cache_root "${CACHE_ROOT}" \
        --summary_path "${CACHE_ROOT}/verification_${split}.json" \
        --require_complete
    echo "Completed and verified GQA ${split}."
done

echo "Completed all requested GQA splits: ${SPLITS}"
