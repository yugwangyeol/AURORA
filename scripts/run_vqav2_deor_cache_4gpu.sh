#!/usr/bin/env bash
# Prepare VQAv2 and build the frozen DeOR object cache on four GPUs.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
IFS=, read -r -a DEVICES <<< "${CUDA_VISIBLE_DEVICES}"
if (( ${#DEVICES[@]} != 4 )); then
  echo 'CUDA_VISIBLE_DEVICES must contain exactly four GPUs.' >&2
  exit 2
fi
declare -A SEEN=()
for device in "${DEVICES[@]}"; do
  if [[ -n "${SEEN[$device]:-}" ]]; then
    echo 'CUDA_VISIBLE_DEVICES must contain four distinct GPUs.' >&2
    exit 2
  fi
  SEEN[$device]=1
done

PYTHON="${PYTHON:-/home/jovyan/.conda/envs/scale_rae/bin/python}"
VQAV2_ROOT="${VQAV2_ROOT:-/home/jovyan/data/vqav2}"
PREPARED_ROOT="${PREPARED_ROOT:-${VQAV2_ROOT}/prepared}"
CACHE_ROOT="${CACHE_ROOT:-${VQAV2_ROOT}/deor_object_cache_m8_r32}"
MODEL_PATH="${MODEL_PATH:-${PROJECT_ROOT}/checkpoints/pgot_oneshot_memory8_register32_noprior/checkpoint-5000}"
BATCH_SIZE="${BATCH_SIZE:-8}"
CPU_THREADS="${CPU_THREADS:-3}"
AR_MAX_NEW_TOKENS="${AR_MAX_NEW_TOKENS:-2048}"
SPLITS="${SPLITS:-train val}"

test -x "${PYTHON}" || { echo "Missing Python: ${PYTHON}" >&2; exit 1; }
test -s "${MODEL_PATH}/config.json" || { echo "Missing model config: ${MODEL_PATH}" >&2; exit 1; }
test -s "${MODEL_PATH}/model.safetensors.index.json" || {
  echo "Missing model index: ${MODEL_PATH}" >&2
  exit 1
}
for file in \
  "${VQAV2_ROOT}/v2_OpenEnded_mscoco_train2014_questions.json" \
  "${VQAV2_ROOT}/v2_mscoco_train2014_annotations.json" \
  "${VQAV2_ROOT}/v2_OpenEnded_mscoco_val2014_questions.json" \
  "${VQAV2_ROOT}/v2_mscoco_val2014_annotations.json"; do
  test -s "${file}" || { echo "Missing VQAv2 file: ${file}" >&2; exit 1; }
done
for directory in "${VQAV2_ROOT}/train2014" "${VQAV2_ROOT}/val2014"; do
  test -d "${directory}" || { echo "Missing COCO directory: ${directory}" >&2; exit 1; }
done
[[ "${BATCH_SIZE}" =~ ^[1-9][0-9]*$ ]] || {
  echo 'BATCH_SIZE must be a positive integer.' >&2
  exit 2
}
[[ "${CPU_THREADS}" =~ ^[1-9][0-9]*$ ]] || {
  echo 'CPU_THREADS must be a positive integer.' >&2
  exit 2
}

# The image cache is estimated at roughly 29 GiB.  Keep an additional margin
# for manifests, logs, and temporary writes.
AVAILABLE_KIB="$(df -Pk "${VQAV2_ROOT}" | awk 'NR == 2 {print $4}')"
MINIMUM_KIB=$(( 40 * 1024 * 1024 ))
if (( AVAILABLE_KIB < MINIMUM_KIB )); then
  echo "At least 40 GiB free space is required; available KiB=${AVAILABLE_KIB}." >&2
  exit 1
fi

manifests_are_complete() {
  PREPARED_ROOT="${PREPARED_ROOT}" PYTHONNOUSERSITE=1 PYTHONPATH="${PROJECT_ROOT}" \
    "${PYTHON}" - <<'PY'
import json
import os
from pathlib import Path

root = Path(os.environ["PREPARED_ROOT"]) / "manifests"
expected = {"train": (82783, 443757), "val": (40504, 214354)}
for split, (num_images, num_questions) in expected.items():
    summary_path = root / f"summary_{split}.json"
    images_path = root / f"images_{split}.jsonl"
    questions_path = root / f"questions_{split}.jsonl"
    if not all(path.is_file() and path.stat().st_size > 0 for path in (
        summary_path, images_path, questions_path
    )):
        raise SystemExit(1)
    summary = json.loads(summary_path.read_text())
    if summary.get("format_version") != "deor-vqav2-object-cache-v1":
        raise SystemExit(1)
    if summary.get("num_images") != num_images or summary.get("num_questions") != num_questions:
        raise SystemExit(1)
    if summary.get("extractor_receives_questions_or_answers") is not False:
        raise SystemExit(1)
    with images_path.open() as stream:
        if sum(1 for line in stream if line.strip()) != num_images:
            raise SystemExit(1)
    with questions_path.open() as stream:
        if sum(1 for line in stream if line.strip()) != num_questions:
            raise SystemExit(1)
PY
}

if manifests_are_complete; then
  echo "Using verified existing VQAv2 manifests: ${PREPARED_ROOT}/manifests"
else
  echo 'Preparing complete VQAv2 train/val manifests.'
  VQAV2_ROOT="${VQAV2_ROOT}" PREPARED_ROOT="${PREPARED_ROOT}" \
    PYTHON="${PYTHON}" SPLITS='train val' CPU_THREADS="${CPU_THREADS}" \
    bash "${PROJECT_ROOT}/scripts/prepare_vqav2_deor_cache.sh"
  manifests_are_complete
  echo 'VQAv2 manifests prepared and verified.'
fi

read -r -a SPLIT_LIST <<< "${SPLITS}"
(( ${#SPLIT_LIST[@]} > 0 )) || { echo 'SPLITS cannot be empty.' >&2; exit 2; }
for split in "${SPLIT_LIST[@]}"; do
  case "${split}" in
    train|val) ;;
    *) echo "Unknown VQAv2 split: ${split}" >&2; exit 2 ;;
  esac
  test -s "${PREPARED_ROOT}/manifests/images_${split}.jsonl" || {
    echo "Missing image manifest for ${split}." >&2
    exit 1
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
  echo "Starting VQAv2 ${split}: four resumable shards, batch=${BATCH_SIZE}"
  pids=()
  for shard in 0 1 2 3; do
    GPU="${DEVICES[$shard]}" SPLIT="${split}" NUM_SHARDS=4 SHARD_INDEX="${shard}" \
      PREPARED_ROOT="${PREPARED_ROOT}" CACHE_ROOT="${CACHE_ROOT}" \
      MODEL_PATH="${MODEL_PATH}" BATCH_SIZE="${BATCH_SIZE}" \
      CPU_THREADS="${CPU_THREADS}" AR_MAX_NEW_TOKENS="${AR_MAX_NEW_TOKENS}" \
      PYTHON="${PYTHON}" \
      bash "${PROJECT_ROOT}/scripts/cache_vqav2_deor_objects.sh" &
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
    LD_LIBRARY_PATH="$(dirname "${PYTHON}")/../lib:${LD_LIBRARY_PATH:-}" \
    "${PYTHON}" -m pgot.eval.vqav2_object_cache verify \
      --images_manifest "${PREPARED_ROOT}/manifests/images_${split}.jsonl" \
      --cache_root "${CACHE_ROOT}" \
      --summary_path "${CACHE_ROOT}/verification_${split}.json" \
      --require_complete
  echo "Completed and verified VQAv2 ${split}."
done

echo "Completed all requested VQAv2 splits: ${SPLITS}"
