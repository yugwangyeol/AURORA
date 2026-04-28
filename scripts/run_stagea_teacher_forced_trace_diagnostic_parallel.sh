#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

OUTPUT_DIR="${OUTPUT_DIR:-/home/jovyan/AURORA/outputs/stagea_teacher_forced_trace_diagnostic_train2017}"
SHARD_ROOT="${OUTPUT_DIR}/shards"
LOG_DIR="${OUTPUT_DIR}/logs"
GPU0="${GPU0:-0}"
GPU1="${GPU1:-1}"
PYTHON_BIN="${PYTHON_BIN:-/home/jovyan/.conda/envs/scale_rae/bin/python}"

mkdir -p "${SHARD_ROOT}" "${LOG_DIR}"

cleanup() {
  jobs -pr | xargs -r kill || true
}

trap cleanup INT TERM

GPU="${GPU0}" \
NUM_SHARDS=2 \
SHARD_INDEX=0 \
OUTPUT_DIR="${SHARD_ROOT}/shard_0_of_2" \
bash "${SCRIPT_DIR}/run_stagea_teacher_forced_trace_diagnostic.sh" \
  > "${LOG_DIR}/shard_0_of_2.log" 2>&1 &
pid0=$!

GPU="${GPU1}" \
NUM_SHARDS=2 \
SHARD_INDEX=1 \
OUTPUT_DIR="${SHARD_ROOT}/shard_1_of_2" \
bash "${SCRIPT_DIR}/run_stagea_teacher_forced_trace_diagnostic.sh" \
  > "${LOG_DIR}/shard_1_of_2.log" 2>&1 &
pid1=$!

echo "Stage A teacher-forced trace logs:"
echo "  ${LOG_DIR}/shard_0_of_2.log"
echo "  ${LOG_DIR}/shard_1_of_2.log"

set +e
wait "${pid0}"
status0=$?
wait "${pid1}"
status1=$?
set -e

if [[ "${status0}" -ne 0 || "${status1}" -ne 0 ]]; then
  echo "Stage A teacher-forced trace shard run failed. Check logs in ${LOG_DIR}" >&2
  exit 1
fi

trap - INT TERM

"${PYTHON_BIN}" "${SCRIPT_DIR}/merge_stagea_generation_trace_diagnostic_shards.py" \
  --root-output-dir "${OUTPUT_DIR}" \
  --shard-dir "${SHARD_ROOT}/shard_0_of_2" \
  --shard-dir "${SHARD_ROOT}/shard_1_of_2"
