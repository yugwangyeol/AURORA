#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export GPU="${GPU:-0}"
export OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/outputs/language_object_editing_transfer_replace_gpu0}"
export REQUESTS="${REQUESTS:-${PROJECT_ROOT}/configs/language_object_editing_transfer_replace_requests.jsonl}"
export REPLACEMENT_MODE=full_bundle
export COMPARE_REPLACEMENT_MODES=1
exec bash "${PROJECT_ROOT}/scripts/run_language_object_editing.sh"
