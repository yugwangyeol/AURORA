#!/usr/bin/env bash
# AR discovery first, then fixed class/point selection; no GT-mask selection.
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export BINDING_ROUTE=ar_router
export QUERY_MODE="${QUERY_MODE:-class_point_natural}"
bash "${PROJECT_ROOT}/scripts/eval_pgot_memory8_binding_hits.sh" "$@"
