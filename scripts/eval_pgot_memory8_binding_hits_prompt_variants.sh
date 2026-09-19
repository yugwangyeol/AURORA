#!/usr/bin/env bash
# Sequential prompt comparison on one GPU. Targets/order/metric stay unchanged.
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -n "${QUERY_MODE:-}" || -n "${OUTPUT_DIR:-}" ]]; then
    echo "Use OUTPUT_ROOT for this two-mode launcher; QUERY_MODE/OUTPUT_DIR are single-mode options." >&2
    exit 2
fi
default_root="${PROJECT_ROOT}/outputs/binding_hits_pgot_memory8_register32_noprior"
if [[ "${SMOKE:-0}" == 1 ]]; then default_root="${PROJECT_ROOT}/outputs/_smoke_binding_hits_pgot"; fi
for query_mode in class_point_short class_point_natural; do
    echo "Evaluating Binding Hits: ${query_mode}"
    QUERY_MODE="${query_mode}" OUTPUT_DIR="${OUTPUT_ROOT:-${default_root}}/${query_mode}" \
        bash "${PROJECT_ROOT}/scripts/eval_pgot_memory8_binding_hits.sh" "$@"
done
