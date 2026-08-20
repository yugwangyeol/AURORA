#!/usr/bin/env bash
# Canonical E8.2 smoke entry point. The shared smoke also creates a one-step
# E8.1-A checkpoint so checkpoint hand-off is tested end to end.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec bash "${PROJECT_ROOT}/scripts/smoke_pgot_e8_visual_memory.sh"
