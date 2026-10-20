#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-/home/jovyan/.conda/envs/scale_rae/bin/python}"
SMOKE_ROOT="${SMOKE_ROOT:-$(mktemp -d "${PROJECT_ROOT}/.smoke_vqav2_deor_cache.XXXXXX")}"
trap 'echo "Smoke failed; outputs preserved at ${SMOKE_ROOT}" >&2' ERR

VQAV2_ROOT="${VQAV2_ROOT:-/home/jovyan/data/vqav2}" \
PREPARED_ROOT="${SMOKE_ROOT}/prepared" SPLITS=val MAX_IMAGES=2 \
  bash "${PROJECT_ROOT}/scripts/prepare_vqav2_deor_cache.sh"

GPU="${GPU:-0}" PREPARED_ROOT="${SMOKE_ROOT}/prepared" \
CACHE_ROOT="${SMOKE_ROOT}/cache" SPLIT=val MAX_IMAGES=2 BATCH_SIZE=2 \
AR_MAX_NEW_TOKENS="${AR_MAX_NEW_TOKENS:-512}" \
  bash "${PROJECT_ROOT}/scripts/cache_vqav2_deor_objects.sh"

PYTHONNOUSERSITE=1 PYTHONPATH="${PROJECT_ROOT}" \
  "${PYTHON}" -m pgot.eval.vqav2_object_cache verify \
    --images_manifest "${SMOKE_ROOT}/prepared/manifests/images_val.jsonl" \
    --cache_root "${SMOKE_ROOT}/cache" --max_images 2 --require_complete \
    --summary_path "${SMOKE_ROOT}/verification.json"

# Audit the hard separation between image-only extraction and VQA annotations.
SMOKE_ROOT="${SMOKE_ROOT}" PYTHONNOUSERSITE=1 PYTHONPATH="${PROJECT_ROOT}" \
  "${PYTHON}" - <<'PY'
import json
import os
from pathlib import Path

import torch

from pgot.eval.gqa_object_cache import cache_path

root = Path(os.environ["SMOKE_ROOT"])
summary = json.loads((root / "prepared/manifests/summary_val.json").read_text())
assert summary["extractor_receives_questions_or_answers"] is False
assert "question" not in summary["extractor_input_fields"]
assert "answer" not in summary["extractor_input_fields"]
image = json.loads((root / "prepared/manifests/images_val.jsonl").read_text().splitlines()[0])
bundle = torch.load(
    cache_path(root / "cache", "val", image["image_id"]),
    map_location="cpu",
    weights_only=False,
)
assert bundle["format_version"] == "deor-vqav2-object-cache-v1"
assert not ({"question", "answer", "answers"} & set(bundle))
assert bundle["semantic"].shape[0] == len(bundle["objects"])
assert bundle["visual_memory"].shape[1:] == (8, 1536)
print("PASS VQAv2 annotation isolation and compact cache audit")
PY

# A complete cache must resume without loading the model again.
GPU="${GPU:-0}" PREPARED_ROOT="${SMOKE_ROOT}/prepared" \
CACHE_ROOT="${SMOKE_ROOT}/cache" SPLIT=val MAX_IMAGES=2 BATCH_SIZE=2 \
AR_MAX_NEW_TOKENS="${AR_MAX_NEW_TOKENS:-512}" \
  bash "${PROJECT_ROOT}/scripts/cache_vqav2_deor_objects.sh"

if [[ "${KEEP_SMOKE:-0}" == 1 ]]; then
  echo "Smoke retained at ${SMOKE_ROOT}"
else
  case "${SMOKE_ROOT}" in
    "${PROJECT_ROOT}/.smoke_vqav2_deor_cache."*) find "${SMOKE_ROOT}" -depth -delete ;;
    *) echo 'Custom smoke root retained; arbitrary paths are never auto-deleted.' ;;
  esac
fi
echo 'PASS VQAv2 DeOR cache smoke'
