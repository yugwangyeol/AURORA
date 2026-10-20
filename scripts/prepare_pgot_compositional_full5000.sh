#!/usr/bin/env bash
# Build and verify a full COCO-val 5,000-image teacher-forced manifest.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-/home/jovyan/.conda/envs/scale_rae/bin/python}"
SOURCE="${SOURCE:-/home/jovyan/data/pix2cap/generated_coco_val_5k/pix2cap_coco_val_generated.json}"
MASK_ROOT="${MASK_ROOT:-/home/jovyan/data/pix2cap/generated_coco_val_5k/masks}"
OUTPUT="${OUTPUT:-${PROJECT_ROOT}/data/pgot_pix2cap_generated_val_full5000.jsonl}"
STATS_OUTPUT="${STATS_OUTPUT:-${PROJECT_ROOT}/data/pgot_pix2cap_generated_val_full5000.stats.json}"
export PYTHONNOUSERSITE=1
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="$(dirname "${PYTHON}")/../lib:${LD_LIBRARY_PATH:-}"

needs_build=1
if [[ -f "${OUTPUT}" ]]; then
    line_count="$(wc -l < "${OUTPUT}")"
    if [[ "${line_count}" == 5000 ]]; then needs_build=0; fi
fi

if (( needs_build )); then
    "${PYTHON}" "${PROJECT_ROOT}/preprocess/prepare_pix2cap_thing_pgot.py" \
        --coco_root /home/jovyan/data/coco --split val \
        --input "${SOURCE}" --panoptic_root "${MASK_ROOT}" \
        --output "${OUTPUT}" --stats_output "${STATS_OUTPUT}" \
        --crop_size 512 --max_objects 50 --max_caption_tokens 1024 \
        --n_ovt_per_object 1 --retain_all_images --expected_images 5000
fi

OUTPUT="${OUTPUT}" SOURCE="${SOURCE}" "${PYTHON}" - <<'PY'
import json
import os

with open(os.environ["SOURCE"]) as handle:
    source = json.load(handle)
expected = {int(image["id"]) for image in source["images"]}
rows = []
with open(os.environ["OUTPUT"]) as handle:
    for line in handle:
        rows.append(json.loads(line))
actual = [int(row["image_id"]) for row in rows]
if len(rows) != 5000 or len(set(actual)) != 5000 or set(actual) != expected:
    raise RuntimeError(
        "Full compositional manifest failed coverage check: "
        f"rows={len(rows)} unique={len(set(actual))} "
        f"missing={len(expected - set(actual))} extra={len(set(actual) - expected)}"
    )
zero = sum(int(row.get("n_objects", 0)) == 0 for row in rows)
incomplete = sum(not bool(row.get("caption_coverage_complete", True)) for row in rows)
print(
    f"PASS full compositional manifest: images=5000 unique=5000 "
    f"zero_object={zero} incomplete_caption_coverage={incomplete}"
)
PY
