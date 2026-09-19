#!/usr/bin/env bash
# GPU 0 only by default; real DeOR + real text-only selector, both conditions.
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SMOKE_ROOT="$(mktemp -d "${PROJECT_ROOT}/.smoke_natural_object_selection.XXXXXX")"
trap 'echo "Smoke outputs retained at ${SMOKE_ROOT}" >&2' ERR
GPU="${GPU:-0}" SELECTOR_GPUS="${GPU:-0}" OUTPUT_DIR="${SMOKE_ROOT}" \
QUERIES="${PROJECT_ROOT}/configs/natural_object_selection_smoke.jsonl" \
IMAGE_IDS=90208,205834 CACHE_BUNDLES=1 SAVE_EXAMPLES=6 \
bash "${PROJECT_ROOT}/scripts/run_pgot_natural_object_selection.sh"
LD_LIBRARY_PATH=/home/jovyan/.conda/envs/scale_rae/lib \
PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}" PYTHONNOUSERSITE=1 \
/home/jovyan/.conda/envs/scale_rae/bin/python - "${SMOKE_ROOT}" <<'PY'
import json
import math
import pathlib
import sys
import numpy as np
import torch

root=pathlib.Path(sys.argv[1])
summary=json.loads((root/'summary.json').read_text())
manifest=json.loads((root/'manifest.json').read_text())
assert len(manifest['samples'])==2 and len(manifest['queries'])==6
assert summary['protocol']['selector_gt_input'] is False
assert summary['protocol']['selector_image_input'] is False
for condition in ('category','description','oracle_id'):
    stats=summary['conditions'][condition]['overall']
    assert stats['num_queries']==6
    assert stats['target_coverage']>0
    for key in ('mean_iou','grounding_success_50','selection_accuracy_present'):
        assert math.isfinite(stats[key]), (condition,key,stats)
    assert summary['invalid_completions'][condition]==0
for sample in manifest['samples']:
    folder=root/'owners'/str(sample['image_id'])
    pred=np.load(folder/'prediction.npz')['prediction']
    assert pred.shape==(512,512)
    bundles=torch.load(folder/'owner_bundles.pt',map_location='cpu',weights_only=True)
    assert set(bundles)=={'semantic_slots','visual_memory','ovt_object_valid','raw_rae_hidden'}
    assert all(torch.isfinite(v).all() for v in bundles.values())
assert len(list((root/'examples').glob('*/selection_overlay.png')))==6
assert (root/'comparison.csv').is_file()
print('Real-GPU smoke passed: two extracted images reused across six requests; both selector conditions + oracle + CSV/overlays.')
PY
# Resume check must reuse the two maps and twelve selector completions.
GPU="${GPU:-0}" SELECTOR_GPUS="${GPU:-0}" OUTPUT_DIR="${SMOKE_ROOT}" \
QUERIES="${PROJECT_ROOT}/configs/natural_object_selection_smoke.jsonl" \
IMAGE_IDS=90208,205834 CACHE_BUNDLES=1 SAVE_EXAMPLES=6 \
bash "${PROJECT_ROOT}/scripts/run_pgot_natural_object_selection.sh"
if [[ "${KEEP_SMOKE:-0}" == 1 ]]; then
    echo "Smoke results retained by KEEP_SMOKE=1: ${SMOKE_ROOT}"
else
    case "${SMOKE_ROOT}" in
        "${PROJECT_ROOT}/.smoke_natural_object_selection."*) find "${SMOKE_ROOT}" -depth -delete ;;
        *) echo "Unsafe cleanup path" >&2; exit 1 ;;
    esac
    echo "Smoke passed; temporary results and owner bundles removed."
fi
