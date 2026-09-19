#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SMOKE_ROOT="${SMOKE_OUTPUT_DIR:-$(mktemp -d "${PROJECT_ROOT}/.smoke_refcoco_object_selection.XXXXXX")}"
trap 'echo "Smoke failed; outputs preserved at ${SMOKE_ROOT}" >&2' ERR
run_smoke() {
    GPU="${GPU:-1}" OUTPUT_DIR="${SMOKE_ROOT}" SPLITS=val MAX_REFS=3 MAX_SAMPLES=0 \
    WITH_CTRLO=1 CACHE_BUNDLES=1 SAVE_EXAMPLES=11 \
    bash "${PROJECT_ROOT}/scripts/run_refcoco_object_selection.sh"
}
run_smoke
PYTHONNOUSERSITE=1 PYTHONPATH="${PROJECT_ROOT}" \
LD_LIBRARY_PATH=/home/jovyan/.conda/envs/scale_rae/lib \
/home/jovyan/.conda/envs/scale_rae/bin/python - "${SMOKE_ROOT}" <<'PY'
import json, math, pathlib, sys
import numpy as np
import torch
root=pathlib.Path(sys.argv[1])
m=json.loads((root/'manifest.json').read_text())
s=json.loads((root/'summary.json').read_text())
assert len(m['queries'])==11 and len(m['samples'])==2
assert s['protocol']['selector_image_input'] is False
assert s['protocol']['selector_gt_input'] is False
assert s['protocol']['full_split_evaluated'] is False
assert s['num_expressions']==11
for method in ('category','description','oracle_id','ctrlo'):
    stats=s['by_split']['val'][method]
    assert stats['num_expressions']==11
    assert all(math.isfinite(stats[k]) and 0<=stats[k]<=1 for k in ('mIoU','oIoU','success_50'))
for sample in m['samples']:
    folder=root/'owners'/str(sample['image_id'])
    pred=np.load(folder/'prediction.npz')['prediction']
    assert pred.shape==(sample['height'],sample['width'])
    assert sample['ovt_count']>0
    bundles=torch.load(folder/'owner_bundles.pt',map_location='cpu',weights_only=True)
    assert all(torch.isfinite(v).all() for v in bundles.values())
for condition in ('category','description'):
    paths=list((root/'selections'/condition/'queries').rglob('*.json'))
    assert len(paths)==11
    # Invalid/abstained completions are valid experimental failures, not repaired.
assert len(list((root/'examples').glob('*/*/*/selection_overlay.png')))==11
assert (root/'comparison.csv').is_file()
print('Actual GPU smoke passed: 2 image-only AR extractions, 11 original expressions, both real selectors, real CTRL-O, original-resolution scoring.')
PY
run_smoke # Resume must avoid reloading ALL model weights.
if [[ "${KEEP_SMOKE:-0}" == 1 ]]; then
    echo "Smoke retained at ${SMOKE_ROOT}"
else
    case "${SMOKE_ROOT}" in
        "${PROJECT_ROOT}/.smoke_refcoco_object_selection."*) find "${SMOKE_ROOT}" -depth -delete ;;
        *) echo 'Custom smoke output retained; no automatic deletion of arbitrary paths.' ;;
    esac
fi
