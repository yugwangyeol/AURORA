#!/usr/bin/env bash
# Four real DDP optimizer steps, checkpoint reload, all three evaluation splits,
# and shared-protocol comparison tables. Never writes production checkpoints.
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export NUM_GPUS="${NUM_GPUS:-2}"
SMOKE_ROOT="${SMOKE_ROOT:-$(mktemp -d /tmp/deor-refcoco-ddp.XXXXXX)}"
export OUTPUT_DIR="${SMOKE_ROOT}/train"
export EVAL_ROOT="${SMOKE_ROOT}/eval"
export EVAL_GPU="${EVAL_GPU:-${CUDA_VISIBLE_DEVICES%%,*}}"
export MAX_STEPS=4 SAVE_STEPS=4 EVAL_STEPS=4 EVAL_NUM_IMAGES=4
export MAX_REFS=3 LOGGING_STEPS=1 WARMUP_RATIO=0
export PGOT_PARAMETER_UPDATE_AUDIT=True
export PGOT_CUDA_MEMORY_DIAGNOSTICS=True
export MASTER_PORT="${MASTER_PORT:-29674}"
mkdir -p "${SMOKE_ROOT}"
bash scripts/run_deor_refcoco_direct_finetune.sh > "${SMOKE_ROOT}/pipeline.log" 2>&1
"${PYTHON:-/home/jovyan/.conda/envs/scale_rae/bin/python}" - "${SMOKE_ROOT}" "${NUM_GPUS}" <<'PY'
import json, sys
from pathlib import Path
root, world = Path(sys.argv[1]), int(sys.argv[2])
audits = []
for rank in range(world):
    audit = json.loads((root/'train'/f'parameter_update_audit.rank{rank}.json').read_text())
    assert audit['global_step'] == 4 and audit['loss_finite_all_logged'], audit
    for group in ('qwen_lora', 'mm_projector', 'register'):
        stats = audit['groups'][group]
        assert stats['gradient_finite_all_sampled'] and stats['update_nonzero_any_sampled'], stats
    assert not set(audit['groups']) & {'writer','reader','dit_adaln','dit_last16_body','rae_query'}
    audits.append(audit)
config = json.loads((root/'train/checkpoint-4/config.json').read_text())
assert config['pgot_refcoco_direct_finetune']
assert config['pgot_e11_object_memories_per_owner'] == 8
assert config['pgot_e11_register_memories_per_owner'] == 32
assert not config['pgot_one_shot_direct_rae_query']
assert not config['pgot_one_shot_writer_owner_prior']
summary = json.loads((root/'eval/summary.json').read_text())
assert set(summary['by_split']) == {'val','testA','testB'}
assert (root/'eval/comparison_full/table.md').is_file()
assert (root/'eval/comparison_nonoverlap/table.md').is_file()
report = dict(status='passed', ranks=world, optimizer_steps=4,
              trainable_groups=audits[0]['groups'],
              rank_steps=[a['steps'] for a in audits],
              evaluation=summary['by_split'])
(root/'verification.json').write_text(json.dumps(report,indent=2)+'\n')
print(f'DDP + save + reload + direct evaluation + comparison PASSED: {root}')
PY
