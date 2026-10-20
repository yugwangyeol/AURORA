# Storage cleanup manifest (2026-10-20)

This manifest records storage removed after the ICLR submission snapshot.
The source code, lightweight evaluation summaries, and paper figures were
committed as `ced9752` and tagged as
`iclr-submission-postcheck-2026-10-20` before cleanup.

## Phase 1: approved generated or superseded artifacts

### Removed

- `checkpoint_archive/pgot_instance_e1/checkpoint-7500`
- `checkpoint_archive/pgot_instance_e1/checkpoint-8000`
  - Approximate total: 52 GiB (`du -sh`).
  - Reason: superseded by
    `checkpoints/pgot_instance_e1/checkpoint-10000`.
- Five generated `oracle_checkpoint.pt` files under
  `outputs/object_bottleneck_oracles_pgot_main_v15_1_coda_mmproj/`.
  - Approximate total: 25 GiB (`du -sh`).
  - Conditions: per-run logs, per-run `summary.json`, aggregate
    `summary.json`, and `summary.csv` are retained and versioned.
- Local smoke-test and temporary verification outputs matching:
  - `.smoke_comp_full5000.OZUQYQ/`
  - `.pytest_cache/`
  - top-level `__pycache__/`
  - `outputs/_smoke*`
  - `outputs/_tmp*`

### Explicitly retained

- All current DeOR checkpoints and ablation checkpoints.
- All datasets.
- All lightweight evaluation summaries and final paper figures.
- The complete `wandb/` directory. It contains 9 `offline-run-*`
  directories whose synchronization status was not established.
- The oracle experiment logs and summary files listed above.

## Phase 2: legacy pre-DeOR checkpoints

### Removed

The following legacy experiment families were removed after confirming that
no PGOT training or evaluation process was running and that current
one-shot DeOR checkpoints were outside the selection:

- `checkpoints/pgot_main*`
- `checkpoints/pgot_e*`
- `checkpoints/pgot_core_alllayer`
- `checkpoints/pgot_instance_e1`
- `checkpoints/pgot_lviscap_hardreg_e2`
- `checkpoints/pgot_pix2cap_thing_e3`
- `checkpoints/pgot_pix2cap_thing_e4`

The audited size before removal was 2,183,590,000,000 bytes approximately
(2,033.6 GiB, or 1.986 TiB). These checkpoints belong to architecture and
training prototypes that predate the final one-shot memory DeOR line. Their
lightweight evaluation summaries and source code remain versioned.

### Explicitly retained

- `pgot_oneshot_memory8_register32_noprior` (DeOR-B)
- `pgot_oneshot_memory8_register32_noprior_7b_fp32_fast` (DeOR-L)
- Reader-supervision and training-objective ablations
- Object-memory count ablations
- Description Init/Attn ablations
- Direct-RAE, contrastive, and DINO-value ablations
- RefCOCO, GQA, VQAv2, and fixed-query revision checkpoints

## Later phases (not removed in Phases 1-2)

- Legacy `pgot_main*`, early `pgot_e*`, and prototype checkpoints.
- Duplicate final-model weights at experiment root and `checkpoint-5000/`.
- Optimizer and scheduler state for experiments that will never be resumed.
- Regenerable raw qualitative caches, embeddings, and rejected candidates.

Each later phase requires a fresh path/reference audit before deletion.
