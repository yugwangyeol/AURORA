# PGOT memory8/register32: full COCO/VOC AR segmentation evaluation

```bash
cd /home/jovyan/PGOT
GPU=0 bash scripts/eval_pgot_memory8_full_coco_voc_ar.sh
```

Both stages use the existing checkpoint
`checkpoints/pgot_oneshot_memory8_register32_noprior/checkpoint-5000`.
Override it with `MODEL_PATH`. Default output root is
`outputs/segeval_pgot_memory8_full_coco_voc_ar`, preserving earlier results.
Individual stages: append `coco` or `voc`.

- COCO: **all 5,000 official val2017 image IDs**, including GT-empty images.
  No dependency on `pgot_pix2cap_generated_val5k.jsonl` (actually 4,720 images),
  Pix2Cap captions, object descriptions, or instance manifests.
  Validate IDs against `instances_val2017.json`; fail on missing images/cache
  entries instead of silently dropping images or using a fallback GT.
- VOC: **all 1,449 VOC2012 segmentation val images** in `sets/val.txt`.
  No filtering by GT object count. VOC val is not a 5,000-image split; do not
  duplicate images or add train images to reach 5,000.
- AR: image + original checkpoint prompts/schema seed only. Generate object
  descriptions/OVTs until scene-end as before; do not use GT descriptions,
  classes, object counts, GT masks, or register routing. GT-empty inputs may
  still produce objects; never replace predictions with an empty scene using GT.
  COCO's collator placeholder is always scene-end-only regardless of GT count,
  and is completely replaced by AR-generated captions before the model forward.
- Readout: existing `ovt_owner`, register routing `unrestricted`, FP32,
  512 metric grid, no diffusion/rFID/KID. All-invalid generated OVT masks can
  still read out register background; do not skip images with zero generated OVTs.
- GT: same COCO CODA-style instance/thing-category caches and separate overlaps
  used by the SPOT/SlotDiffusion COCO5000 evaluation. Preserve cache labels.
  VOC uses the existing 512 center crop and union of 255-ignore pixels.
- Counts: visible GT instance IDs after crop/overlap exclusion (not max ID,
  generated OVT count, or number of caption segments). COCO count buckets now
  match the SPOT/SlotDiffusion full-set buckets, unlike manifest-based AR counts.
- Default batch: COCO 4; VOC 1, matching the earlier evaluation scripts.
  Override both with `BATCH_SIZE`. Workers 2, OMP/MKL threads 3.
  Preserve default `AR_MAX_NEW_TOKENS=512` for comparable metrics.

Each stage writes `summary.json`, `per_image.jsonl`, AR caption records, and a log:

```
outputs/segeval_pgot_memory8_full_coco_voc_ar/
  coco_ar/summary.json
  coco_ar/per_image.jsonl
  voc_ar/summary.json
  voc_ar/per_image.jsonl
```

Summaries include total/evaluated image counts, `full_split_evaluated`,
`zero_gt_object_samples`, overall metric valid counts, bucket `0`, and exact
object-count results. Per-image rows include metrics, GT count, predicted segment
count, generated-object count, and whether scene-end was actually generated.

Objectless-image metric semantics are unchanged from `pgot_metrics`:
FG-ARI degenerates to 1 with no foreground; mBO has no foreground target and is
NaN (excluded from its mean, reported with `mBO_valid_n=0` in bucket `0`);
PGOT's mIoU includes background and remains defined. A total of 5,000 evaluated
images does **not** mean every metric necessarily has 5,000 defined values.

## Smoke and tests

```bash
GPU=0 SMOKE=1 BATCH_SIZE=1 NUM_WORKERS=0 \
 OUTPUT_ROOT=/home/jovyan/PGOT/outputs/_smoke_pgot_full_coco_voc_ar \
 bash scripts/eval_pgot_memory8_full_coco_voc_ar.sh
```

COCO smoke selects one ordinary image and a real GT-empty image from the official
cache (not just the first N images); VOC smoke selects the first two val images.
`SMOKE=1` defaults to the separate `_smoke_pgot_full_coco_voc_ar` output root;
an explicit `OUTPUT_ROOT` overrides it. Do not point smoke at full results.
`SMOKE=1` and `MAX_SAMPLES` are mutually exclusive. Full runs use neither.
`--sample_indices` is for COCO/VOC/MOVi smoke/debugging only, not legacy JSONL
evaluation, and cannot be combined with `--max_samples`.

CPU regression tests:

```bash
PYTHONNOUSERSITE=1 OMP_NUM_THREADS=3 \
 /home/jovyan/.conda/envs/scale_rae/bin/python -m unittest discover \
 -s tests -p test_full_coco_voc_ar_dataset.py -v
```

They check exact 5,000-image coverage, rejection of incomplete COCO caches,
GT-count-independent COCO placeholders, VOC GT-empty retention, batch collation,
NaN-safe zero-count buckets, and register-background readout with no valid OVTs.

Verified 2026-09-16: all four CPU regression tests passed; GPU AR smoke completed
both stages (COCO 2/5,000 selected, VOC 2/1,449 selected). COCO image 25593 had
zero visible GT instances but the model independently generated one object and
scene-end; its prediction was scored, not discarded or GT-forced to background.
The zero bucket retained this image with `n=1`, `mBO_valid_n=0`.
Smoke artifacts are in `outputs/_smoke_pgot_full_coco_voc_ar`.
These are smoke checks only; the full evaluation has not yet been run.
