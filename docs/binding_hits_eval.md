# PGOT COCO Binding Hits (annotation-only queries)

```bash
cd /home/jovyan/PGOT
GPU=0 bash scripts/eval_pgot_memory8_binding_hits.sh
```

Default checkpoint: `checkpoints/pgot_oneshot_memory8_register32_noprior/checkpoint-5000`:
8 memories/object, 32 memories/background register, 4 semantic registers, no
writer owner prior. Default batch size 1, FP32, workers 2, seed 0, metric grid 512.
No diffusion sampling, reconstruction decoder, AR generation, or extra training.
These defaults describe the original `BINDING_ROUTE=direct` path. The separate
[AR + class/point router evaluator](ar_binding_router_eval.md) uses
`scripts/eval_pgot_memory8_binding_hits_ar_router.sh` and distinct output folders;
it does not replace or overwrite the direct query-conditioned protocol.

The source is official COCO val2017 (all 5,000 image IDs),
`/home/jovyan/data/coco/annotations/instances_val2017.json` and `val2017` images.
**No Pix2Cap captions/data or cache-label/category guessing is used.**

This is **query-conditioned grounding, not GT-free AR segmentation**. Input
queries give GT class names and locations; GT mask pixels never enter the model,
and registers remain unrestricted. We put the queries into PGOT's caption/OVT
format while preserving the checkpoint's system/user prompts. Each query is
paired with its exact original COCO `annotation_id` before inference, and its
OVT index stays fixed. No mask-based matching is used to decide the query target.

## Query protocol

- Same PIL resize-min-side/center crop for RGB and binary GT masks.
- GT masks are independent binary instances, not a resolved integer mask cache.
  Retain all noncrowd, nonignored, positive-area foreground instances visible
  after crop; do not merge instances of the same category.
- Default `QUERY_MODE=class_point`: class name plus normalized `(x,y)` point
  written as text, x left-to-right, y top-to-bottom. `CENTER_SOURCE=mask` computes
  the center of mass of the visible GT mask in the cropped view. Optional
  `CENTER_SOURCE=bbox` uses the cropped mask bounding-box center.
- Default `MAX_BINDINGS=7`: sample up to seven targets per image without replacement
  using an image-ID-dependent seed. Slot order is also randomized. This is a
  fixed query cap, **not** an exact reproduction of CTRL-O's random binding count
  or preprocessing. Unqueried visible instances stay in the GT matching pool.
- `MAX_BINDINGS=0` queries every visible instance. Default safety capacity is 64;
  raise an error rather than silently truncate if a scene exceeds it. Captions
  also fail rather than truncate if their token count exceeds 4096.
- `QUERY_MODE=class_only` drops the point from model input but keeps identical
  selected targets and slot order at the same seed. Same-class duplicates are
  intrinsically ambiguous in this condition, even though the expected identities
  remain known to the scorer. Instance ordinals are just labels, not evidence.
- `QUERY_MODE=class_point_short` preserves the same point and uses the shorter
  description `a person at normalized coordinates (x=0.2500, y=0.6000)`.
- `QUERY_MODE=class_point_natural` adds a natural-language location derived only
  from that same point: `a person in the middle-left part of the image, at
  normalized coordinates (x=0.2500, y=0.6000)`. Horizontal and vertical bins are
  fixed thirds; x increases to the right, y downwards. Boundaries belong to the
  next bin. This adds no color, size, relationship, or other GT attribute.
  Both new modes retain the legacy four-decimal coordinates, query sampling,
  random order, special-token wrapper, mask readout, and metric.
- Overlaps are retained by default, as in CTRL-O's binding metric configuration.
  `IGNORE_OVERLAPS=1` excludes overlap pixels and filters any newly empty targets
  before fixing queries. This changes GT counts and constitutes a different protocol.
- GT-empty scenes are recorded, but have no binding queries and contribute no
  observations to the BH denominator. In all-empty batches model inference is
  omitted, since no binding prediction is being evaluated.

Because this uses independent masks and optionally retains overlaps, GT counts
can differ from the resolved overlap-excluded cache used for earlier segmentation
count buckets. Report the count/protocol saved by this evaluator, not old buckets.

## Metric

PGOT's existing `ovt_owner` readout bilinearly upsamples object ownership and the
sum of all register probabilities, then takes argmax. Keep one integer prediction
label per query plus aggregate background. Compare all these predicted regions
to **all visible foreground GT instances** (including distractors) using maximum
total-IoU Hungarian assignment. A hit occurs if a query's predicted region is
assigned to its **pre-specified** target. Unmatched queries are misses, not skipped.
The aggregate background can match a GT instance, but never counts as a query hit.

Primary `binding_hits = total hits / total queries` (0..1);
`binding_hits_percent` is its percentage. This is not IoU@0.5.
Zero-IoU assignment ties can yield hits; retain official-style unthresholded
scoring but expose `zero_iou_hits`, `positive_iou_binding_hits`,
`binding_hits_iou50` (hit AND target IoU >= .5), and directly paired `target_mIoU`
so degenerate correspondence does not look like good mask quality.
`binding_hits_image_mean` is separately reported; it is not the primary average.

Sources: [CTRL-O paper, section 4.1 and Appendix F](https://arxiv.org/html/2503.21747v1),
[official BindingHits implementation](https://github.com/dido1998/CTRL-O/blob/master/ocl/metrics/masks.py#L134),
[official query preprocessing](https://github.com/dido1998/CTRL-O/blob/master/ocl/preprocessing.py#L40).
The released implementation uses Hungarian assignment regardless of its YAML
`matching` option; we use explicit predicted-row identities, including rectangular
assignments, rather than assuming the returned assignment rows are contiguous.

**This is CTRL-O-style scoring, not bit-exact reproduction of its 61.3% result.**
CTRL-O uses a learned point embedding, not coordinate text; crop resolution,
selected queries, prediction masks/background handling, and model supervision
also differ. A paired system comparison requires re-running CTRL-O on the exported
manifest with compatible crop and query identities; that adapter is not included.

## Outputs and variants

```
outputs/binding_hits_pgot_memory8_register32_noprior/class_point/
  summary.json          # overall, quality-aware metrics and GT-count buckets
  per_image.jsonl       # per-query target/matched annotation IDs, IoUs and hits
  query_manifest.jsonl  # exact class/point/query identities for paired reuse
```

Artifacts are flushed every batch. An existing output directory containing
evaluation artifacts is refused unless you explicitly pass `--overwrite`.
Use distinct output directories for different query caps, seeds, or center modes.
New summaries include `query_identity_sha256`, a text-independent fingerprint of
image IDs, all visible GT identities/points, and ordered selected targets. Equal
fingerprints across modes verify that the paired query inputs differ only in
text (also check matching/resolution/overlap settings). Old summaries lack it;
their exported manifests still contain the same identity fields.

```bash
# Run both new prompts sequentially on one GPU, each on all 5,000 images.
GPU=0 bash scripts/eval_pgot_memory8_binding_hits_prompt_variants.sh

# Or run just one mode.
GPU=0 QUERY_MODE=class_point_short bash scripts/eval_pgot_memory8_binding_hits.sh
GPU=0 QUERY_MODE=class_point_natural bash scripts/eval_pgot_memory8_binding_hits.sh

# Smoke both prompts on image 139 and GT-empty image 25593.
GPU=0 SMOKE=1 NUM_WORKERS=0 bash scripts/eval_pgot_memory8_binding_hits_prompt_variants.sh
```

The two-mode launcher stores separate `class_point_short/` and
`class_point_natural/` directories below the existing output root. Override with
`OUTPUT_ROOT`; do not set single-mode `OUTPUT_DIR` or `QUERY_MODE` on that launcher.
It performs no full AR generation, retraining, spatial prior, or GT-based output
reordering. Prompt variants are exploratory comparisons; choose a prompt on
development data and freeze it for a held-out final comparison, rather than
presenting the best full-val prompt score as an untuned benchmark result.

```bash
# No point in the model input; identical target sample and slot order.
GPU=0 QUERY_MODE=class_only bash scripts/eval_pgot_memory8_binding_hits.sh

# Query ALL visible instances (not capped at 7).
GPU=0 MAX_BINDINGS=0 \
 OUTPUT_DIR=/home/jovyan/PGOT/outputs/binding_hits_pgot_memory8_register32_noprior/class_point_all \
 bash scripts/eval_pgot_memory8_binding_hits.sh

# Separate GPU smoke: ordinary image 139 plus real GT-empty image 25593.
GPU=0 SMOKE=1 NUM_WORKERS=0 bash scripts/eval_pgot_memory8_binding_hits.sh

# CPU regression tests.
PYTHONNOUSERSITE=1 OMP_NUM_THREADS=3 \
 /home/jovyan/.conda/envs/scale_rae/bin/python -m unittest discover \
 -s tests -p test_binding_hits.py -v
```

Seven tests cover perfect/swapped identity, distractors, empty/unmatched predictions,
rectangular matching, overlap ignore, invalid identities, query-weighted averaging,
empty scenes, same-category annotation pairs, stable selection across query modes,
collation, full-split validation, all-target mode and overflow rejection.
They also cover unchanged legacy query text, exact target/coordinate/order
preservation across prompts, and the fixed natural-position grid/boundaries.

Verified 2026-09-17: all five CPU tests passed and FP32 CUDA smoke completed on
the memory8/register32/noprior checkpoint, with seven queries on image 139 and
zero queries on GT-empty image 25593. Both image records and their manifest
identities were validated. Actual-tokenizer CPU preflight also checked all-target
mode on image 435081 (62 instances, 2,950 caption tokens). The largest uncropped
noncrowd annotation count in the official split is 62, within capacity 64.
Original smoke output is in `outputs/_smoke_binding_hits_pgot/class_point`.
The user has since completed the original class-point and class-only full-split
evaluations; these artifacts are preserved. New prompt variants require their
own full-split runs and do not overwrite the original results.

Prompt-variant verification on 2026-09-17: all seven CPU tests passed. Both
`class_point_short` and `class_point_natural` completed FP32 CUDA smoke on images
139 and 25593 (seven queries plus a zero-query scene). Their text-independent
fingerprints match, and exported GT identities, coordinates, and query order
also match the original class-point smoke and the corresponding entries from
the original full-split manifest. Checkpoint metadata confirms object memory 8,
register memory 32 x 4, and writer owner prior disabled. New prompt full-split
evaluations have not been launched by this verification.
