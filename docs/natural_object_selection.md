# Natural-language selection of DeOR objects and cached regions

Frozen original **1.5B Object 8 / Register 32** checkpoint:
`checkpoints/pgot_oneshot_memory8_register32_noprior/checkpoint-5000`.
This is separate from the running w/o-grounding ablation and the newer 7B run.
No weights are trained. DeOR extracts each image's ownership map once from
its original image-only AR caption; subsequent requests retrieve cached regions.
The user request is not fed back into DeOR.

## Protocol and claims

The same pinned Qwen3-VL-4B-Instruct checkpoint acts as an external **text-only**
selector. Its visual encoder receives no images. Compare:

1. Generated category names and opaque object IDs only.
2. The same category names/IDs plus the complete generated object descriptions.
3. Fixed GT-correspondence IDs as an oracle diagnostic, not a deployable selector.

System instruction, candidate order, query, model and greedy decoding are equal
in conditions1/2. Descriptions are never shortened. IDs/order are deterministically
shuffled to avoid exposing caption ordinals/size ordering. Neither predicted
coordinates, masks, GT instance IDs nor GT labels are passed to the selector.
The request necessarily contains the user's referring information, which is
common to both conditions. Invalid JSON/IDs and null responses are retained,
never repaired by choosing the first or best-overlapping object.

**Default requests are independent Pix2Cap annotation descriptions**, not DeOR
outputs reused as ground-truth requests. For every visible thing segment in
`data/pgot_pix2cap_generated_val5k.jsonl`, request:
`Find the <category> described as: <annotation description>`.
These are annotation-description retrieval queries, **not RefCOCO**, and the
provided manifest is not the official full5000-image COCO split. Its actual
image/query counts are recorded. Annotation text itself can be inaccurate;
this is not a human-verified referring-expression benchmark.

Duplicate descriptions for same-category GT objects make requests ambiguous.
They remain in overall scores and have a separately reported subgroup. Do not
present this subgroup's single-target accuracy as uniquely solvable language
understanding. Type groups (color, position, relation, occlusion) are overlapping
lexical diagnostics, not manually labeled tasks. Same-category multiplicity is
defined by visible GT instances, independently of generated object count.

Both images and Pix2Cap segment-ID PNGs use the existing PIL min-side resize
and512 center crop, nearest for masks. Fully cropped-out targets are counted
as exclusions before making requests. Custom requests for invisible targets
raise an error rather than being silently excluded. GT masks are used only
for scoring; no GT class/count/mask/caption is supplied to DeOR extraction.
Only the **generated AR caption** is replayed, with all four PGOT token IDs
registered for correct own-caption conditioning.

Before evaluating requests, establish a **class-compatible, one-to-one,
maximum-cardinality GT/predicted-region correspondence** using IoU>=0.10,
breaking ties by total IoU. This never uses request text or selector output.
The presence threshold is configurable and should be disclosed; it is an
operational correspondence definition, not perfect-object coverage.

Scores:

- Target coverage: fraction of requests whose GT target has a fixed matched ID.
- Selection accuracy on present targets, and all-target accuracy separately.
- Mean IoU and region-search Success@0.50/@0.75 over **all requests**.
  Missing targets and null/invalid selections contribute zero.
  `raw_selected_mask_iou` is also saved without the missing-target override.
- Same-category multi-instance, ambiguous/unambiguous and query-type subgroups.

Only the actual selected ID's hard region is scored. There is no post-selection
Hungarian repair or query-conditioned choice of the best mask. The oracle row
diagnoses this fixed mapping, not an absolute upper bound across all possible
region matches. Empty/malformed/unbound generated descriptions are retained;
unbound trailing text has an empty region. Predicted masks are not modified.

This experiment supports description-addressable object/region reuse. It does
**not** show visual-memory benefits in QA, editing quality, or that DeOR itself
interprets arbitrary user requests. Instruction interpretation belongs to the
external selector. No performance advantage is assumed before measurement.

## Run

Single available GPU, all stages sequential:

```bash
cd /home/jovyan/PGOT
GPU=0 bash scripts/run_pgot_natural_object_selection.sh
```

Use0/1 for the two selector conditions in parallel **after** extraction on0:

```bash
cd /home/jovyan/PGOT
GPU=0 SELECTOR_GPUS=0,1 bash scripts/run_pgot_natural_object_selection.sh
```

DeOR runs in `scale_rae` (FP32), selector in `metaquery` (BF16 inference). Each
has its own library path. The selector alone explicitly prepends
`/home/jovyan/.local/lib/python3.11/site-packages` (Transformers 5.3.0), because
metaquery's environment-local Transformers 4.49.0 does not support Qwen3-VL.
This does not install/upgrade packages or affect the native DeOR/training stages.
Override `SELECTOR_SITE_PACKAGES` for another compatible selector environment.
No model download/login is needed for the cached pinned
selector snapshot. GPU2/3 are not used. Full evaluation is NOT started by smoke.

Small independent annotation-query pilot, use a separate output root:

```bash
GPU=0 MAX_IMAGES=50 \
OUTPUT_DIR=/home/jovyan/PGOT/outputs/natural_object_selection_pilot50 \
bash scripts/run_pgot_natural_object_selection.sh
```

Custom requests JSONL schema:

```json
{"image_id":90208,"target_segment_id":4,"request":"Find the green bus.","query_types":["color"],"ambiguous":false}
```

`target_segment_id` is from the **Pix2Cap segment mask**, not a COCO annotation
ID or a predicted object ID. Supply `QUERIES=/absolute/path/queries.jsonl` and
a fresh `OUTPUT_DIR`; only queried images are extracted. `IMAGE_IDS=90208,205834`
and `MAX_IMAGES` select a fixed subset, never by GT prediction quality.
`CACHE_BUNDLES=1` additionally stores semantic slots, visual memories, validity
and raw RAE queries for future editing; it is off by default to limit disk use.

## Smoke and output

```bash
GPU=0 bash scripts/smoke_pgot_natural_object_selection.sh
```

Real frozen DeOR and real selector on0, two curated images/six authored requests:
green/red bus, left relation, partially visible bus, dog and bowl. The same two
maps serve all six requests. Both selector conditions, oracle scoring, CSV,
overlays and optional owner bundles are checked. Re-running must reuse completed
caches without loading DeOR/selector weights. Temporary results are deleted
on success; failures are preserved for debugging. `KEEP_SMOKE=1` retains results.
Curated smoke is runtime validation, **not representative performance**.

Validation on 2026-09-18: all 14 focused CPU tests and the real GPU0 pipeline
passed (two images, six requests, both selectors, oracle, overlays, finite
owner bundles). Category-only selected 4/6 correctly; generated descriptions
selected 6/6, with mean IoU 0.9392. These authored requests on curated images
are a smoke demonstration only, not evidence of a benchmark advantage.

Default output root: `outputs/natural_object_selection_memory8_register32_noprior/`.

- `results.md`: overall three-row table.
- `comparison.csv`: all conditions/subgroups.
- `summary.json`: metrics, denominators, protocol and selector provenance.
- `per_query.jsonl`: request, target, selected/expected ID, actual IoU, status.
- `examples/<query_sha256>/`: original, three-condition overlay and full metadata.
- `owners/<image_id>/`: image and immutable native predicted regions.
- `manifest.json`, `deor_model_info.json`, `selections/*/selector_info.json`.
- `logs/`: prepare/extract/category/description/score logs.

Each completion is saved atomically. Identical reruns resume completed stages;
configuration/caption/source/checkpoint mismatches fail instead of mixing results.
Checkpoint signatures use config/index SHA256 and shard paths/sizes/mtimes,
not full multi-GB weight-file hashes. Changing any model/request/protocol needs
a fresh output root. Selector conditions must use identical model/settings.
Training-image overlap is checked against the actual DeOR training manifest;
this does not audit the external selector's pretraining contamination.
