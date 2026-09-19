# RefCOCO: frozen DeOR vs. CTRL-O with the same original expression

This is **referring expression segmentation**, not Binding Hit or same-generated-AR-text
Class–Mask Pair evaluation. The human expression already identifies a particular
annotation. Only that target is scored; no class matching or Hungarian repair.

## Default protocol

- RefCOCO, official UNC split, all sentences in val / testA / testB.
  Locally: 10,834 / 5,657 / 5,095 expressions, respectively (21,586 total).
- Original `raw` text, including capitalization and punctuation, goes unchanged
  to CTRL-O's official LLM2Vec encoder and into DeOR's fixed structural caption:
  `<thing> Object 1: {original expression} <ovt>. <scene_end>`.
  The generic `Object 1` header does **not** reveal a GT class. This is external
  expression conditioning, **not** image-only AR discovery or AR caption replay.
- DeOR: original object-memory 8 / register-memory 32 / noprior checkpoint-5000,
  one valid OVT; queried owner competes with summed native void-owner probabilities.
  No other object descriptions, GT classes/counts, points, boxes, or masks supplied.
- CTRL-O: official released `pretrained_model.ckpt`, strict loading; one expression
  bound to input slot 0, six unbound learned slots. Their language projections are
  masked. Argmax across **all seven** slots, then select slot 0. No GT-best slot.
- One expression per forward pass for both models. Every sentence is retained;
  empty predictions get IoU=0 rather than being excluded.
- Entire frame resized to 512x512, no center crop, flip, or spatial prompt rewrite.
  Model-specific processors afterward (CTRL-O 224; DeOR checkpoint processors).
  Native ownership probabilities are bilinearly interpolated to original image
  H,W before argmax. GT is decoded at original resolution, without crop/ignore repair.
- Expression-weighted mIoU; oIoU = summed intersection / summed union; P@0.5,
  P@0.6, P@0.7, P@0.8, P@0.9 = fractions of expressions meeting the IoU threshold.
  CSV/JSON values are fractions in [0,1], multiply by 100 for paper tables.
- Annotation masks are decoded only in the final scoring stage. Prediction receives
  an explicit allowlist of image + expression inputs. All source/checkpoint hashes
  and per-expression prediction provenance are saved. Incomplete full splits cannot
  be scored as completed; rerunning the same command resumes completed queries.

## Important limits on claims

These checkpoints receive **no additional RefCOCO training**, but RefCOCO uses COCO
images. All local UNC val/testA/testB images are available in **COCO train2017**.
Since DeOR was COCO-trained, this is NOT an unseen-image zero-shot benchmark; image
and object-mask overlap with its COCO training is possible. Report existing training
supervision explicitly. CTRL-O uses the released VG+COCO checkpoint (global step 71000),
not the VG-only checkpoints underlying the paper's Table 3. Do not copy the published
Table 3 numbers into the same row as if the checkpoint/protocol were identical.
The direct DeOR caption format is an inference adaptation, whose quality must be
measured rather than assumed. No prompt search on testA/testB is permitted.

Official dataset/API: <https://github.com/lichengunc/refer>.
CTRL-O Table 3: <https://arxiv.org/html/2503.21747v1>.

## Run inside tmux

The annotations are already under `/home/jovyan/data/coco/refcoco`. The script maps
COCO train2014 image IDs to existing train2017/val2017 filenames. No download required.
Current available GPU is CUDA index 0, B200 MIG 45GB. Stages run sequentially in the
existing PGOT and CTRL-O environments so the models do not coexist in GPU memory.

```bash
cd /home/jovyan/PGOT
GPU=0 bash scripts/eval_deor_ctrlo_refcoco.sh
```

Default: all three splits. Full results:
`outputs/eval_deor_ctrlo_refcoco_direct/{summary.json,comparison.csv,per_expression.jsonl}`.
Log: `outputs/eval_deor_ctrlo_refcoco_direct/run.log`.

Small execution smoke, first 3 references by ID in val, **all their sentences**:

```bash
GPU=0 SMOKE=1 bash scripts/eval_deor_ctrlo_refcoco.sh
```

Smoke has a separate output directory, never contaminates full results.
To evaluate only val, use another output directory (do not mix configurations):

```bash
GPU=0 SPLITS=val OUTPUT_ROOT=outputs/eval_deor_ctrlo_refcoco_val_direct \
  bash scripts/eval_deor_ctrlo_refcoco.sh
```

Optional separately labeled benchmarks:

```bash
GPU=0 REF_DATASET=refcoco+ bash scripts/eval_deor_ctrlo_refcoco.sh
GPU=0 REF_DATASET=refcocog bash scripts/eval_deor_ctrlo_refcoco.sh
```

RefCOCOg defaults to UMD val/test. These are separate from the paper's Gref split;
never compare across split conventions without alignment. `REF_DATA_ROOT`,
`REF_IMAGE_ROOT`, `MODEL_PATH`, `CTRLO_CHECKPOINT_DIR`, `SPLIT_BY`, `SPLITS`,
`EMBEDDING_BATCH_SIZE`, `LLM2VEC_BASE`, `LLM2VEC_ADAPTER` are configurable.
`PREFLIGHT_ONLY=1` validates data/encoder access without GPU inference.

## Paper table

| Method | RefCOCO val mIoU | testA mIoU | testB mIoU |
|---|---:|---:|---:|
| CTRL-O (released VG+COCO checkpoint) | — | — | — |
| DeOR (COCO-trained; external expression conditioning) | — | — | — |

Caption should disclose supervision, COCO image overlap, no additional RefCOCO
training, original expressions, full-frame preprocessing, original-resolution
mask IoU, no coordinates, and fixed input-associated mask readout.
Report oIoU and P@0.5 in an additional table or appendix.

## Validation (2026-09-18)

- 16 new CPU unit tests plus 24 existing class/mask/binding regression tests passed.
- Actual GPU 0 (45GB) full-text smoke completed: first 3 val references by ID,
  11 original expressions / 2 images, all 22 model predictions at native resolution.
  Verified original raw-text/target identity, official encoder, memory8/register32/noprior,
  strict CTRL-O loading, per-query provenance, and JSON/CSV output.
- Entire default dataset passed prepare/preflight: 21,586 expressions / 3,000 images,
  no missing images. This created the full manifest, but **full GPU evaluation was not run**.
- Smoke output: `outputs/smoke_deor_ctrlo_refcoco_direct/summary.json`.
  The fixed tiny subset is for execution checks, not representative performance.
