# RefCOCO: language access to cached DeOR object representations

This is **not** `eval_deor_ctrlo_refcoco.sh` (which directly conditions DeOR
on an external expression). This experiment preserves the editing application's
lookup mechanism: image-only discovery, then text-only ID selection, then reuse
of the selected object's immutable region. No training or diffusion generation.

## Protocol

- Frozen original 1.5B checkpoint-5000: object memory8, register memory32×4,
  RAE queries through MLLM, SigLIP values, no adapter/contrastive loss.
- RefCOCO UNC val by default; original `raw` expression unchanged, all sentences.
  RefCOCO/testA/testB and separate RefCOCO+/RefCOCOg also supported.
- DeOR generates its native structured AR caption **once per distinct image**
  using the image only. A second own-caption forward extracts ownership and
  representations. This image-only pair is reused for every expression.
  No annotation class/count/caption/mask or user expression enters DeOR.
- External pinned Qwen3-VL-4B-Instruct is **text-only**, not a visual selector.
  Conditions: generated categories+opaque IDs; add complete generated
  descriptions. Same candidate IDs/order, system, greedy decoding and selector.
  IDs/order shuffled deterministically. No geometry, images, slot ordinals,
  annotation IDs, GT labels or answer IDs reach the selector.
- Null/invalid/unbound/empty selections are empty masks; no retries, guessed
  IDs or best-mask correction. Missing discoveries remain failures.
- Entire image resized to512×512 without cropping. All valid native owners
  and summed void probability compete; probability maps interpolated to ORIGINAL
  H,W **before** argmax. Masks are immutable after extraction.
- CTRL-O uses the SAME original expression, official LLM2Vec encoder and
  released checkpoint. Slot0 bound, six unbound slots; argmax over all seven,
  slot0 only. Separate processes prevent GPU model coexistence.
- Primary scores compare the actually returned mask to the exact referred
  annotation, irrespective of generated class or diagnostic target coverage.
  mIoU is expression-weighted; oIoU aggregates intersection/union pixels;
  Success@0.5/@0.75 count all expressions, including failures.
- Secondary coverage/selection accuracy uses fixed, class-compatible one-to-one
  max-cardinality correspondence at IoU>=0.10, established independently of
  expression/selector. `oracle_id` is this GT-only diagnostic, NOT a deployable
  method or guaranteed global upper bound. CTRL-O selection accuracy is N/A
  (its expression-bound slot is not a candidate ID).
- Source/config/checkpoint fingerprints reject incompatible caches. Outputs
  completed per image/expression resume; incomplete full splits cannot be scored
  as completed. AR token budget2048; incomplete generations use native whole-OVT
  packing, remain evaluated, and are counted in `ar_incomplete_images`. Existing
  complete descriptions are never shortened for selector prompts.

## Run on GPU1

```bash
cd /home/jovyan/PGOT
GPU=1 bash scripts/run_refcoco_object_selection.sh
```

Stages: prepare → encoder preflight → DeOR image-only extraction → category
selector → description selector → official LLM2Vec → CTRL-O → scoring.
Default evaluates **RefCOCO val**, not all three splits. Default result root:
`outputs/refcoco_object_selection_memory8_register32/`.

All three official evaluation splits in a separate root:

```bash
GPU=1 SPLITS="val testA testB" \
OUTPUT_DIR=/home/jovyan/PGOT/outputs/refcoco_object_selection_memory8_register32_all \
bash scripts/run_refcoco_object_selection.sh
```

Identical rerun resumes. `WITH_CTRLO=0` skips only CTRL-O and its encoder, using
a **fresh output root**; both DeOR selector conditions still run. `MAX_REFS=20`
selects a fixed ref-ID prefix per split for a pilot (every sentence retained).
`MAX_SAMPLES` alternatively limits expressions per split. Never choose subsets
using prediction quality. `PREFLIGHT_ONLY=1` validates full data/encoder access
without GPU inference. `CACHE_BUNDLES=1` stores representations for reuse.

Real GPU smoke:

```bash
GPU=1 bash scripts/smoke_refcoco_object_selection.sh
```

First3 val references byID, **2images/11expressions**, both real selectors,
real CTRL-O/encoder, native masks and finite optional object bundles. Then
reruns to verify model-free cache reuse. Temporary outputs deleted on success;
failures preserved; `KEEP_SMOKE=1` retains successful outputs. Smoke scores
are execution checks, not representative benchmark results.

## Outputs / limits

- `results.md`: split/method table in **percent**.
- `summary.json`, `comparison.csv`: rates in **[0,1]**, denominators and protocol.
- `per_query.jsonl`: exact target/expression, actual selection and mask scores.
- `owners/<image_id>/`: AR text/provenance and original-resolution labels.
- `selections/{category,description}/`: raw responses and selector provenance.
- `baseline/`: official encoder embeddings and CTRL-O predictions/provenance.
- `examples/`: selected-mask overlays with query metadata; `logs/`: every stage.

Results measure **DeOR + external text selector**, not DeOR alone. They show
description-addressability and localization; visual-memory usefulness requires
the separate editing experiment. No additional RefCOCO training does NOT imply
unseen images: train-image overlap is audited against the actual DeOR manifest.
COCO training/image/object-mask supervision differ across methods. The released
CTRL-O VG+COCO checkpoint is NOT the VG-only model in paper Table3; published
numbers must not be mixed as a matching reproduction.

## Validation (2026-09-18)

- 49 CPU unit/regression tests passed, including actual shell stage ordering,
  GPU1 isolation, selector-package isolation, fail-fast behavior, strict source
  resume checks, native ownership competition and no class/coverage gating of
  primary IoU.
- Real GPU1 smoke passed prediction and scoring for 2 images / 11 expressions:
  image-only DeOR AR/ownership, both real selector conditions, official LLM2Vec,
  released CTRL-O, original-resolution masks, finite cached object bundles and
  all overlays/CSV/JSON outputs. Identical rerun reused every completed cache
  without reloading any model weights; temporary smoke results were removed.
- Full default val preparation/encoder preflight passed: 1,500 images / 10,834
  expressions / 8,064 distinct texts, 191 images overlapping the specified DeOR
  training manifest. **Full GPU evaluation has not been run.** This overlap
  audit does not establish that other images were unseen by bootstrap/pretraining.

Execution smoke only (first3 references; not representative benchmark results):

| Input/method | mIoU (%) | oIoU (%) | Success@0.5 (%) |
| --- | ---: | ---: | ---: |
| DeOR + selector, categories only | 13.27 | 5.19 | 18.18 |
| DeOR + selector, generated descriptions | 38.26 | 34.19 | 54.55 |
| CTRL-O, original expression | 33.05 | 29.04 | 18.18 |

No benchmark advantage should be inferred from these smoke numbers.
