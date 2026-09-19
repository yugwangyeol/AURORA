# Ours AR / native CTRL-O qualitative region comparison

This is GT-assisted qualitative case mining, not an unbiased benchmark or a
comparison against the paper's reported Binding Hit score.

## Models and inputs

- Ours: `pgot_oneshot_memory8_register32_noprior/checkpoint-5000`, NOT the new
  7B checkpoint. Reuses image-only AR captions and their original cached region
  maps from `outputs/figure_segmentation_over_under`.
- CTRL-O: authors' public `adidolkar123/pretrained_coco_vgcoco` checkpoint,
  global step **71000**, strict loading of all **272** model state tensors,
  including the frozen DINOv2 backbone. Native seven-slot `LangConditioning`
  and language-conditioned decoder; no retraining or random replacement layers.
- CTRL-O input is the normalized **category name** from Ours' AR caption,
  encoded with the official repository's
  `scripts/datasets/data/category_name_to_llama3_emb.pkl`. Remaining slots
  use `other` and conditioning mask zero, exactly the official demo convention.
- Full AR descriptions are retained in metadata, **NOT fed into CTRL-O**.
  Encoding arbitrary descriptions requires the official LLM2Vec + gated
  `meta-llama/Meta-Llama-3-8B-Instruct` base, which returned HTTP 401 in this
  environment. Never substitute another text encoder: it would invalidate the
  pretrained conditioning interface.
- Ours is **not** given the category as a new prompt. It already generated its
  caption/region pair from the image. Selection retrieves that original pair
  without changing the caption or mask. Therefore these figures are a
  *generated object/category-region comparison*, not a matched full-sentence
  referring-expression experiment or proof of arbitrary prompt-following.

Both see the same existing 512-square CODA center-crop. CTRL-O applies official
PIL bilinear square resize to 224, ToTensor, ImageNet normalization. Its official
224 mask probabilities are bilinearly upsampled to 512 before seven-slot argmax.
Ours retains its cached native readout. All labels compete, including unbound
CTRL-O slots; never select another slot by GT IoU to repair a query failure.

Inference uses fixed per-image seeds and no retries or best-seed selection.
The category query follows original AR order, capped at seven unique categories.
Repeated AR categories are omitted because category-only queries cannot identify
individual same-class instances. The released `LangConditioning` does not use
point input; points/bboxes carry the official visualization's -1 sentinel.

## Search and selection

Start from the prior Ours-quality shortlist of 160 COCO validation images,
not the full unbiased COCO population. Native inference is performed on the
subset having at least one usable unique category.
GT masks never enter either model's inputs. Evaluation/qualitative mining only
uses COCO noncrowd instance masks, nearest-neighbor 512 center crop, and the
existing instance-overlap ignore mask. Retain only one surviving GT instance
of the queried category and at least 2621 valid object pixels (1% of the image).
This avoids ambiguous category-only targets.

Strict candidate: Ours target IoU >= .70, CTRL-O target IoU <= .40, gap >= .35.
The completed search produced 180 eligible query-region pairs and 46 strict
candidate pairs. These numbers describe the preselected qualitative search,
NOT model accuracy on COCO. Manually curate ten distinct images for obvious
targets and visual diversity, avoiding heavily occluded objects/tiny fragments.
All curated examples still meet the strict rule; no mask alteration.

Final ten image IDs in order:
`112997,275791,474039,450686,312213,339823,22935,481573,98839,554735`.
Inspect `outputs/figure_ours_ctrlo_category/curated_selection.json` for scores.

Each curated folder contains:

- `comparison.png`: Image / GT reference / Ours AR / native CTRL-O, with IoU.
- `paper_panel.png`: Image / Ours AR / native CTRL-O, without GT/scores.
- `0_input.png`, `1_gt.png`, `2_ours.png`, `3_ctrlo.png`: separate 512 panels.
- `masks.npz`: original binary masks for this query, not GT-rematched predictions.
- `metadata.json`: exact CTRL-O input prompts, original AR description, slot
  labels, annotation identity, seed and source cache provenance.

Both models use identical binary-overlay color and alpha; unmasked pixels are
unchanged. These are segmentation overlays, **not AI-generated images**.

## Reproduce

The isolated `/home/jovyan/.venvs/ctrlo_eval` uses system packages from the
existing conda environment plus local CTRL-O dependencies; always set
`PYTHONNOUSERSITE=1`. Original conda package files were not modified.
The source repo and public checkpoint are in `/home/jovyan/baselines/CTRL-O`
and `/home/jovyan/baselines/ckpt/ctrlo`.

```bash
cd /home/jovyan/PGOT
GPU=<a-free-MIG-UUID-or-device> \
SELECTION_IDS=112997,275791,474039,450686,312213,339823,22935,481573,98839,554735 \
bash scripts/mine_ours_ctrlo_figures.sh
```

Inference is resumable. An existing metadata file skips that image; do not change
checkpoint, seed, query protocol, or source predictions in the same output root.
Use a fresh `OUTPUT_ROOT` for a different protocol. Visual curation can be rerun
on CPU without loading either model:

```bash
PYTHONNOUSERSITE=1 PYTHONPATH=/home/jovyan/PGOT \
/home/jovyan/.venvs/ctrlo_eval/bin/python -m pgot.eval.figure_ctrlo_examples \
  --stage select --selection_subdir curated \
  --selection_ids 112997,275791,474039,450686,312213,339823,22935,481573,98839,554735
```

Official sources: https://github.com/dido1998/CTRL-O and
https://huggingface.co/adidolkar123/pretrained_coco_vgcoco.
