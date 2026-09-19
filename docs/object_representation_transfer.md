# Native object-representation transfer pilot

Current priority: object-representation editing first; RefCOCO object selection
is deferred. This experiment does not train new weights or use an external image
editor/LLM selector. Original 1.5B Object 8 / Register 32 checkpoint is frozen:
`checkpoints/pgot_oneshot_memory8_register32_noprior/checkpoint-5000`.
There are eight memories per object and 32 per each of four background registers.

## Fixed protocol

Six manual same-category pairs are fixed in
`configs/object_representation_transfer_pairs.jsonl`: brown/black-white dogs
in both directions, grey/tabby cats in both directions, silver/patterned suitcase,
white/green bus. AR slots are zero-based and MUST agree with generated descriptions.
These curated sources include images previously shortlisted with GT quality;
this is a qualitative feasibility pilot, not an unbiased benchmark.
All pairs and failures are retained; no seed/step/intervention search is performed.

Source descriptions come from cached image-only DeOR AR generations. The data
manifest supplies image IDs/paths only, never its GT descriptions, masks/counts.
Owners are freshly extracted with all four PGOT token IDs registered, not copied
from old figure caches. Images are the original 512 center crops.

Each pair produces these five conditions with identical noise:

| Condition | Target semantic state | Target visual memory |
| --- | --- | --- |
| Reconstruction | A | A |
| Text only | A re-encoded with B's generated description substituted | Original A |
| Semantic only | B | A |
| Visual only | A | B |
| Full bundle swap | B | B |

Text-only never uses B image features. It substitutes one description in A's
caption and takes ONLY that owner's new semantic state; all other owners and
all original visual memories remain A's. It is a text-derived-semantic control,
not a separately trained text-to-image baseline. Donor semantic-only states,
unlike text-only states, come from processing image B.

All four registers, other objects and source RAE queries remain A's; exactly one
owner tuple is replaced in place. Native learned Reader -> DiT -> RAE decoder,
FP32, guidance 1, 10 steps, seed 1234. No GT/predicted spatial prior, foreground
memory rewrite, patch exclusion, spatial warp, pixel paste or inpainting.
Predicted masks are used only for overlays/change diagnostics, not to guide edits.
Rebuilding A's original owner set must reproduce its Reader condition within
relative max error 1e-3. Latents and conditions must be finite.

**Native donor representations encode context/location. Keeping other input
tuples unchanged does not guarantee unchanged pixels or preserved target pose,
location or scale.** Register appearance leakage may retain the original object.
This pilot deliberately measures native transfer before adding inference tricks.
The displayed instructions describe manual operations, not instructions interpreted
by DeOR. No general instruction-following capability is claimed.

## Run / resume

```bash
cd /home/jovyan/PGOT
GPU=0 bash scripts/run_object_representation_transfer.sh
```

`LIMIT=1` runs the first pair for smoke. An identical full rerun reuses its owner
cache and completed outputs, then processes remaining pairs. Model/source/pair/
seed/step changes require a fresh `OUTPUT_DIR`; incomplete runs preserve caches.
GPU2/3 training is not used or changed. Full pilot outputs are retained for review.

Default root: `outputs/object_representation_transfer_memory8_register32/`.

- `contact_sheet.jpg`: all six cases, never success-only filtering.
- `pairs/<pair_id>/figure.png`: thin target/donor outlines (not color-obscuring
  filled overlays), reconstruction, text-only,
  visual-only and full transfer. Header contains the manual operation.
- `pairs/<pair_id>/ablation.png`: all five generated conditions including semantic-only.
- `*_native.png`: actual native decoder outputs (no super-resolution).
- `metadata.json`: slots, original descriptions, identity error and pixel diagnostics.
- `manifest.json`, `protocol.json`, `owners/`: complete provenance and immutable
  extracted inputs with checkpoint-signature guards.
- `change_diagnostics.csv`, `summary.json`, `results.md`: per-case records;
  **not semantic edit-success metrics**.
- `visual_review.json`: manually inspected pilot outcomes and limitations, not
  blind human benchmark ratings.
- `validation.json`: CPU audit of all completed pair outputs, finite bundles,
  reconstruction identity and change diagnostics; not semantic edit success.

Change diagnostics compare every edit against the same-noise reconstruction.
Both target-only and target/donor-region union are recorded so relocation is not
hidden. These pixel-change values do not measure donor identity transfer.
Review donor appearance, intended placement and background/non-target preservation
separately. Successful category changes alone do not establish donor-appearance
transfer. To argue MLLM extraction is necessary, an encoder/extraction ablation
would still be required even if this pilot succeeds.

## Initial validation and observations (2026-09-18)

GPU0 smoke and all six fixed transfers completed; 30 native images (five
conditions per pair), decoder size 224x224. Rebuild relative max errors ranged
from 5.70e-6 to 9.78e-6, well below 1e-3. All six pairs passed the finite cache/
image/identity audit. Focused CPU tests passed (31 total including related
existing selection/figure tests). A full identical rerun verifies cache reuse.

Assistant qualitative review is in `visual_review.json`; this is neither blind
human evaluation nor benchmark accuracy. Dog coat and suitcase pattern transfer
are visible, whereas text-only changes largely retain A's appearance. Reverse
dog transfer deforms shape, cat examples are inconclusive/duplicated, and bus
transfer produces red rather than the selected green donor. Every case is kept.
No exact donor identity, pixel-perfect preservation, or necessity of MLLM
extraction is established by these examples.

```bash
PYTHONNOUSERSITE=1 PYTHONPATH=/home/jovyan/PGOT \
LD_LIBRARY_PATH=/home/jovyan/.conda/envs/scale_rae/lib \
/home/jovyan/.conda/envs/scale_rae/bin/python -m pgot.eval.run_object_representation_transfer --stage review
```

This creates `review_contact_sheet.jpg` and per-pair `review.png`, including
clear attribute transfer, partial, inconclusive and failure labels.
