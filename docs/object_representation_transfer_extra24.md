# Additional qualitative candidates (GPU1)

User requests: keep the original pilot's **row2** (`dog_black_white_to_brown`)
and **row5** (`suitcase_silver_to_patterned`), generate multiple new candidates,
and deliver the best two additional examples. Original files are not overwritten.

24 extra pairs are fixed before inference in
`configs/object_representation_transfer_extra24.jsonl`, including the requested
193429 snowboard -> 554595 surfboard pair, five other snowboard transfers,
phones, refrigerators, mice, benches, umbrellas, reverse suitcase, shaggy dogs,
pizzas, buses and reverse surfboard transfer. Categories/owner IDs are verified
against GENERATED AR text, not GT labels. Category-changing text-only controls
substitute BOTH donor category and description while retaining the same OVT index
and source visual memory. Same-category controls and the old six cases are unchanged.

**Do not mix algorithms:** the earlier
`figure_model_editing_memory8_register32_noprior/final/replace/07_193429`
example used predicted target-patch exclusion / Writer memory rewriting.
This new batch instead uses the current pilot's native direct one-owner replacement,
without Writer rewrite, spatial prior, patch exclusion or pixel paste. The same
source/donor pair is freshly rendered, not copied from the earlier edit figure.

Frozen original 1.5B Object8 / Register32 (four background registers), all-FP32,
same per-target seed1234 noise, 10 steps, guidance1, original Reader->DiT->RAE
decoder. Every pair produces reconstruction, text-only, semantic-only,
visual-only and full owner swap (120 native images planned). No seed/step/model
search or training. GPU1 only; existing other-GPU tasks are not changed.

```bash
cd /home/jovyan/PGOT
GPU=1 \
PAIRS=/home/jovyan/PGOT/configs/object_representation_transfer_extra24.jsonl \
OUTPUT_DIR=/home/jovyan/PGOT/outputs/object_representation_transfer_extra24_gpu1 \
bash scripts/run_object_representation_transfer.sh
```

`LIMIT=1` first validates the requested snowboard pair; the full identical
command resumes that case/cache and renders the other23. The original six-case
output remains `outputs/object_representation_transfer_memory8_register32`.

New output: `outputs/object_representation_transfer_extra24_gpu1/`:

- `pairs/<pair_id>/figure.png`: six-column source/donor/control/composition figure.
- `ablation.png`, `*_native.png`, masks, metadata: all actual outputs and diagnostics.
- `contact_sheet_page_01.jpg` ... `06`: four cases per page for readable review.
- `contact_sheet.jpg`: all24 (requires zoom).
- `validation.json`: finite owner/latent/image and identity audit, not edit success.
- `visual_review.json`: explicit qualitative review of ALL24 including failures.

Empty predicted regions, if any, are reported as model outcomes rather than
silently dropping cases or repairing ownership with GT. A finite-output/identity
audit is not a semantic success assertion.

After direct inspection, selection uses donor attribute transfer and non-target
preservation, not the pixel-change screening score alone. Qualitative best-example
selection is explicitly cherry-picked and must not be presented as benchmark
accuracy or an average success rate. Even a visually compelling snowboard transfer
may establish donor color/pattern transfer without proving surfboard category identity.

`pgot.eval.object_representation_gallery` assembles selected cases into a new
delivery directory, preserves native PNG pixels byte-for-byte, records original
source metadata/fingerprints and rejects mixed inference protocols. This avoids
silently combining old patch-exclusion edits with current native transfers.

## Completed delivery

GPU1 inference and audit completed: 24 pairs / 120 native images, all five
conditions per pair. Focused transfer/gallery/figure/selection tests: 40 passed.
All 24 pairs were visually reviewed; unselected and failed cases are retained.

- `selected_best4/overview.jpg`: original row2 dog, original row5 suitcase,
  new snowboard -> yellow patterned board, new green -> white/black bus.
- `selected_new2/overview.jpg`: just the two additional recommendations.
- `shortlist_new4/overview.jpg`: those two plus brown -> shaggy dog and
  patterned -> silver suitcase as alternatives.

All paths above are relative to the new output directory. Each gallery includes
byte-identical native images and `selection.json` with provenance. Assembly
selections are saved as configs named `object_representation_transfer_best4_selection.json`,
`object_representation_transfer_new2_selection.json` and
`object_representation_transfer_new4_shortlist.json` (no git commit implied).

Interpretation: the board demonstrates donor color/pattern transfer, not proven
surfboard identity. The bus demonstrates donor-like white/black exterior while
retaining the source double-decker geometry; the donor is a single-deck coach.
These remain manually selected qualitative examples, not automatic instruction
following or a benchmark success rate. Original pilot outputs are untouched.
