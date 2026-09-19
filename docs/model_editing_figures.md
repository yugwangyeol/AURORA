# Native PGOT editing figures

Generate candidate figures for four operations: remove object, add a donor
object, replace an object bundle, and replace background registers.
The requested delivery is eight example sets per operation (32 sets).

## Scope and honesty

These are **real outputs of PGOT's existing Reader -> DiT -> RAE decoder**,
not external image-editor output or RGB copy/paste. The current checkpoint
is `pgot_oneshot_memory8_register32_noprior/checkpoint-5000`, not the new 7B
checkpoint. Its memory counts are eight visual memories per object and 32 per
background register, with four background registers.

The model is not trained as a freeform instruction-following image editor.
An explicit operation chooses previously generated object/region units;
the code executes that operation on learned semantic and visual-memory
bundles. Category labels shown in figures describe the operation; they are
not new editing instructions processed by the MLLM.

Source captions are cached **image-only AR generations**, re-encoded without
GT caption input. Neither model forward nor editing uses GT masks. The source
shortlist originally came from GT-assisted qualitative mining; this is not an
unbiased test set. All overlays use unmodified cached predicted masks.

## Editing mechanics

- Remove: drop the target object owner. Because the noprior Writer can leak
  object appearance into other memories, also rewrite memories while excluding
  that object's own predicted winning patches, dilated by one patch. This is
  the existing `probe_removal_and_instruction.rewrite_memories` inference
  intervention, not a trained eraser.
- Add: retain A objects/registers and append B's selected object bundle.
- Replace: predicted-patch removal of A's target, then append B's object bundle.
- Background: retain A object bundles and exchange all background registers
  with B's. B registers are rewritten with all B predicted foreground winning
  patches excluded (same one-patch dilation) to reduce donor-object leakage.

No GT owner prior is inserted. All edits retain the checkpoint's existing
semantic states, learned Writer/Reader weights and diffusion/decoder weights.
Donor appearance encodes the donor's native spatial location. No prompt-directed
placement, scale change, or reliable occlusion/inpainting is claimed.
Background/object units can still leak appearance; selected image edits may
be imperfect even if the operation is implemented correctly.

Rebuilding each original owner set must agree with original `rae_hidden`
within relative max error 1e-3, or generation fails. Reconstruction and all
edits use identical per-target diffusion noise (seed 1234), guidance 1, ten
diffusion steps, FP32. Source images are shared 512 center-crops. The generator
has its native decoder resolution; figure resizing does NOT improve its true
resolution. `edited_native.png` and `reconstruction_native.png` retain it.

## Outputs and selection

Default directory: `outputs/figure_model_editing_memory8_register32_noprior`.

- `tasks.json`: fixed source images, AR text provenance, operation and object IDs.
- `owner_cache/*.pt`: native forward results (CPU tensors), reuse/resume only
  with the same checkpoint, source captions and inference settings.
- `candidates/{remove,add,replace,background}/...`: all actual outputs.
- `selected/{operation}/...`: eight candidates per operation.
- `selected/{operation}_contact_sheet.jpg`: review sheet per operation.
- `selected_selection.json`: identities and change-localization scores.
- Final visually reviewed delivery: `final/{operation}/...`,
  `final/{operation}_contact_sheet.jpg`, and `final_selection.json`.
  The run wrapper defaults to this delivery subdirectory.
- `final/overview_all_32.jpg`: reference-like four-quadrant sheet with all
  32 sets and visible qualitative review labels.
- `protocol.json`: model, inference and editing-intervention details.
- `visual_review.json`: qualitative review labels/limitations per task (not an
  independent quantitative metric); selection prefers success then partial.

Each set contains a three-column `figure.png` analogous to the reference,
original/donor images, separate panels, native generated output, and
`sanity.png` (Original / Unedited reconstruction / Actual model edit).
Always check `sanity.png`: comparing only against the original can confuse
reconstruction changes with successful edits.

Automatic ranking uses in-region pixel change minus .75 times outside change,
relative to the unedited reconstruction with identical noise. This is a
**screening heuristic, NOT semantic edit success**. It cannot tell whether an
object was actually removed/added/replaced; manual inspection remains necessary.
Metadata initializes `visual_success_verified=false`. Reviewed delivery uses
`visual_review.json`; only explicitly reviewed clear-operation successes set
that field true, and selected review sheets visibly label SUCCESS/PARTIAL/FAILURE.
The clean `figure.png` is retained alongside labeled `review.png`. Do not call all 32
generated candidates successful paper examples without reviewing them.
Never count reconstructed blur, complete scene collapse, or a different object
as success. Report deficiencies plainly rather than synthesize better output.

Current delivery was selected after reviewing all 96 native edit candidates.
With eight distinct target images per operation, qualitative labels are:

| Operation | Clear operation | Partial | Failure |
| --- | ---: | ---: | ---: |
| Remove | 7 | 1 | 0 |
| Add | 4 | 3 | 1 |
| Replace | 5 | 3 | 0 |
| Background | 0 | 8 | 0 |

These labels describe visible operations, not photorealistic quality or an
independent benchmark. Background examples all exhibit foreground deformation
and/or donor content leakage. The add failure is retained to satisfy the
requested eight distinct source scenes, not as a claimed success.

## Run / resume

```bash
cd /home/jovyan/PGOT
GPU=<free-device-or-MIG-UUID> bash scripts/make_model_editing_figures.sh
```

`CANDIDATES=16` generates 16 attempts per operation, then `EXAMPLES=8` exports
eight per operation. `LIMIT=4` is the four-operation first-target smoke run.
Existing task metadata skips completed generation. To change seed, checkpoint,
source captions or interventions, use a fresh output root rather than reusing
cache. Do not overwrite a task manifest while an inference process is running.

After initial 64 edits, an additional bounded search appended 16 add and 16
background tasks (96 total). Add donors are larger; background donors exclude
person captions. This changes source-pair selection only, not the editing
algorithm, model weights or diffusion seed. Run `--stage augment` only when no
render process is running; it is idempotent and does not alter original tasks.

Selection alone is CPU-only:

```bash
PYTHONNOUSERSITE=1 /home/jovyan/.conda/envs/scale_rae/bin/python \
  -m pgot.eval.figure_model_editing --stage select \
  --selection_subdir final
```
