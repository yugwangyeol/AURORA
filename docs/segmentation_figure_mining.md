# COCO/VOC qualitative split and merge mining

Run `GPU=0 bash scripts/mine_segmentation_figure_examples.sh` from PGOT.
Default output: `outputs/figure_segmentation_over_under`.
Stages can be selected individually: `coda metaslot adaslot ours select`.
`LIMIT=1` runs an inference smoke, and subsequent runs resume missing images.

## Protocol

- All four checkpoints were trained on COCO. Ours is memory8/register32/noprior,
  checkpoint-5000, FP32, batch 1, unrestricted register routing, OVT-owner masks.
- Ours replays captions from completed image-only AR full COCO/VOC evaluation;
  there is no teacher-forced GT caption and no additional diffusion generation.
- Cached AR text is encoded again, retaining special tokens and OVT order.
- Each baseline uses the existing official-checkpoint adapter and its own
  evaluation RGB preprocessing. Baseline soft masks are bilinearly upsampled
  to 512 then argmaxed, as in previous baseline evaluation.
- AdaSlot/MetaSlot randomness is fixed per image by a deterministic seed hash.
  No stochastic retry or best-seed optimization is performed. These are new
  qualitative runs, not exact replay of prior stochastic full-evaluation masks.
- GT is the same 512 CODA center-crop protocol and ignore policy as previous
  segmentation evaluation. fARI/mBO/mIoU are computed by equivalent contingency
  tables; foreground gaps and metric background matching are preserved.

## Selection, not a benchmark

This is intentionally GT-assisted qualitative case mining. Search starts from
images where the completed Ours AR run had high mIoU, with separate low/high
object-count search budgets, then compares all four newly cached maps.
It must not be presented as representative average performance.

Illustrative split/merge diagnostics exclude ignored pixels and GT background.
Only GT objects with >=786 valid pixels participate. A split means >=2 hard
labels each capture >=20% of one object's pixels; a merge means one hard label
captures >=20% of each of >=2 objects. These heuristic diagnostics are NOT
published standard metrics and still need visual review. All hard labels are
considered, including register background for Ours: multiple missed foreground
objects can appear as a merge into background. Reviewers must distinguish this
from a foreground-object merge when choosing a paper illustration.

STRICT candidates require all three baselines to have more target-type errors
than Ours, zero split and merge errors for Ours under the above diagnostic,
Ours mIoU >=0.60, and a >=0.10 mIoU lead over the strongest baseline.
If there are fewer than ten strict candidates, export weaker RELAXED candidates
explicitly marked for review; never claim that ten strict cases were found.

`selection.json` reports strict and review counts separately. `all_scores.json`
and `scores.csv` retain every completely evaluated candidate, not just winners.
`cache/<dataset>/<id>` contains original crop, GT/ignore, all four raw hard
predictions (`.npz`) and diagnostics (`.json`). Figures show Input, GT, AdaSlot,
MetaSlot, CODA, Ours: upper row maps and lower row RGB overlays.
Hungarian GT matching affects colors for comparison ONLY, never the masks,
metrics, captions, slot count, or model inputs. Ignore pixels are gray.

## Detached server execution

Keep the remote GPU server running. A tmux session can continue the batch after
the client PC disconnects; this does not guarantee that the IDE's assistant
session will continue reasoning or visually reviewing the outputs.
