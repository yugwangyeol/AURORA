# PGOT AR + class/point router Binding Hits

```bash
cd /home/jovyan/PGOT
GPU=0 bash scripts/eval_pgot_memory8_binding_hits_ar_router.sh
```

Default checkpoint: `pgot_oneshot_memory8_register32_noprior/checkpoint-5000`.
Object memory 8, register memory 32 x 4, writer owner prior disabled. Batch 1,
FP32, 512 evaluation grid, AR token budget 512, AR capacity 64, seed 0, up to
seven fixed queries per image. All official 5,000 COCO val2017 images are kept.
No diffusion generation or retraining is performed.

## Two-stage pipeline and GT separation

1. Greedy AR generates descriptions and OVTs from RGB images only, using the
   existing model's training-compatible prefix/attention and pass-two packing.
2. Forward **the generated caption only** through PGOT to obtain ownership maps,
   semantic slots, visual memories, and hard masks. Register route is unrestricted.
3. Parse every original OVT without compacting positions. Nonempty `thing` masks
   with parseable generated category labels become router candidates.
4. Router matches input class names to generated class names, with a fixed small
   synonym map (`man/woman -> person`, `sofa -> couch`, etc.). Among compatible
   candidates it minimizes total squared predicted-centroid/input-point distance
   using a one-to-one assignment. Dummy assignments allow missing candidates.
5. Only AFTER routing, GT masks and fixed target identities enter the IoU scorer.

This is a **lexical class + point router**, not CLIP/VLM semantic retrieval and
not arbitrary free-form referring-expression understanding. It does not use full
query descriptions. `QUERY_MODE` controls the exported reference text and whether
points are available, but all class-point template variants give the router the
same structured class/point inputs. Re-running different point prompt templates
does not constitute different AR-router experiments.

Coordinates are rounded to four decimals, exactly the precision transmitted by
the original direct text queries. Centers come ONLY from predicted hard masks,
never GT pixels. Category aliases, center rule, and unmatched costs are fixed
before evaluation. `CENTER_SOURCE=mask` uses centers of mass (default); `bbox`
uses predicted-mask bounding-box centers, consistently with that query variant.
Points need not lie inside a mask. Router API accepts only class/point keys and
rejects GT annotation IDs, target indices, masks, ignore maps, and category IDs.

The existing annotation-derived query sampler, crop, overlap handling, distractor
GT instances, target annotation IDs, query ordering, and text-independent identity
hash are reused unchanged. GT class/point information enters ONLY the router,
not AR generation or mask extraction. No GT count limits AR generation. AR runs
on GT-empty scenes as well. Missing/malformed AR objects and routing failures
are never used to filter evaluation images or queries. AR token/capacity cutoff
and format/scene-end rates remain visible in the output.

## Scoring and comparison boundaries

Primary `binding_hits_percent` uses the original Q-query-region + background
Hungarian-IoU rule against ALL visible GT instances. Each selected AR region gets
the query's label; register background and all unselected AR regions are merged
into residual background. **Unrouted queries always count as misses**, even if
an empty-region/zero-IoU Hungarian tie might otherwise produce a hit.

Secondary `all_ar_slots_metric` keeps ALL original AR regions (including
unselected distractor predictions) plus original register background in the
Hungarian pool, then scores only the labels already selected by the router. It
is provided because changing the prediction/background pool can affect assignment.
Report the pool explicitly; do not mix primary and secondary numbers. The same
quality-aware target IoU / IoU50 / positive-IoU / zero-IoU-hit diagnostics are kept.

This is **AR discovery followed by retrieval/selection**, not native query-guided
object extraction and not exact reproduction of CTRL-O's published result. It
cannot recover objects missed or merged by AR. A pipeline comparison should use
the same query manifest, scoring pool, and downstream selection budget for other
models. Improved scores do not establish better native query controllability.

## Outputs

Default folder:
`outputs/binding_hits_pgot_memory8_register32_noprior/ar_router_class_point_natural/`

- `summary.json`: primary/secondary scores, routed/missing counts, AR rates,
  GT-separation flags, query identity hash, checkpoint and router protocol.
- `per_image.jsonl`: each fixed query, selected AR label/category/center, candidate
  count, missing reason, IoU score and secondary score, generated caption/slots.
- `query_manifest.jsonl`: original query identities plus exact AR/router inputs.

Existing evaluation artifacts are refused by default. Use a different
`OUTPUT_DIR`, rather than overwriting completed evaluations unintentionally.

```bash
# GPU smoke: same ordinary and GT-empty images as direct Binding Hits.
GPU=0 SMOKE=1 NUM_WORKERS=0 bash scripts/eval_pgot_memory8_binding_hits_ar_router.sh

# Exploratory subset; keep its results separate from full-split output.
GPU=0 MAX_SAMPLES=100 \
 OUTPUT_DIR=/home/jovyan/PGOT/outputs/binding_hits_pgot_ar_router_first100 \
 bash scripts/eval_pgot_memory8_binding_hits_ar_router.sh

# CPU regression tests (original metric + new router).
PYTHONNOUSERSITE=1 OMP_NUM_THREADS=3 \
 /home/jovyan/.conda/envs/scale_rae/bin/python -m unittest discover \
 -s tests -p '*binding*.py' -v
```

Verified 2026-09-17: 13 CPU tests passed, including unchanged direct metric
regressions, same-class one-to-one routing, hard class constraints, invalid GT
metadata rejection, malformed/stuff/empty slots, zero-AR predictions, unmatched
queries, and full-slot scoring. FP32 CUDA smoke completed on images 139 and
25593. Both images ran AR, all seven fixed queries were retained (five routed,
two unmatched), and the query identity hash matches the original direct smoke.
GT-separation flags, checkpoint memory8/register32/noprior configuration, and
per-query class/point inputs were checked. Smoke results demonstrate execution,
not full-dataset performance; the full 5,000-image AR-router run has not begun.
