# E2-Pix-FVW checkpoint-10000 analysis

## Fixed-protocol results

Evaluation uses 4,720 Pix2Cap-generated validation records, COCO instance GT,
CODA center-crop at 512, 32x32 (1,024) image patches, teacher-forced captions,
and overlap-excluded thing metrics.

| model/readout | fARI | mBO | mIoU | rFID |
|---|---:|---:|---:|---:|
| E2 | 0.1425 | 0.2534 | 0.3350 | 20.8517 |
| E2-Pix-OVT-Isolated | 0.0970 | 0.2194 | 0.3031 | 22.1546 |
| E2-Pix-FVW / FVW write attention / R2 | 0.0922 | 0.2174 | 0.2948 | 23.5756 |
| E2-Pix-FVW / Qwen core attention | 0.1062 | 0.2287 | 0.3114 | n/a |
| E2-Pix-FVW / OVT-register centroid | 0.1616 | 0.1414 | 0.2099 | n/a |

CODA reference segmentation scores supplied after this analysis are FG-ARI
0.475, mBO 0.363, and mIoU 0.3641 (CODA rFID 10.65).

The R2 foreground mask blocks only 3.24% of image patches. Against the COCO
foreground union it has precision 90.54%, recall 8.96%, and IoU 8.88%.

Changing the segmentation readout does not recover object masks. Qwen core
attention is only slightly better than the sparse FVW write map, while centroid
assignment raises fARI but sharply lowers mBO/mIoU. Centroid fARI also collapses
from 0.2854 for 1--3 objects to 0.0467 for 4--6 and 0.0226 for 11+ objects.

## Training dynamics

- Validation FVW inside mass rises from 0.5873 at step 500 to 0.6542 at step
  10,000, but its normalized entropy falls from 0.2303 to 0.0734.
- This corresponds to an effective support of about 4.93 patches at step 500
  and only 1.66 patches at step 10,000.
- Validation reconstruction loss is not monotonic: 0.6662 at step 500,
  0.5615 at step 6,000, 0.5904 at step 10,000, and 0.5795 at step 20,000.
- FVW inside mass peaks near step 16,000 (0.6679) and falls to 0.6333 at step
  20,000. Longer training does not repair the failure.

## Root-cause diagnostics

### Matched DINOv2 comparison

On 32 identical CODA-cropped object-removal interventions, final far-background
relative L2 change is 0.4003 for SigLIP2 and 0.1473 for DINOv2-ViT-B/14. Their
inside-object responses are comparable (1.1276 and 1.0852), so the far/inside
contamination ratio is 0.3550 for SigLIP and 0.1357 for DINO. DINO is still a
global transformer, but preserves local instance structure much better.

The independent GT-prototype feature oracle agrees: raw SigLIP reaches fARI
0.3281/mBO 0.3383, whereas raw DINOv2 reaches fARI 0.5796/mBO 0.5173 on the
same 32x32 grid. This is an oracle ceiling diagnostic, not a PGOT result.

### SigLIP leakage (D1)

Erasing an object before SigLIP leaves the patch embedding outside that object
unchanged, but changes far-away final SigLIP tokens by relative L2 0.3334 and
cosine distance 0.0969. The final vision tokens are globally contextualized;
blocking foreground token indices after SigLIP therefore does not remove all
foreground information from registers.

### Reconstruction pathways (D2)

On the fixed diagnostic examples, register-only reconstruction remains
plausible even after foreground access is blocked. Pre-SigLIP foreground erasure
still produces category-plausible but instance-wrong objects. RAE-self-only is
substantially worse, so FVW reduced the old RAE-self shortcut but shifted the
remaining shortcut toward globally mixed vision tokens and the generative
decoder prior.

### OVT swap and causal locality (D3/D4)

- Same-category OVT swaps produce a mean reconstruction change of 0.1361
  inside the target mask and 0.0743 outside (locality ratio 2.05).
- Swap localization AUROC is 0.7180; blocking the object's own OVT gives AUROC
  0.6922.
- Large objects localize better, while the small-dog example is close to random.
- The affected region is more local than in E2, but donor appearance does not
  transfer. Local change is not evidence that the OVT stores instance appearance.

### Appearance probes

The raw OVT predicts only R2=0.1738 of the visual target; a caption/word-only
baseline is better at R2=0.1909. After category and caption information are
regressed out, the OVT explains only 4.95% of instance residual variance, versus
16.45% for the LLM image stream and 33.47% for raw SigLIP.

FVW improves the OVT residual probe over E2-Pix-OVT-Isolated (1.47% to 4.95%),
but collapses the OVT representation: participation-ratio effective dimension is
4.65 out of 1,536, with only seven dimensions explaining 90% of variance.

## Interpretation

FVW partially succeeds at making OVT interventions spatially local, but fails to
make OVTs object-complete, instance-specific visual latents. Patch-softmax
outside loss has a degenerate optimum: each head can put nearly all probability
on one easy patch inside the mask. The role-routed reconstruction objective is
also not a true paired counterfactual: different samples randomly receive full,
OVT-only, or register-only routes, and the GT foreground support is injected by
zeroing RAE conditions and reused as the reconstruction loss mask. The decoder
can therefore minimize the objective without learning invariance to outside
content or sensitivity to the object's appearance.

The final FVW write after layer 27 is not consumed by another transformer layer,
so RAE query states used for reconstruction cannot causally read that last write.
At evaluation, the sparse predicted support blocks only 3.24% of register image
access, leaving most foreground information reachable through globally mixed
SigLIP tokens.
