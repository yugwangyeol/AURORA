# E2 D1–D4 Root-Cause Diagnostics

## Setup

- Checkpoint: `checkpoints/pgot_lviscap_hardreg_e2/checkpoint-10000`
- Validation manifest: `data/pgot_pix2cap_generated_val5k.jsonl` (4,720 images)
- One OVT per object
- CODA 512 center-crop preprocessing
- Fixed diffusion noise seed: 123
- Diffusion inference steps: 25

## D1 — Does a final SigLIP background patch contain foreground information?

For 64 images, one object occupying 1–35% of the 32×32 patch grid was
replaced before SigLIP with that image's mean background colour. Every pixel
outside the object stayed bit-identical. Original and object-removed SigLIP
features were compared at every layer and grouped by distance from the object.

| SigLIP state | Far-background relative L2 | Far-background cosine distance |
|---|---:|---:|
| Patch embedding (0) | 0.0000 | 0.0000 |
| Layer 1 | 0.0127 | 0.00015 |
| Layer 7 | 0.0805 | 0.00617 |
| Layer 14 | 0.1707 | 0.03324 |
| Final layer (27) | 0.3334 | 0.09693 |

The intervention has exactly zero effect on non-overlapping patch embeddings,
but the effect spreads across the full image as SigLIP becomes deeper.
Therefore E2's Qwen-level register hard mask is applied too late to guarantee
that a permitted background token is free of foreground information.

Outputs:

- `outputs/analysis_e2_d1_d4_rootcause/D1_siglip_leakage/summary.json`
- `outputs/analysis_e2_d1_d4_rootcause/D1_siglip_leakage/layerwise_leakage.png`

## D2 — Does reconstruction actually use this leakage?

The expanded test uses 16 one-object images whose object occupies 5–30% of
the crop. With fixed diffusion noise, it compares:

1. E2 unrestricted;
2. oracle post-SigLIP register hard mask;
3. post-SigLIP register-only;
4. foreground removed before SigLIP, register-only;
5. RAE self-only;
6. original OVT plus registers transplanted from the pre-SigLIP-masked image.

Changing only the register evidence from post-SigLIP to pre-SigLIP-masked
causes a mean absolute reconstruction change of:

- foreground: 0.1923;
- background: 0.1356.

Thus the change is stronger in the removed object's region. In visual examples,
detailed cats, dogs, and bears survive with post-SigLIP register-only but
mostly disappear or become weak hallucinations after pre-SigLIP removal.

Pixel MSE against the source does not monotonically worsen because E2
reconstruction is not pixel-faithful and a scene prior can accidentally make a
similar object. A person can still be hallucinated in a mirror even after the
person pixels are removed. Thus two effects coexist:

1. final-SigLIP background tokens leak foreground evidence;
2. background layout plus the decoder prior can predict likely objects even
   after the direct visual evidence is removed.

The original-OVT/masked-register hybrid improves foreground MSE over the
masked-register-only condition in 12/16 images, but the mean improvement is
small (0.0112 MSE) and the recovered objects are usually blurred or generic.
This is evidence that E2 OVTs have some useful object signal, but not enough
instance-specific visual detail.

Outputs:

- `outputs/analysis_e2_d2_rootcause_16/D2_register_leakage/summary.json`
- `outputs/analysis_e2_d2_rootcause_16/D2_register_leakage/image_*_fixed_noise_grid.png`

## D3 — Same-category appearance swaps

Six one-object pairs control category while changing appearance:

- brown bear → white polar bear;
- black dog → white/brown dog;
- black cow → white/brown cow;
- ginger cat → grey striped cat;
- black horse → white/brown horse;
- red bird → blue/orange bird.

The donor OVT is substituted before every Qwen layer and self/swap
reconstructions use identical diffusion noise.

The intended appearance generally does not transfer. The bear stays brown,
the bird stays red, the horse stays dark, and the cat remains mostly orange.
For the dog, the swap changes global object placement/composition rather than
only appearance.

Mean absolute self-vs-swap image change:

- inside the A-object mask: 0.0688;
- outside: 0.0629;
- inside/outside ratio: 1.294.

The causal effect is only weakly object-local and is not a reliable transfer of
instance appearance.

Outputs:

- `outputs/analysis_e2_d3_d4_rootcause_expanded/D3_same_category_swap/summary.json`
- `outputs/analysis_e2_d3_d4_rootcause_expanded/D3_same_category_swap/pair_*.png`

## D4 — Is there a 256-query spatial binding?

For each of the six D3 source images, two interventions are measured at the
final 256 RAE queries and reshaped to 16×16:

1. donor OVT swap;
2. blocking A's own OVT from all RAE queries.

| Intervention | AUROC vs A-object mask | Inside/outside change | Top-GT-area precision |
|---|---:|---:|---:|
| Donor OVT swap | 0.683 | 1.949 | 0.505 |
| Block A's own OVT | 0.822 | 3.015 | 0.629 |

Blocking A's own OVT changes the RAE queries in the correct source-object
region, so the current E2 model already has a real, though weak, spatial
binding. The final relative query change is only about 4.6% inside versus
1.5% outside.

Donor swaps are less localized because an OVT is not a position-invariant
appearance code. The clearest dog example has:

- donor-swap AUROC: 0.433;
- own-OVT-block AUROC: 0.819.

The donor swap changes the upper-right RAE queries, matching the donor's
geometry/context, while blocking A's own OVT changes the left-side queries
where A's dog actually lies. Therefore “no binding mechanism exists” is too
strong. The more accurate conclusion is:

- own-image OVT-to-RAE spatial binding exists;
- its causal magnitude is weak;
- OVT content entangles appearance, geometry, and scene context, so it is not
  a clean exchangeable object representation.

Outputs:

- `outputs/analysis_e2_d3_d4_rootcause_expanded/D4_rae_binding/summary.json`
- `outputs/analysis_e2_d3_d4_rootcause_expanded/D4_rae_binding/pair_*_binding.png`

## Updated root-cause conclusion

1. **Confirmed shortcut:** register-visible final SigLIP background tokens
   contain foreground information before Qwen's hard mask.
2. **Residual shortcut:** background layout and DiT/decoder priors can still
   hallucinate a likely object after its pixels are removed.
3. **OVT content problem:** OVTs carry category/rough appearance but little
   reliable instance appearance; same-category appearance swaps mostly fail.
4. **Binding is not absent:** source OVTs influence the correct spatial RAE
   queries, but weakly. Donor swaps expose geometry/context entanglement.
5. **Consequent training problem:** reconstruction can be solved without
   placing sufficient instance-specific visual information in the OVT.

## Cheaper alternative to a second full SigLIP register branch

Running original and foreground-masked SigLIP encoders is not required at
inference. D1 shows that layer 0 background features have zero measured
foreground leakage and layer 1 has only 0.0127 far-background relative change.
SigLIP already computes hidden states internally.

A cheaper design is:

1. OVT visual write uses the final semantic SigLIP features.
2. Register visual write uses layer-0 or layer-1 local features from the same
   SigLIP forward, through a small register-to-local-feature cross-attention.
3. Register queries are blocked from the shared final SigLIP image tokens.
4. During training, GT foreground masks restrict register local attention.
5. During inference, generated OVT ownership restricts the same attention in a
   small routing/RAE pass; SigLIP is not run twice.

This removes the measured final-feature leakage cheaply. It does not alone
prevent scene-prior hallucination, so foreground RAE conditions must also be
causally dependent on their owner OVT and not recoverable from registers.

A learned filter is possible using an original-vs-object-removed register
invariance loss, but it needs a second SigLIP forward during at least some
training batches and has a degenerate solution: both register states may encode
the contextually hallucinated object. It is therefore weaker than using local
features plus explicit foreground/background causal routing.

## E2-Pix-FVW outcome (checkpoint 10k)

The forced visual-write (FVW) experiment did not solve the object
representation problem. On the fixed 4,720-image CODA-compatible validation
set, the main FVW readout obtains:

| Readout | FG-ARI | mBO | mIoU | rFID |
|---|---:|---:|---:|---:|
| FVW main | 0.0922 | 0.2174 | 0.2948 | 23.5756 |
| Original/core OVT attention | 0.1062 | 0.2287 | 0.3114 | — |
| Centroid clustering | 0.1616 | 0.1414 | 0.2099 | — |

The main FVW readout being worse than the unchanged core attention is direct
evidence that the added visual-write state did not become a better ownership
representation. During training, its effective attention support collapsed
from about 4.93 patches to 1.66 patches. The appearance probe also remains weak
(residual R² 0.0495; effective representation dimension 4.65). Thus forcing a
write from a small set of in-mask positions allows a sparse positional
shortcut; it does not force instance-complete visual information into the OVT.

Reproducibility outputs (not committed because `outputs/` is an artifact
directory):

- `outputs/eval_pgot_e2_pix_fvw_10k_r2_tf_coda512_rfid/summary.json`
- `outputs/eval_pgot_e2_pix_fvw_10k_core_attention_tf_coda512/summary.json`
- `outputs/eval_pgot_e2_pix_fvw_10k_centroid_tf_coda512/summary.json`
- `outputs/analysis_e2_pix_fvw_10k_appearance_probe_512/summary.json`

## Matched SigLIP versus DINOv2 leakage probe

A matched 32-intervention probe compares original and object-removed images on
the same CODA crop and 32×32 grid. The relative far-background feature change
is 0.4003 for final SigLIP features and 0.1473 for final DINOv2 features;
normalized by the inside-object change, the leakage ratios are 0.3550 and
0.1357 respectively. DINOv2 therefore mixes substantially less removed-object
information into distant patches in this probe, although its leakage is not
zero.

The frozen-feature nearest-centroid oracle is also stronger with DINOv2:

| Frozen feature | FG-ARI | mBO |
|---|---:|---:|
| SigLIP | 0.3281 | 0.3383 |
| DINOv2 | 0.5796 | 0.5173 |

This result supports testing DINOv2 as the spatial visual source, but it does
not by itself guarantee that OVTs will use those features. The decoder path
must still be made causally dependent on object-owned OVT information, and the
register path must be unable to reconstruct foreground objects independently.

Probe output:

- `outputs/probe_encoder_leakage_siglip_dino/summary.json`
