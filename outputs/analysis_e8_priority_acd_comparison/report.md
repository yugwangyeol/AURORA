# E8 A/C/D priority analysis (visible-object protocol)

- Dataset: first 512 validation images, 2,857 visible GT objects.
- 347 captioned object slots whose GT masks disappeared after center crop remain model competitors but are excluded as evaluation targets.
- Causal branches use the same RF timestep and noise; only one object memory is zeroed.
- Appearance probes use an image-disjoint 70/10/20 train/validation/test split and word-residual targets.

## Compact comparison

| Metric | A | C | D |
|---|---:|---:|---:|
| Full-eval fARI | 0.6131 | 0.4386 | 0.1342 |
| Full-eval mBO | 0.5368 | 0.3385 | 0.1487 |
| rFID | 18.9945 | 19.5171 | 21.5660 |
| Writer L27 named IoU | 0.4512 | 0.3083 | 0.0961 |
| Writer L27 oracle IoU | 0.4693 | 0.3184 | 0.1525 |
| Writer FG consistency 21→24 | 0.8467 | 0.5721 | 0.4216 |
| Writer FG consistency 24→27 | 0.8993 | 0.6347 | 0.5907 |
| Causal diagonal share (mean) | 0.6435 | 0.6478 | 0.3052 |
| Selected region dominant | 0.7063 | 0.7095 | 0.1806 |
| Global diagonal/other influence | 8.6086 | 8.8306 | 1.0094 |
| Direct self appearance R² | 0.1362 | 0.1557 | 0.0703 |
| Direct non-self appearance R² | 0.0516 | 0.0546 | 0.0652 |
| Unique self ΔR² | 0.0845 | 0.0886 | 0.0002 |
| Reader target-region self mass | 0.4307 | 0.4250 | 0.1364 |
| Reader target-region other-object mass | 0.3464 | 0.3141 | 0.6278 |
| Reader target-region register mass | 0.2229 | 0.2609 | 0.2358 |

## Findings

1. **C is not a slot-permutation failure.** Its layer-27 named/oracle IoU is 0.308/0.318; oracle matching recovers only 0.010. Object partition quality itself is worse than A.
2. **C still stores and uses named appearance.** Its direct self-memory residual R² and causal locality are at least as strong as A. GT Reader supervision can anchor reconstruction responsibility even when the Writer ownership map is poor.
3. **D loses named responsibility.** Self Reader mass falls to 13.6% while other-object mass rises to 62.8%. Its unique-self ΔR² is 0.0002, effectively zero.
4. **A is partially, not fully, disentangled.** A has strong global diagonal/other causal influence, but only 70.6% of interventions are strongest on the selected object. The Reader assigns only 43.1% of target-region mass to the matching memory.
5. **Next experiment:** continue A with E8.2. The evidence points to an exclusivity/routing problem, while A already has non-zero unique appearance. Evaluate whether E8.2 raises selected-dominant fraction, diagonal share, Reader self mass, and unique routed ΔR² without degrading rFID.
6. **Separate GT-free branch:** C/D do not reproduce CODA's contrastive slot-image alignment. Removing Writer GT should be revisited only with an explicit self-supervised alignment/iterative competition objective.

## Causal locality by object area

| Area on 16×16 target grid | A dominant | C dominant | D dominant |
|---|---:|---:|---:|
| tiny_<2% (n=848) | 0.5578 | 0.5613 | 0.1285 |
| small_2-5% (n=430) | 0.7116 | 0.7140 | 0.1907 |
| medium_5-10% (n=404) | 0.7450 | 0.7748 | 0.1733 |
| large_>=10% (n=1175) | 0.7983 | 0.7923 | 0.2170 |

A/C의 가장 약한 구간은 면적 2% 미만 객체로, selected-region dominant 비율이 약 56%다. E8.2 평가는 전체 평균뿐 아니라 이 bucket을 별도로 봐야 한다.
