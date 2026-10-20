# Appendix qualitative revision v2

All revised deliverables remain in their original figure folders and use the
`*_v2` suffix. The original v1 assets were not overwritten.

- E.1 COCO: replaced row 2 of `4–6` (`160556` -> `568710`) and row 1 of
  `7–10` (`86582` -> `209613`).
- E.2: expanded from six to eight examples (four rows). The zebra case was
  removed because the generated age claim was not visually justified. Every
  displayed first sentence was manually checked against its image and mask.
- E.3: intentionally not rebuilt.
- E.4: restored the reference-style left/right layout. The two weak removal
  examples and pizza example were removed. It now includes three clean removes,
  cross-category train-to-bus appearance transfer, cross-category full-bundle
  toilet-to-bench replacement, and suitcase appearance transfer.
- E.5: expanded to four rows by six columns, with two examples for each direct
  operation. Failed additional-object attempts were audited but excluded.

`overlap_audit_v2.json` verifies that displayed source/donor images are not
reused across revised E.1/E.2/E.4/E.5 or the audited main-paper figures.

Rebuild code: `pgot/eval/build_appendix_qualitative_figures_v2.py`.
