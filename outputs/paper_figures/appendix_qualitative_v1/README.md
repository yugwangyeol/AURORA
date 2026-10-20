# Appendix qualitative figures

E.1, E.2, E.4, E.5, and E.6 use the Reader-supervised DeOR-B checkpoint:
`/home/jovyan/PGOT/checkpoints/pgot_oneshot_memory8_register32_noprior/checkpoint-5000`

E.3 uses the saved DeOR-L 7B compositional outputs. Every directory contains a PDF,
a lossless PNG, a lightweight JPG preview, and a provenance manifest.

- **E.1 COCO:** `e1_object_decomposition_coco/object_decomposition_coco.pdf`  
  Object decomposition from sparse to crowded COCO scenes.

- **E.1 VOC:** `e1_object_decomposition_voc/object_decomposition_voc.pdf`  
  Zero-shot object decomposition from sparse to crowded VOC scenes.

- **E.2:** `e2_description_region_alignment/description_region_alignment.pdf`  
  Verbatim autoregressive descriptions paired with their predicted regions.

- **E.3:** `e3_compositional_generation/compositional_generation.pdf`  
  DeOR-L object representations recomposed across images.

- **E.4:** `e4_language_guided_editing/language_guided_object_editing.pdf`  
  Text-only language selection followed by object removal or appearance transfer.

- **E.5:** `e5_direct_manipulation/direct_object_manipulation.pdf`  
  Direct remove, replace, add, and background-transfer operations.

- **E.6:** `e6_failure_cases/failure_cases.pdf`  
  Observed description, decomposition, and editing failures.

Case reuse is checked by `overlap_audit.json`. Selection is qualitative and is not
reported as an unbiased quantitative estimate.
