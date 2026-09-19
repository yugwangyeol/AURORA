# Language-selected object editing: Remove / Transfer / Replace

Main figure: `object_editing_language_selected.pdf` (PNG and SVG also provided).
- Left: remove cup, bench, giraffe using selected predicted ownership, patch exclusion and Writer rewrite.
- Right, rows 1 and 3: appearance transfer for dog and refrigerator; retain source semantic owner and substitute the donor visual memory.
- Right, row 2: replace snowboard with surfboard; substitute the donor semantic owner and visual memory together (full bundle).

`replacement_comparison_edit_004.pdf` compares reconstruction, visual-only `(s_A, M_B)` and full bundle `(s_B, M_B)` with the same initial diffusion noise and actual LLM-selected target/donor IDs. All other owners/registers and source RAE query are retained.

New requests were actually sent to the text-only external Qwen3-VL-4B controller; native PGOT images were regenerated on GPU 0. Six main examples are curated qualitative examples, not an editing benchmark. The checkpoint is the original standard M8/R32 no-prior checkpoint-5000 used in the prior language editing figure, not the separate 7B checkpoint. Green outlines/fills show predicted masks for display only; generated output pixels are unaltered.

Selection-execution audit covers all 8 requests. This checks execution of recorded selections, not correct interpretation or editing success. The additional negative-control request edit_007 (remove absent airplane) incorrectly selected Pizza in this run; that decision and its generated result are retained in the source experiment and review sheet without manual correction.

Reproduce or resume:
```bash
cd /home/jovyan/PGOT
GPU=0 bash scripts/run_language_object_editing_transfer_replace.sh
```

Source experiment: `/home/jovyan/PGOT/outputs/language_object_editing_transfer_replace_gpu0`
Fingerprint: `8b02842bed2c0551e26761e0e7295d88b43d9fbfd8f94572248a481a0051b94f`
