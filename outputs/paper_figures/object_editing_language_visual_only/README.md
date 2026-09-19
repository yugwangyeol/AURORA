# Object editing with visual-only appearance transfer

Only the snowboard edited panel was switched from full bundle to the previously generated, audited visual-only comparison image. Other example panels and target/donor selections are unchanged. No GPU inference was needed.

The snowboard display caption is "Transfer the surfboard's appearance from image B to the snowboard in image A." This caption describes the visual-only operation; the recorded original external-LLM selector request remains "Replace the snowboard in image A with the surfboard from image B." The new display wording was not submitted again to the selector. Actual original requests, selections, output paths and hashes are retained in manifest.json.

Regenerate the figure:
```bash
cd /home/jovyan/PGOT
PYTHONNOUSERSITE=1 /home/jovyan/.conda/envs/scale_rae/bin/python -m pgot.eval.render_language_object_editing \
  --output_dir outputs/language_object_editing_transfer_replace_gpu0 \
  --figure_output_dir outputs/paper_figures/object_editing_language_visual_only \
  --display_overrides configs/object_editing_language_visual_only_figure.json
```
