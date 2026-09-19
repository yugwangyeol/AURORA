"""Matplotlib-only composition of six existing edits; no generation or inference.

Replacement examples contain overview columns 1,2,3,5. Removal examples use
original/reconstruction/edited images, optionally including the saved mask overlay.
Labels are manual-ID edit descriptions, NOT evidence of
freeform language instruction following. Original image pixels are not retouched.
"""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from PIL import Image


def extract_panels(path):
    """Reuse exact overview image tiles, including existing predicted outlines."""
    with Image.open(path) as image:
        image = image.convert("RGB")
        if image.width % 6:
            raise ValueError(f"Expected six square image columns: {path}")
        side = image.width // 6
        top = image.height - side
        if top < 0:
            raise ValueError(f"Missing image strip: {path}")
        boxes = [(col * side, top, (col + 1) * side, image.height)
                 for col in (0, 1, 2, 4)]
        return [image.crop(box) for box in boxes], boxes


def wrap_prompt(text, fig, width_inches, font_size):
    """Wrap by rendered text width, without shrinking the requested font."""
    probe = fig.text(0, 0, "", fontsize=font_size, fontfamily="DejaVu Sans")
    renderer = fig.canvas.get_renderer()
    lines = []
    for word in text.split():
        candidate = (lines[-1] + " " + word) if lines else word
        probe.set_text(candidate)
        if lines and probe.get_window_extent(renderer).width > width_inches * fig.dpi:
            lines.append(word)
        elif lines:
            lines[-1] = candidate
        else:
            lines.append(word)
    if len(lines) > 2:
        probe.remove()
        raise ValueError("Prompt needs more than two lines; enlarge the layout")
    if len(lines) == 2:
        # Balance two-line prompts instead of leaving a single dangling word.
        words = text.split()
        choices = []
        for split in range(1, len(words)):
            candidate_lines = [" ".join(words[:split]), " ".join(words[split:])]
            widths = []
            for candidate in candidate_lines:
                probe.set_text(candidate)
                widths.append(probe.get_window_extent(renderer).width)
            if max(widths) <= width_inches * fig.dpi:
                choices.append((abs(widths[0] - widths[1]), candidate_lines))
        if choices:
            lines = min(choices, key=lambda choice: choice[0])[1]
    probe.remove()
    return "\n".join(lines)


def compose(selection, output_dir, dpi=200):
    chosen = json.loads(Path(selection).read_text())
    entries = chosen["examples"]
    if len(entries) != 6 or len({r["pair_id"] for r in entries}) != 6:
        raise ValueError("A 3x2 layout requires exactly six distinct cases")
    # Wide landscape canvas keeps eight images per row and readable prompts.
    width, height = 24.0, float(chosen.get("canvas_height_inches", 13.0))
    outer, group_gap, image_gap, side = 0.25, 0.60, 0.08, 2.77
    group_width = 4 * side + 3 * image_gap
    compact = chosen.get("compact", False)
    removal_mask_panel = chosen.get("removal_mask_panel", False)
    show_prompts = chosen.get("show_prompts", True)
    headers_top_only = chosen.get("headers_top_only", False)
    replacement_outline = chosen.get("replacement_outline")
    group_widths = [group_width, group_width]
    if compact:
        if any(entry.get("mode", "replace") != ("remove" if i % 2 == 0 else "replace")
               for i, entry in enumerate(entries)):
            raise ValueError("Compact layout expects Remove left and Replace right")
        outer, group_gap = 0.0, 0.28
        remove_panel_count = 4 if removal_mask_panel else 3
        group_widths = [remove_panel_count * side + (remove_panel_count - 1) * image_gap,
                        group_width]
        width = sum(group_widths) + group_gap
    group_lefts = [outer, outer + group_widths[0] + group_gap]
    row_height = float(chosen.get("row_pitch_inches", 4.25))
    header_size, prompt_size = 21, 23
    fig = plt.figure(figsize=(width, height), dpi=dpi, facecolor="white")
    plt.rcParams.update({"font.family": "DejaVu Sans", "pdf.fonttype": 42,
                         "svg.fonttype": "none"})
    column_titles = chosen.get("column_titles")
    if column_titles:
        if len(column_titles) != 2:
            raise ValueError("Expected two column headings")
        for column, title in enumerate(column_titles):
            center = group_lefts[column] + group_widths[column] / 2
            fig.text(center / width, (height - 0.1) / height, title,
                     fontsize=27, fontweight="bold", ha="center", va="top")
    if chosen.get("center_line", False):
        line_x = (group_lefts[1] - group_gap / 2) / width
        fig.add_artist(Line2D([line_x, line_x], [0.025, 0.99],
                       transform=fig.transFigure, color="#999999", linewidth=0.8))
    records = []
    for index, entry in enumerate(entries):
        mode = entry.get("mode", "replace")
        if mode not in ("replace", "remove"):
            raise ValueError(f"Unsupported operation: {mode}")
        folder = (Path(entry["source_dir"]) if mode == "remove" else
                  Path(entry["source_root"]) / "pairs" / entry["pair_id"])
        metadata = json.loads((folder / "metadata.json").read_text())
        if mode == "replace":
            source = folder / "figure.png"
            panels, boxes = extract_panels(source)
            titles = ["Source (A)", "Donor (B)", "Reconstruction", "Edited result"]
            prompt = metadata["instruction"]
            provenance = dict(source_figure=str(source.resolve()),
                source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                overview_columns=[1, 2, 3, 5], crop_boxes=boxes,
                result_condition="visual_only", writer_rewrite=False)
        else:
            if metadata["mode"] != "remove":
                raise ValueError("Selected removal metadata has a different operation")
            paths = [folder / name for name in
                     ("original.png", "reconstruction_native.png", "edited_native.png")]
            titles = ["Source (A)", "Reconstruction", "Edited result"]
            if removal_mask_panel:
                # This saved panel already overlays the selected predicted owner mask.
                # Reuse it exactly; no inference or newly drawn mask is needed.
                paths.insert(1, folder / "panel_1.png")
                titles.insert(1, "Remove item")
            panels = []
            for path in paths:
                with Image.open(path) as image:
                    panels.append(image.convert("RGB"))
            prompt = f"Remove the {metadata['target_category']} from image A."
            provenance = dict(source_panels=[dict(path=str(path.resolve()),
                sha256=hashlib.sha256(path.read_bytes()).hexdigest()) for path in paths],
                result_condition="remove_with_predicted_patch_exclusion_and_writer_rewrite",
                writer_rewrite=True)
        row, column = divmod(index, 2)
        left = group_lefts[column]
        current_width = group_widths[column]
        image_top = height - (0.95 if column_titles else 0.55) - row * row_height
        panel_left = left + (current_width - len(panels) * side - (len(panels)-1) * image_gap) / 2
        for j, panel in enumerate(panels):
            x = panel_left + j * (side + image_gap)
            ax = fig.add_axes([x / width, (image_top - side) / height,
                               side / width, side / height])
            ax.imshow(panel, interpolation="none")
            if mode == "replace" and replacement_outline and j < 2:
                mask_path = folder / ("target_prediction_mask.png" if j == 0 else
                                      "donor_prediction_mask.png")
                with Image.open(mask_path) as mask_image:
                    mask = np.asarray(mask_image, dtype=bool)
                # Draw a crisp vector annotation from the saved prediction mask.
                # The overview tile and model result pixels remain unchanged.
                x_coords = (np.arange(mask.shape[1]) + 0.5) * panel.width / mask.shape[1] - 0.5
                y_coords = (np.arange(mask.shape[0]) + 0.5) * panel.height / mask.shape[0] - 0.5
                ax.contour(x_coords, y_coords, mask.astype(float), levels=[0.5],
                           colors=[replacement_outline.get("color", "#22b84a")],
                           linewidths=replacement_outline.get("linewidth_pt", 1.2))
            ax.set_axis_off()
            if not headers_top_only or row == 0:
                fig.text((x + side / 2) / width, (image_top + 0.13) / height,
                         titles[j], ha="center", va="bottom", fontsize=header_size)
        wrapped = wrap_prompt(prompt, fig, current_width - 0.15, prompt_size) if show_prompts else ""
        if not show_prompts:
            artists = []
        elif chosen.get("red_operation_verbs", False):
            artists = []
            renderer = fig.canvas.get_renderer()
            for line_index, line in enumerate(wrapped.split("\n")):
                verb, separator, remainder = line.partition(" ")
                parts = ([(verb, "#c72b2b", "bold"), (separator + remainder, "black", "normal")]
                         if line_index == 0 and verb in ("Remove", "Replace") else
                         [(line, "black", "normal")])
                widths = []
                for text, color, weight in parts:
                    probe = fig.text(0, 0, text, fontsize=prompt_size, fontweight=weight)
                    widths.append(probe.get_window_extent(renderer).width / dpi)
                    probe.remove()
                x = left + (current_width - sum(widths)) / 2
                y = image_top - side - 0.19 - line_index * prompt_size / 72 * 1.25
                for (text, color, weight), text_width in zip(parts, widths):
                    artists.append(fig.text(x / width, y / height, text,
                        fontsize=prompt_size, color=color, fontweight=weight,
                        ha="left", va="top"))
                    x += text_width
        else:
            artists = [fig.text((left + current_width / 2) / width,
                          (image_top - side - 0.19) / height, wrapped,
                          fontsize=prompt_size, ha="center", va="top",
                          linespacing=1.25)]
        records.append(dict(entry, grid_position=[row + 1, column + 1],
            prompt=prompt, rendered_prompt=wrapped, mode=mode,
            source_metadata=metadata, panel_count=len(panels),
            **provenance, _artists=artists))
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for record in records:
        for artist in record.pop("_artists"):
            bbox = artist.get_window_extent(renderer)
            if bbox.x0 < 0 or bbox.y0 < 0 or bbox.x1 > width*dpi or bbox.y1 > height*dpi:
                raise ValueError("Prompt extends beyond the canvas")
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    stem = out / chosen.get("filename", "object_transfer_3x2")
    if stem.parent != out:
        raise ValueError("Filename must not include directories")
    fig.savefig(stem.with_suffix(".png"), dpi=dpi, facecolor="white")
    fig.savefig(stem.with_suffix(".pdf"), facecolor="white")
    fig.savefig(stem.with_suffix(".svg"), facecolor="white")
    fig.savefig(out / "preview.jpg", dpi=100, facecolor="white",
                pil_kwargs={"quality": 95})
    plt.close(fig)
    manifest = dict(layout=f"3 rows x 2 examples, row-major; remove:{4 if removal_mask_panel else 3} images / replace:4 images",
        composer="matplotlib (no AI image generation)", canvas_inches=[width, height],
        dpi=dpi, header_font_pt=header_size, prompt_font_pt=prompt_size,
        existing_image_tiles_reused=True, image_retouched=False,
        prompts_are_manual_operation_labels=True,
        freeform_instruction_following_claim=False, column_titles=column_titles,
        compact=compact, outer_margin_inches=outer, center_gap_inches=group_gap,
        removal_mask_panel=removal_mask_panel, show_prompts=show_prompts,
        headers_top_only=headers_top_only,
        replacement_outline=replacement_outline,
        row_pitch_inches=row_height,
        center_line=chosen.get("center_line", False),
        red_operation_verbs=chosen.get("red_operation_verbs", False),
        editing_protocols_differ_by_operation=any(r["mode"] == "remove" for r in records),
        examples=records)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(stem.with_suffix(".png"), flush=True)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--dpi", type=int, default=200)
    args = parser.parse_args()
    compose(args.selection, args.output_dir, args.dpi)


if __name__ == "__main__":
    main()
