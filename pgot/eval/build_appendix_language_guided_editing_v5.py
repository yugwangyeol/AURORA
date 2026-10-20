"""Render appendix language-guided editing v5 from visually audited outputs."""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, ImageDraw

from pgot.eval.build_appendix_qualitative_figures import (
    APPENDIX_ROOT,
    centered,
    font,
    save_figure,
    write_json,
)
from pgot.eval.build_appendix_qualitative_figures_v2 import (
    _instruction,
    _load_edit,
    outline_overlay,
)


def render(*, right_override=None, revision: str = "v5",
           output_stem: str = "language_guided_object_editing_v5",
           change_note: str | None = None) -> None:
    directory = APPENDIX_ROOT / "e4_language_guided_editing"
    root_v2 = directory / "_inference_v2"
    root_v4 = directory / "_inference_v4_candidates"
    root_v5 = directory / "_inference_v5_candidates"

    left = (
        (root_v2, "appendix_v2_remove_tv", "Remove", "the television from image A."),
        (root_v2, "appendix_v2_remove_dog", "Remove", "the dog from image A."),
        (root_v4, "appendix_v4_remove_bus", "Remove", "the bus from image A."),
    )
    right = right_override or (
        (root_v5, "appendix_v5_transfer_umbrella_blue_pink", "Transfer",
         "the pink pattern from B to A's umbrella."),
        (root_v5, "appendix_v5_transfer_jump_board_orange", "Transfer",
         "the orange pattern from B to A's snowboard."),
        (root_v2, "appendix_v2_transfer_suitcase", "Transfer",
         "B's pattern to A's suitcase."),
    )

    panel, gap, block_gap = 300, 6, 32
    header, instruction_h, row_gap, outer = 44, 42, 13, 10
    left_w, right_w = 3 * panel + 2 * gap, 4 * panel + 3 * gap
    width = outer * 2 + left_w + block_gap + right_w
    height = header + 3 * (panel + instruction_h) + 2 * row_gap + outer
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    x_left, x_right = outer, outer + left_w + block_gap

    for index, title in enumerate(("Source (A)", "Reconstruction", "Edited result")):
        centered(draw, title, (x_left + index * (panel + gap), 0, panel, header), font(20))
    for index, title in enumerate(("Source (A)", "Donor (B)", "Reconstruction", "Edited result")):
        centered(draw, title, (x_right + index * (panel + gap), 0, panel, header), font(20))
    divider = x_left + left_w + block_gap // 2
    draw.line((divider, 0, divider, height - outer), fill=(150, 150, 150), width=2)

    manifest = {
        "revision": revision,
        "model": "Reader-supervised DeOR-B",
        "selector": "Qwen3-VL-4B-Instruct (text-only)",
        "selector_image_input": False,
        "selector_gt_input": False,
        "manual_plan_correction": False,
        "layout": "three removal examples at left and three donor-guided edits at right",
        f"{revision}_change": change_note or (
            "Replaced right rows 1 and 2 after enlarged visual audit; the v4 keyboard retained "
            "residual keys and the clock was texture-mixed into the pizza."
        ),
        "examples": [],
    }

    for row, (root, edit_id, operation, rest) in enumerate(left):
        folder, metadata, plan = _load_edit(root, edit_id)
        y = header + row * (panel + instruction_h + row_gap)
        selected = outline_overlay(
            Image.open(folder / "source_A.png").convert("RGB"),
            folder / "target_prediction_mask.png",
        )
        images = (
            selected,
            Image.open(folder / "reconstruction_native.png").convert("RGB"),
            Image.open(folder / "edited_native.png").convert("RGB"),
        )
        for column, image in enumerate(images):
            canvas.paste(
                image.resize((panel, panel), Image.Resampling.LANCZOS),
                (x_left + column * (panel + gap), y),
            )
        _instruction(draw, x_left, y + panel + 7, left_w, operation, rest)
        manifest["examples"].append({
            "side": "left",
            "id": edit_id,
            "request": metadata["query"]["request"],
            "root": str(root.resolve()),
            "selector_status": plan["status"],
            "actual_plan": plan["plan"],
            "visual_review": "pass",
        })

    for row, (root, edit_id, operation, rest) in enumerate(right):
        folder, metadata, plan = _load_edit(root, edit_id)
        y = header + row * (panel + instruction_h + row_gap)
        source = outline_overlay(
            Image.open(folder / "source_A.png").convert("RGB"),
            folder / "target_prediction_mask.png",
        )
        donor = outline_overlay(
            Image.open(folder / "donor_B.png").convert("RGB"),
            folder / "donor_prediction_mask.png",
        )
        images = (
            source,
            donor,
            Image.open(folder / "reconstruction_native.png").convert("RGB"),
            Image.open(folder / "edited_native.png").convert("RGB"),
        )
        for column, image in enumerate(images):
            canvas.paste(
                image.resize((panel, panel), Image.Resampling.LANCZOS),
                (x_right + column * (panel + gap), y),
            )
        _instruction(draw, x_right, y + panel + 7, right_w, operation, rest)
        manifest["examples"].append({
            "side": "right",
            "id": edit_id,
            "request": metadata["query"]["request"],
            "root": str(root.resolve()),
            "selector_status": plan["status"],
            "actual_plan": plan["plan"],
            "visual_review": "pass",
            "replacement_mode": metadata.get("replacement_mode", metadata.get("execution_mode")),
        })

    save_figure(canvas, directory, output_stem)
    write_json(directory / f"manifest_{revision}.json", manifest)


if __name__ == "__main__":
    render()
