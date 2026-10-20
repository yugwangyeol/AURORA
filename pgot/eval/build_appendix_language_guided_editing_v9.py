"""Render the six-row E4 figure combining v7 and v8 with fresh removals."""

from __future__ import annotations

from PIL import Image, ImageDraw

from pgot.eval.build_appendix_qualitative_figures import (
    APPENDIX_ROOT,
    centered,
    font,
    save_figure,
    write_json,
)
from pgot.eval.build_appendix_qualitative_figures_v2 import _load_edit, outline_overlay


def _instruction(draw: ImageDraw.ImageDraw, x: int, y: int, width: int,
                 operation: str, rest: str) -> None:
    red_font, black_font = font(25, True), font(24)
    op_text = f"{operation} "
    total = draw.textlength(op_text, font=red_font) + draw.textlength(rest, font=black_font)
    start = x + max(0, (width - total) / 2)
    draw.text((start, y), op_text, font=red_font, fill=(195, 25, 25))
    draw.text((start + draw.textlength(op_text, font=red_font), y), rest,
              font=black_font, fill="black")


def main(*, show_group_titles: bool = True, revision: str = "v9",
         output_stem: str = "language_guided_object_editing_v9_combined",
         unified_six_rows: bool = False,
         header_font_size: int = 24) -> None:
    directory = APPENDIX_ROOT / "e4_language_guided_editing"
    root_v2 = directory / "_inference_v2"
    root_v4 = directory / "_inference_v4_candidates"
    root_v7 = directory / "_inference_v7b_candidates"
    root_v8 = directory / "_inference_v8_cross_category_candidates"
    root_v9 = directory / "_inference_v9_remove_candidates"
    root_v9b = directory / "_inference_v9b_remove_candidates"

    blocks = (
        {
            "left_title": "Object removal",
            "right_title": "Appearance transfer",
            "left": (
                (root_v2, "appendix_v2_remove_tv", "Remove", "the television from image A."),
                (root_v2, "appendix_v2_remove_dog", "Remove", "the dog from image A."),
                (root_v4, "appendix_v4_remove_bus", "Remove", "the bus from image A."),
            ),
            "right": (
                (root_v7, "appendix_v7b_plane_white_to_blue", "Transfer",
                 "B's blue paint to A's airplane."),
                (root_v7, "appendix_v7b_sheep_white_to_black", "Transfer",
                 "B's black wool to A's sheep."),
                (root_v2, "appendix_v2_transfer_suitcase", "Transfer",
                 "B's pattern to A's suitcase."),
            ),
        },
        {
            "left_title": "Additional object removal",
            "right_title": "Cross-category replacement",
            "left": (
                (root_v9, "appendix_v9_remove_bicycle", "Remove",
                 "the bicycle from image A."),
                (root_v9, "appendix_v9_remove_suitcase", "Remove",
                 "the suitcase from image A."),
                (root_v9b, "appendix_v9b_remove_clock", "Remove",
                 "the clock from image A."),
            ),
            "right": (
                (root_v8, "appendix_v8_replace_bicycle_motorcycle", "Replace",
                 "A's bicycle with B's motorcycle."),
                (root_v8, "appendix_v8_replace_cup_vase", "Replace",
                 "A's cup with B's vase."),
                (root_v8, "appendix_v8_replace_horse_cow", "Replace",
                 "A's horse with B's cow."),
            ),
        },
    )

    panel, gap, block_gap, outer = 360, 7, 38, 12
    title_h = 44 if show_group_titles else 0
    header_h = 96 if unified_six_rows else 50
    instruction_h, row_gap, inter_block = 48, 16, 28
    left_w, right_w = 3 * panel + 2 * gap, 4 * panel + 3 * gap
    block_h = title_h + header_h + 3 * (panel + instruction_h) + 2 * row_gap
    width = outer * 2 + left_w + block_gap + right_w
    if unified_six_rows:
        height = outer * 2 + header_h + 6 * (panel + instruction_h) + 5 * row_gap
    else:
        height = outer * 2 + 2 * block_h + inter_block
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    x_left, x_right = outer, outer + left_w + block_gap
    divider_x = x_left + left_w + block_gap // 2

    manifest = {
        "revision": revision,
        "model": "Reader-supervised DeOR-B",
        "selector": "Qwen3-VL-4B-Instruct (text-only)",
        "selector_image_input": False,
        "selector_gt_input": False,
        "manual_plan_correction": False,
        "layout": (
            "One six-row table with one shared column-header row."
            if unified_six_rows else
            "Two vertically stacked three-row blocks with column headers only."
            if not show_group_titles else
            "Two vertically stacked three-row blocks. Top combines object removal with "
            "same-category appearance transfer; bottom combines three fresh removals with "
            "cross-category full-bundle replacement."
        ),
        "group_titles_displayed": show_group_titles,
        "unified_six_rows": unified_six_rows,
        "header_font_size": header_font_size,
        "panel_pixels": panel,
        "examples": [],
    }

    def draw_headers(block_top: int, left_title: str, right_title: str) -> int:
        if show_group_titles:
            centered(draw, left_title, (x_left, block_top, left_w, title_h), font(29, True))
            centered(draw, right_title, (x_right, block_top, right_w, title_h), font(29, True))
        header_y = block_top + title_h
        for index, title in enumerate(("Source (A)", "Reconstruction", "Edited result")):
            centered(draw, title, (x_left + index * (panel + gap), header_y,
                                   panel, header_h), font(header_font_size, True))
        for index, title in enumerate(("Source (A)", "Donor (B)",
                                       "Reconstruction", "Edited result")):
            centered(draw, title, (x_right + index * (panel + gap), header_y,
                                   panel, header_h), font(header_font_size, True))
        return header_y + header_h

    def draw_left_row(spec, y: int, block_index: int, row: int) -> None:
        root, edit_id, operation, rest = spec
        folder, metadata, plan = _load_edit(root, edit_id)
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
            canvas.paste(image.resize((panel, panel), Image.Resampling.LANCZOS),
                         (x_left + column * (panel + gap), y))
        _instruction(draw, x_left, y + panel + 8, left_w, operation, rest)
        manifest["examples"].append({
            "block": block_index + 1,
            "row": row + 1,
            "side": "left",
            "id": edit_id,
            "request": metadata["query"]["request"],
            "root": str(root.resolve()),
            "selector_status": plan["status"],
            "actual_plan": plan["plan"],
            "visual_review": "pass",
        })

    def draw_right_row(spec, y: int, block_index: int, row: int) -> None:
        root, edit_id, operation, rest = spec
        folder, metadata, plan = _load_edit(root, edit_id)
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
            canvas.paste(image.resize((panel, panel), Image.Resampling.LANCZOS),
                         (x_right + column * (panel + gap), y))
        _instruction(draw, x_right, y + panel + 8, right_w, operation, rest)
        manifest["examples"].append({
            "block": block_index + 1,
            "row": row + 1,
            "side": "right",
            "id": edit_id,
            "request": metadata["query"]["request"],
            "root": str(root.resolve()),
            "selector_status": plan["status"],
            "actual_plan": plan["plan"],
            "visual_review": "pass",
            "replacement_mode": metadata.get("replacement_mode", metadata.get("execution_mode")),
        })

    if unified_six_rows:
        first_row_y = draw_headers(outer, "", "")
        flat_row = 0
        for block_index, block in enumerate(blocks):
            for row in range(3):
                y = first_row_y + flat_row * (panel + instruction_h + row_gap)
                draw_left_row(block["left"][row], y, block_index, row)
                draw_right_row(block["right"][row], y, block_index, row)
                flat_row += 1
        draw.line((divider_x, outer, divider_x, height - outer),
                  fill=(145, 145, 145), width=2)
    else:
        for block_index, block in enumerate(blocks):
            block_top = outer + block_index * (block_h + inter_block)
            first_row_y = draw_headers(block_top, block["left_title"], block["right_title"])
            for row in range(3):
                y = first_row_y + row * (panel + instruction_h + row_gap)
                draw_left_row(block["left"][row], y, block_index, row)
                draw_right_row(block["right"][row], y, block_index, row)
            draw.line((divider_x, block_top, divider_x, block_top + block_h),
                      fill=(145, 145, 145), width=2)

        separator_y = outer + block_h + inter_block // 2
        draw.line((outer, separator_y, width - outer, separator_y),
                  fill=(110, 110, 110), width=3)

    save_figure(canvas, directory, output_stem)
    write_json(directory / f"manifest_{revision}.json", manifest)


if __name__ == "__main__":
    main()
