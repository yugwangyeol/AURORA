"""Build the coordinated three-figure object-editing revision.

Outputs:
1. Language-selected 2x2 figure without the cup and board cases.
2. No-Reader 2x2 figure without the same two cases.
3. Seven-row appendix figure with the removed language-selected cup/board
   examples inserted between the original third and fourth rows.

All panels are reloaded from their recorded native artifacts. No generated
image pixels are retouched.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from pgot.eval.build_appendix_qualitative_figures import APPENDIX_ROOT, centered, font
from pgot.eval.build_appendix_qualitative_figures_v2 import _load_edit, outline_overlay
from pgot.eval.rebuild_woreader_paper_figures import (
    REMOVE_OUT,
    SOURCE,
    TRANSFER_OUT,
    green_outline,
    green_overlay,
    read_json,
)
from pgot.eval.render_language_object_editing import REMOVE_MASK_ALPHA, outlined


PROJECT = Path(__file__).resolve().parents[2]
LANGUAGE_ROOT = PROJECT / "outputs/language_object_editing_transfer_replace_gpu0"
OUTPUT_DIR = PROJECT / "outputs/paper_figures/object_editing_rearranged_v1"

PANEL = 360
GAP = 7
BLOCK_GAP = 38
OUTER = 12
HEADER_H = 100
INSTRUCTION_H = 78
ROW_GAP = 16
LEFT_W = 3 * PANEL + 2 * GAP
RIGHT_W = 4 * PANEL + 3 * GAP
WIDTH = OUTER * 2 + LEFT_W + BLOCK_GAP + RIGHT_W
X_LEFT = OUTER
X_RIGHT = OUTER + LEFT_W + BLOCK_GAP
DIVIDER_X = X_LEFT + LEFT_W + BLOCK_GAP // 2


def split_instruction(request: str) -> tuple[str, str]:
    operation, rest = request.split(" ", 1)
    return operation, rest


def draw_instruction(
    draw: ImageDraw.ImageDraw,
    x: int,
    y: int,
    width: int,
    request: str,
) -> None:
    """Draw one large, single-line command, shrinking only when necessary."""
    operation, rest = split_instruction(request)
    # Keep instructions on one line, including the longer source-figure wording.
    for size in range(48, 23, -1):
        red_font = font(size, True)
        black_font = font(size)
        operation_text = f"{operation} "
        red_width = draw.textlength(operation_text, font=red_font)
        black_width = draw.textlength(rest, font=black_font)
        if red_width + black_width <= width - 16:
            break
    total_width = red_width + black_width
    red_box = draw.textbbox((0, 0), operation_text, font=red_font)
    black_box = draw.textbbox((0, 0), rest, font=black_font)
    text_height = max(red_box[3] - red_box[1], black_box[3] - black_box[1])
    start_x = x + (width - total_width) / 2
    start_y = y + (INSTRUCTION_H - text_height) / 2 - min(red_box[1], black_box[1])
    draw.text((start_x, start_y), operation_text, font=red_font, fill=(195, 25, 25))
    draw.text((start_x + red_width, start_y), rest, font=black_font, fill="black")


def draw_headers(canvas: Image.Image) -> None:
    draw = ImageDraw.Draw(canvas)
    for index, title in enumerate(("Source A", "Recon.", "Edited")):
        centered(
            draw,
            title,
            (X_LEFT + index * (PANEL + GAP), OUTER, PANEL, HEADER_H),
            font(56, True),
        )
    for index, title in enumerate(("Source A", "Donor B", "Recon.", "Edited")):
        centered(
            draw,
            title,
            (X_RIGHT + index * (PANEL + GAP), OUTER, PANEL, HEADER_H),
            font(56, True),
        )


def paste_row(
    canvas: Image.Image,
    row_index: int,
    left_images: tuple[Image.Image, ...],
    left_request: str,
    right_images: tuple[Image.Image, ...],
    right_request: str,
) -> None:
    if len(left_images) != 3 or len(right_images) != 4:
        raise ValueError("Expected three left panels and four right panels")
    draw = ImageDraw.Draw(canvas)
    y = OUTER + HEADER_H + row_index * (PANEL + INSTRUCTION_H + ROW_GAP)
    for column, image in enumerate(left_images):
        canvas.paste(
            image.resize((PANEL, PANEL), Image.Resampling.LANCZOS),
            (X_LEFT + column * (PANEL + GAP), y),
        )
    for column, image in enumerate(right_images):
        canvas.paste(
            image.resize((PANEL, PANEL), Image.Resampling.LANCZOS),
            (X_RIGHT + column * (PANEL + GAP), y),
        )
    draw_instruction(draw, X_LEFT, y + PANEL, LEFT_W, left_request)
    draw_instruction(draw, X_RIGHT, y + PANEL, RIGHT_W, right_request)


def blank_figure(rows: int) -> Image.Image:
    height = OUTER * 2 + HEADER_H + rows * (PANEL + INSTRUCTION_H) + (rows - 1) * ROW_GAP
    canvas = Image.new("RGB", (WIDTH, height), "white")
    draw_headers(canvas)
    ImageDraw.Draw(canvas).line(
        (DIVIDER_X, OUTER, DIVIDER_X, height - OUTER),
        fill=(145, 145, 145),
        width=2,
    )
    return canvas


def save_png_and_pdf(canvas: Image.Image, destination: Path) -> None:
    canvas.save(destination, dpi=(300, 300))
    canvas.save(destination.with_suffix(".pdf"), resolution=300)


def language_row(
    edit_id: str,
    *,
    edited_name: str = "edited_native.png",
    display_request: str | None = None,
):
    folder = LANGUAGE_ROOT / "edits" / edit_id
    metadata = read_json(folder / "metadata.json")
    request = metadata["query"]["request"]
    operation = metadata["executed"]["operation"]
    source = outlined(
        folder / "source_A.png",
        folder / "target_prediction_mask.png",
        fill_alpha=REMOVE_MASK_ALPHA if operation == "remove" else 0.0,
    )
    images = [source]
    if operation in ("transfer", "replace"):
        images.append(outlined(folder / "donor_B.png", folder / "donor_prediction_mask.png"))
    images.extend(
        (
            Image.open(folder / "reconstruction_native.png").convert("RGB"),
            Image.open(folder / edited_name).convert("RGB"),
        )
    )
    return tuple(images), display_request or request


def woreader_remove_row(task_id: str, image_id: str, request: str):
    folder = REMOVE_OUT / "candidates/remove" / task_id
    metadata = read_json(folder / "metadata.json")
    source = Image.open(folder / "original.png").convert("RGB")
    prediction = np.load(SOURCE / "cache/coco" / image_id / "ours.npz")["prediction"]
    selected = green_overlay(source, prediction == metadata["target_prediction_label"])
    return (
        selected,
        Image.open(folder / "reconstruction_native.png").convert("RGB"),
        Image.open(folder / "edited_native.png").convert("RGB"),
    ), request


def woreader_replace_row(pair_id: str, request: str):
    folder = TRANSFER_OUT / "pairs" / pair_id
    source = Image.open(folder / "source_A.png").convert("RGB")
    donor = Image.open(folder / "donor_B.png").convert("RGB")
    target_mask = np.asarray(Image.open(folder / "target_prediction_mask.png"), dtype=bool)
    donor_mask = np.asarray(Image.open(folder / "donor_prediction_mask.png"), dtype=bool)
    return (
        green_outline(source, target_mask),
        green_outline(donor, donor_mask),
        Image.open(folder / "reconstruction_native.png").convert("RGB"),
        Image.open(folder / "visual_only_native.png").convert("RGB"),
    ), request


def appendix_left_row(root: Path, edit_id: str, request: str):
    folder, _metadata, _plan = _load_edit(root, edit_id)
    return (
        outline_overlay(
            Image.open(folder / "source_A.png").convert("RGB"),
            folder / "target_prediction_mask.png",
        ),
        Image.open(folder / "reconstruction_native.png").convert("RGB"),
        Image.open(folder / "edited_native.png").convert("RGB"),
    ), request


def appendix_right_row(root: Path, edit_id: str, request: str):
    folder, _metadata, _plan = _load_edit(root, edit_id)
    return (
        outline_overlay(
            Image.open(folder / "source_A.png").convert("RGB"),
            folder / "target_prediction_mask.png",
        ),
        outline_overlay(
            Image.open(folder / "donor_B.png").convert("RGB"),
            folder / "donor_prediction_mask.png",
        ),
        Image.open(folder / "reconstruction_native.png").convert("RGB"),
        Image.open(folder / "edited_native.png").convert("RGB"),
    ), request


def build_language_selected_without_cup_board() -> Path:
    rows = (
        (
            language_row("edit_003", display_request="Remove A's bench."),
            language_row(
                "edit_002",
                display_request="Transfer the brown dog from image B to the dog in image A.",
            ),
        ),
        (
            language_row("edit_005", display_request="Remove A's giraffe."),
            language_row(
                "edit_006",
                display_request="Transfer the refrigerator from image B to the refrigerator in image A.",
            ),
        ),
    )
    canvas = blank_figure(len(rows))
    for index, (left, right) in enumerate(rows):
        paste_row(canvas, index, left[0], left[1], right[0], right[1])
    destination = OUTPUT_DIR / "object_editing_language_selected.png"
    save_png_and_pdf(canvas, destination)
    return destination


def build_woreader_without_cup_board() -> Path:
    rows = (
        (
            woreader_remove_row("remove_08_261161", "261161", "Remove A's bench."),
            woreader_replace_row(
                "dog_black_white_to_brown",
                "Transfer the brown dog from image B to the dog in image A.",
            ),
        ),
        (
            woreader_remove_row("remove_13_449909", "449909", "Remove A's giraffe."),
            woreader_replace_row(
                "refrigerator_white_to_steel",
                "Transfer the refrigerator from image B to the refrigerator in image A.",
            ),
        ),
    )
    canvas = blank_figure(len(rows))
    for index, (left, right) in enumerate(rows):
        paste_row(canvas, index, left[0], left[1], right[0], right[1])
    destination = OUTPUT_DIR / "object_editing_remove_replace_3x2_woreader.png"
    save_png_and_pdf(canvas, destination)
    return destination


def build_appendix_with_inserted_row() -> Path:
    directory = APPENDIX_ROOT / "e4_language_guided_editing"
    root_v2 = directory / "_inference_v2"
    root_v4 = directory / "_inference_v4_candidates"
    root_v7 = directory / "_inference_v7b_candidates"
    root_v8 = directory / "_inference_v8_cross_category_candidates"
    root_v9 = directory / "_inference_v9_remove_candidates"
    root_v9b = directory / "_inference_v9b_remove_candidates"

    standard_rows = (
        (
            appendix_left_row(root_v2, "appendix_v2_remove_tv", "Remove A's television."),
            appendix_right_row(root_v7, "appendix_v7b_plane_white_to_blue", "Transfer B's blue paint to A's airplane."),
        ),
        (
            appendix_left_row(root_v2, "appendix_v2_remove_dog", "Remove A's dog."),
            appendix_right_row(root_v7, "appendix_v7b_sheep_white_to_black", "Transfer B's black wool to A's sheep."),
        ),
        (
            appendix_left_row(root_v4, "appendix_v4_remove_bus", "Remove A's bus."),
            appendix_right_row(root_v2, "appendix_v2_transfer_suitcase", "Transfer B's pattern to A's suitcase."),
        ),
        # Inserted verbatim native examples removed from object_editing_language_selected.
        (
            language_row("edit_001", display_request="Remove A's cup."),
            language_row(
                "edit_004",
                edited_name="full_bundle_native.png",
                display_request="Replace A's snowboard with B's surfboard.",
            ),
        ),
        (
            appendix_left_row(root_v9, "appendix_v9_remove_bicycle", "Remove A's bicycle."),
            appendix_right_row(root_v8, "appendix_v8_replace_bicycle_motorcycle", "Replace A's bicycle with B's motorcycle."),
        ),
        (
            appendix_left_row(root_v9, "appendix_v9_remove_suitcase", "Remove A's suitcase."),
            appendix_right_row(root_v8, "appendix_v8_replace_cup_vase", "Replace A's cup with B's vase."),
        ),
        (
            appendix_left_row(root_v9b, "appendix_v9b_remove_clock", "Remove A's clock."),
            appendix_right_row(root_v8, "appendix_v8_replace_horse_cow", "Replace A's horse with B's cow."),
        ),
    )
    canvas = blank_figure(len(standard_rows))
    for index, (left, right) in enumerate(standard_rows):
        paste_row(canvas, index, left[0], left[1], right[0], right[1])
    destination = OUTPUT_DIR / "language_guided_object_editing_v11_single_header.png"
    save_png_and_pdf(canvas, destination)
    return destination


def main() -> None:
    Image.init()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    paths = (
        build_language_selected_without_cup_board(),
        build_woreader_without_cup_board(),
        build_appendix_with_inserted_row(),
    )
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
