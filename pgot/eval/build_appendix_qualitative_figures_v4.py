"""Build the v4 appendix editing figures after targeted visual re-audit.

The existing v3 assets are preserved.  This revision only replaces the four
cells explicitly rejected during review; all panels remain direct model
outputs and language-guided plans remain text-only external-LLM selections.
"""

from __future__ import annotations

import argparse
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


def render_e4() -> None:
    root_v2 = APPENDIX_ROOT / "e4_language_guided_editing/_inference_v2"
    root_v3 = APPENDIX_ROOT / "e4_language_guided_editing/_inference_v3_candidates"
    root_v3b = APPENDIX_ROOT / "e4_language_guided_editing/_inference_v3b_candidates"
    root_v4 = APPENDIX_ROOT / "e4_language_guided_editing/_inference_v4_candidates"
    left = (
        (root_v2, "appendix_v2_remove_tv", "Remove", "the television from image A."),
        (root_v2, "appendix_v2_remove_dog", "Remove", "the dog from image A."),
        (root_v4, "appendix_v4_remove_bus", "Remove", "the bus from image A."),
    )
    right = (
        (root_v3b, "appendix_v3b_replace_keyboard_laptop", "Replace", "A's keyboard with B's laptop."),
        (root_v3, "appendix_v3_replace_pizza_clock", "Replace", "A's pizza with B's clock."),
        (root_v2, "appendix_v2_transfer_suitcase", "Transfer", "B's pattern to A's suitcase."),
    )
    panel, gap, block_gap, header, instruction_h, row_gap, outer = 300, 6, 32, 44, 42, 13, 10
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
    draw.line((x_left + left_w + block_gap // 2, 0,
               x_left + left_w + block_gap // 2, height - outer), fill=(150, 150, 150), width=2)

    manifest = {
        "revision": "v4",
        "model": "Reader-supervised DeOR-B",
        "selector": "Qwen3-VL-4B-Instruct (text-only)",
        "selector_image_input": False,
        "selector_gt_input": False,
        "manual_plan_correction": False,
        "layout": "three removal examples at left and three donor-guided edits at right",
        "v4_change": (
            "Replaced weak mouse removal, dog-to-cat replacement, and same-category pizza transfer "
            "after direct visual audit."
        ),
        "examples": [],
    }
    for row, (root, edit_id, operation, rest) in enumerate(left):
        folder, metadata, plan = _load_edit(root, edit_id)
        y = header + row * (panel + instruction_h + row_gap)
        selected = outline_overlay(Image.open(folder / "source_A.png").convert("RGB"),
                                   folder / "target_prediction_mask.png")
        images = (selected, Image.open(folder / "reconstruction_native.png").convert("RGB"),
                  Image.open(folder / "edited_native.png").convert("RGB"))
        for column, image in enumerate(images):
            canvas.paste(image.resize((panel, panel), Image.Resampling.LANCZOS),
                         (x_left + column * (panel + gap), y))
        _instruction(draw, x_left, y + panel + 7, left_w, operation, rest)
        manifest["examples"].append({
            "side": "left", "id": edit_id, "request": metadata["query"]["request"],
            "root": str(root.resolve()), "selector_status": plan["status"],
            "actual_plan": plan["plan"], "visual_review": "pass",
        })
    for row, (root, edit_id, operation, rest) in enumerate(right):
        folder, metadata, plan = _load_edit(root, edit_id)
        y = header + row * (panel + instruction_h + row_gap)
        source = outline_overlay(Image.open(folder / "source_A.png").convert("RGB"),
                                 folder / "target_prediction_mask.png")
        donor = outline_overlay(Image.open(folder / "donor_B.png").convert("RGB"),
                                folder / "donor_prediction_mask.png")
        images = (source, donor, Image.open(folder / "reconstruction_native.png").convert("RGB"),
                  Image.open(folder / "edited_native.png").convert("RGB"))
        for column, image in enumerate(images):
            canvas.paste(image.resize((panel, panel), Image.Resampling.LANCZOS),
                         (x_right + column * (panel + gap), y))
        _instruction(draw, x_right, y + panel + 7, right_w, operation, rest)
        manifest["examples"].append({
            "side": "right", "id": edit_id, "request": metadata["query"]["request"],
            "root": str(root.resolve()), "selector_status": plan["status"],
            "actual_plan": plan["plan"], "visual_review": "pass",
            "replacement_mode": metadata.get("replacement_mode", metadata.get("execution_mode")),
        })
    directory = APPENDIX_ROOT / "e4_language_guided_editing"
    save_figure(canvas, directory, "language_guided_object_editing_v4")
    write_json(directory / "manifest_v4.json", manifest)


def render_e5() -> None:
    examples = (
        ("Remove",
         APPENDIX_ROOT / "e5_direct_retry/candidates/remove/appendix_remove_00_503755",
         APPENDIX_ROOT.parents[1] / "figure_model_editing_memory8_register32_noprior/final/remove/07_138819"),
        ("Replace",
         APPENDIX_ROOT / "e5_direct_retry/candidates/replace/appendix_replace_05_312213",
         APPENDIX_ROOT / "e5_direct_manipulation/_inference_v2/candidates/replace/appendix_v2_replace_00_39480"),
        ("Add",
         APPENDIX_ROOT / "e5_direct_retry/candidates/add/appendix_add_09_54931",
         APPENDIX_ROOT / "e5_direct_manipulation/_inference_v2/candidates/add/appendix_v2_add_05_112997"),
        ("Background Transfer",
         APPENDIX_ROOT.parents[1] / "figure_model_editing_retry_add_background/final_native/background/01_568690",
         APPENDIX_ROOT / "e5_direct_manipulation/_inference_v3_candidates/candidates/background/appendix_v3_background_07_364166"),
    )
    # The original 21 px headings became roughly 5 pt at paper scale.  Use a
    # taller header and wrap the long middle label so all headings remain close
    # to the caption size after reduction.
    panel, gap, header, row_gap, left, block_gap, outer = 300, 6, 96, 22, 220, 28, 8
    block_w = 3 * panel + 2 * gap
    width = outer * 2 + left + 2 * block_w + block_gap
    height = header + 4 * panel + 3 * row_gap + outer
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    x_first, x_second = outer + left, outer + left + block_w + block_gap
    heading_font = font(36, True)
    for base in (x_first, x_second):
        centered(draw, "Source", (base, 0, panel, header), heading_font)
        centered(draw, "Selected", (base + panel + gap, 4, panel, header // 2), heading_font)
        centered(draw, "representation", (base + panel + gap, header // 2 - 4,
                                            panel, header // 2), heading_font)
        centered(draw, "Edited result", (base + 2 * (panel + gap), 0,
                                          panel, header), heading_font)
    draw.line((x_first + block_w + block_gap // 2, 0,
               x_first + block_w + block_gap // 2, height - outer), fill=(160, 160, 160), width=2)
    manifest = {
        "revision": "v4", "model": "Reader-supervised DeOR-B",
        "selection_input": "direct representation IDs; no natural-language selector",
        "layout": "four operation rows by two three-panel examples (six columns)",
        "v4_change": "Replaced the weak right Add result with a visually audited phone-scene plus cat result.",
        "operations": [],
    }
    for row, (operation, first, second) in enumerate(examples):
        y = header + row * (panel + row_gap)
        for base, folder in ((x_first, first), (x_second, second)):
            metadata = json.loads((folder / "metadata.json").read_text())
            for column, filename in enumerate(("panel_0.png", "panel_1.png", "panel_2.png")):
                image = Image.open(folder / filename).convert("RGB").resize((panel, panel), Image.Resampling.LANCZOS)
                canvas.paste(image, (base + column * (panel + gap), y))
            manifest["operations"].append({
                "display_name": operation, "metadata": metadata,
                "directory": str(folder.resolve()), "appendix_visual_review": "pass",
            })
        label = "Background\nTransfer" if operation == "Background Transfer" else operation
        lines = label.splitlines()
        line_height = 38
        top = y + (panel - len(lines) * line_height) // 2
        for line_index, line in enumerate(lines):
            centered(draw, line, (outer, top + line_index * line_height, left - 10, line_height),
                     font(28 if operation == "Background Transfer" else 30, True))
    directory = APPENDIX_ROOT / "e5_direct_manipulation"
    save_figure(canvas, directory, "direct_object_manipulation_v4")
    write_json(directory / "manifest_v4.json", manifest)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True, choices=("render_e4", "render_e5", "render_all"))
    args = parser.parse_args()
    if args.stage in ("render_e4", "render_all"):
        render_e4()
    if args.stage in ("render_e5", "render_all"):
        render_e5()


if __name__ == "__main__":
    main()
