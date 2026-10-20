"""Build the visually re-audited v3 appendix qualitative figures.

Only versioned ``*_v3`` assets are written.  Every displayed segmentation or
edit is an unchanged model output; visual matching only changes overlay colour.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from pgot.eval.build_appendix_qualitative_figures import (
    APPENDIX_ROOT,
    DEFAULT_MODEL,
    MODEL_COLUMNS,
    SEGMENTATION_ROOT,
    blend,
    cached_example,
    centered,
    first_sentence,
    font,
    label_colors,
    save_figure,
    wrap_pixels,
    write_json,
)
from pgot.eval.ar_binding_router import parse_ar_slots
from pgot.eval.build_appendix_qualitative_figures_v2 import (
    _instruction,
    _load_edit,
    outline_overlay,
)


E1_V3_SELECTIONS = {
    "1–3": ("248631", "402473"),
    # Both v2 cases were replaced after direct visual review of DeOR's masks.
    "4–6": ("125129", "32334"),
    "7–10": ("209613", "166478"),
    "11+": ("66926", "537506"),
}

E2_V3_FOLDERS = (
    APPENDIX_ROOT.parents[1] / "figure_deor_ar_grounding/appendix/01_coco_22935",
    APPENDIX_ROOT.parents[1] / "figure_deor_ar_grounding/appendix_v2_candidates_b/02_coco_147745",
    APPENDIX_ROOT.parents[1] / "figure_deor_ar_grounding/appendix/03_coco_481573",
    APPENDIX_ROOT.parents[1] / "figure_deor_ar_grounding/appendix_v2_candidates/03_coco_409630",
    APPENDIX_ROOT.parents[1] / "figure_deor_ar_grounding/appendix_v2_candidates/01_coco_112997",
    APPENDIX_ROOT.parents[1] / "figure_deor_ar_grounding/appendix/05_voc_2007_005845",
    APPENDIX_ROOT.parents[1] / "figure_deor_ar_grounding/appendix/06_voc_2007_007417",
    APPENDIX_ROOT.parents[1] / "figure_deor_ar_grounding/appendix_v2_candidates_b/01_coco_157807",
)


def render_e1(alpha: float = .64) -> None:
    # Match the effective paper-scale type size of the surrounding caption.
    panel, gap, header, band_gap, margin = 384, 6, 82, 24, 104
    names = ("Image", "GT") + tuple(name for _, name in MODEL_COLUMNS)
    width = margin + len(names) * panel + (len(names) - 1) * gap
    height = header + 8 * panel + 3 * band_gap
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    for column, name in enumerate(names):
        centered(draw, name, (margin + column * (panel + gap), 0, panel, header), font(48, True))

    manifest = {
        "revision": "v3",
        "dataset": "coco",
        "model": "Reader-supervised DeOR-B",
        "checkpoint": str(DEFAULT_MODEL),
        "selection": "GT-count-assisted qualitative selection; two visually audited cases per bin; not a benchmark",
        "v3_change": "Both 4–6 rows replaced because the v2 DeOR panels visibly omitted objects.",
        "prediction_labels_modified": False,
        "colour_matching": "Hungarian IoU matching for display colours only",
        "overlay_alpha": alpha,
        "examples": [],
    }
    row = 0
    band_positions = []
    for band_index, (count_range, sample_ids) in enumerate(E1_V3_SELECTIONS.items()):
        band_top = header + row * panel + band_index * band_gap
        band_positions.append((count_range, band_top))
        for sample_id in sample_ids:
            cache, scores = cached_example("coco", sample_id)
            data = np.load(cache / "gt.npz")
            gt, ignore = data["gt"], data["ignore"].astype(bool)
            original = np.asarray(Image.open(cache / "input.png").convert("RGB"))
            panels, details = [original], {}
            for model in ("gt",) + tuple(key for key, _ in MODEL_COLUMNS):
                source = cache / ("gt.npz" if model == "gt" else f"{model}.npz")
                labels = gt if model == "gt" else np.load(source)["prediction"]
                before = labels.copy()
                colours, mapping = label_colors(labels, gt, ignore, (245, 55, 104))
                panels.append(blend(original, colours, alpha))
                if not np.array_equal(labels, before):
                    raise AssertionError("Rendering modified a prediction")
                details[model] = {
                    "source": str(source.resolve()),
                    "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                    "display_colours": {str(k): list(v) for k, v in mapping.items()},
                }
            y = header + row * panel + band_index * band_gap
            for column, image in enumerate(panels):
                canvas.paste(Image.fromarray(image).resize((panel, panel), Image.Resampling.LANCZOS),
                             (margin + column * (panel + gap), y))
            manifest["examples"].append({
                "count_range": count_range, "id": sample_id,
                "scores": scores, "panels": details,
            })
            row += 1
    for count_range, band_top in band_positions:
        centered(draw, count_range, (0, band_top, margin - 8, panel * 2), font(36, True))
    directory = APPENDIX_ROOT / "e1_object_decomposition_coco"
    save_figure(canvas, directory, "object_decomposition_coco_v3")
    write_json(directory / "manifest_v3.json", manifest)


def render_e2() -> None:
    panel, panel_gap, tile_gap, outer, header = 300, 5, 30, 12, 150
    tile_w, tile_h = panel * 3 + panel_gap * 2, header + panel
    canvas = Image.new("RGB", (outer * 2 + tile_w * 2 + tile_gap,
                               outer * 2 + tile_h * 4 + tile_gap * 3), "white")
    draw = ImageDraw.Draw(canvas)
    heading_font, text_font = font(23, True), font(20)
    manifest = {
        "revision": "v3",
        "model": "Reader-supervised DeOR-B",
        "input": "image-only autoregressive",
        "external_text_query": False,
        "caption_display": "verbatim generated first sentence",
        "prediction_masks_modified": False,
        "semantic_review": "Every displayed first sentence was manually checked against both the image and its paired slot mask.",
        "v3_exclusions": [
            {"id": "voc:2007_009346", "reason": "The two generated cat coat descriptions were not reliably attributable to their paired masks."},
        ],
        "examples": [],
    }
    for index, folder in enumerate(E2_V3_FOLDERS):
        metadata = json.loads((folder / "metadata.json").read_text())
        chosen_slots = metadata["slots"][:2]
        x0 = outer + (index % 2) * (tile_w + tile_gap)
        y0 = outer + (index // 2) * (tile_h + tile_gap)
        paths = (folder / "image.png", folder / "object_01_overlay.png", folder / "object_02_overlay.png")
        record = {
            "dataset": metadata["dataset"], "id": metadata["id"],
            "descriptions": [], "sources": [], "semantic_review": "pass",
        }
        for column, path in enumerate(paths):
            x = x0 + column * (panel + panel_gap)
            if column == 0:
                # Bottom-align the label so it sits immediately above its image.
                label_h = 30
                centered(draw, "Image", (x, y0 + header - label_h - 6, panel, label_h), heading_font)
            else:
                slot = chosen_slots[column - 1]
                sentence = first_sentence(slot["description"])
                record["descriptions"].append({"category": slot["category"], "text": sentence})
                lines = wrap_pixels(draw, sentence, text_font, panel - 18)
                line_height = 25
                top = y0 + max(0, header - len(lines) * line_height - 6)
                for line_index, line in enumerate(lines):
                    draw.text((x + 9, top + line_index * line_height), line,
                              font=text_font, fill="black")
            image = Image.open(path).convert("RGB").resize((panel, panel), Image.Resampling.LANCZOS)
            canvas.paste(image, (x, y0 + header))
            record["sources"].append({
                "path": str(path.resolve()),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            })
        manifest["examples"].append(record)
    directory = APPENDIX_ROOT / "e2_description_region_alignment"
    save_figure(canvas, directory, "description_region_alignment_v3")
    write_json(directory / "manifest_v3.json", manifest)


def prepare_e5_candidates() -> None:
    """Prepare bounded fresh Add/Background candidates for the two weak v2 cells."""
    output = APPENDIX_ROOT / "e5_direct_manipulation/_inference_v3_candidates"
    candidates = json.loads((SEGMENTATION_ROOT / "candidates.json").read_text())["entries"]
    by_id = {str(entry["id"]): entry for entry in candidates if entry["dataset"] == "coco"}
    specs = (
        ("add", "125806", "cow", "22589", "sheep"),
        ("add", "263860", "elephant", "25139", "zebra"),
        ("add", "508602", "car", "579321", "dog"),
        ("add", "364166", "zebra", "210708", "elephant"),
        ("add", "560266", "bear", "22589", "sheep"),
        ("background", "125806", "cow", "22589", "sheep"),
        ("background", "263860", "elephant", "93965", "zebra"),
        ("background", "364166", "zebra", "210708", "elephant"),
        ("background", "508602", "car", "530099", "cat"),
    )

    def slot(image_id: str, category: str):
        entry = by_id[image_id]
        slots = parse_ar_slots(entry["ar_caption"]["text"], entry["ar_caption"]["ovt_count"])
        matches = [value for value in slots if value["normalized_category"] == category]
        if not matches:
            raise ValueError(f"Missing {category!r} in COCO {image_id}")
        return matches[0]

    tasks, entries = [], {}
    for index, (mode, target_id, target_category, donor_id, donor_category) in enumerate(specs):
        target, donor = slot(target_id, target_category), slot(donor_id, donor_category)
        entries[target_id], entries[donor_id] = by_id[target_id], by_id[donor_id]
        tasks.append({
            "task_id": f"appendix_v3_{mode}_{index:02d}_{target_id}",
            "mode": mode,
            "target_id": target_id,
            "target_slot": target["slot_index"],
            "target_category": target_category,
            "target_prediction_label": target["prediction_label"],
            "donor_id": donor_id,
            "donor_slot": donor["slot_index"],
            "donor_category": donor_category,
            "donor_prediction_label": donor["prediction_label"],
            "placement": "donor native location; no text-controlled repositioning",
        })
    write_json(output / "tasks.json", {
        "checkpoint": str(DEFAULT_MODEL), "seed": 1234, "steps": 10,
        "entries": entries, "tasks": tasks,
        "selection": "bounded fresh scene-compatible candidates for replacing the weak right Add and Background Transfer cells",
    })
    print(output / "tasks.json")


def render_e4() -> None:
    root_v2 = APPENDIX_ROOT / "e4_language_guided_editing/_inference_v2"
    root_v3 = APPENDIX_ROOT / "e4_language_guided_editing/_inference_v3_candidates"
    root_v3b = APPENDIX_ROOT / "e4_language_guided_editing/_inference_v3b_candidates"
    root_raw = APPENDIX_ROOT / "e4_language_guided_raw"
    left = (
        (root_v2, "appendix_v2_remove_tv", "Remove", "the television from image A."),
        (root_v2, "appendix_v2_remove_dog", "Remove", "the dog from image A."),
        (root_v3, "appendix_v3_remove_mouse", "Remove", "the computer mouse from image A."),
    )
    # Row one is resolved after the fresh v3b candidates are visually audited.
    right = (
        (root_v3b, "appendix_v3b_replace_dog_cat", "Replace", "A's dog with B's cat."),
        (root_raw, "appendix_edit_003", "Transfer", "B's appearance to A's pizza."),
        (root_v2, "appendix_v2_transfer_suitcase", "Transfer", "B's pattern to A's suitcase."),
    )
    panel, gap, block_gap, header, instruction_h, row_gap, outer = 300, 6, 32, 44, 42, 13, 10
    left_w, right_w = 3 * panel + 2 * gap, 4 * panel + 3 * gap
    width = outer * 2 + left_w + block_gap + right_w
    height = header + 3 * (panel + instruction_h) + 2 * row_gap + outer
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    x_left, x_right = outer, outer + left_w + block_gap
    for i, title in enumerate(("Source (A)", "Reconstruction", "Edited result")):
        centered(draw, title, (x_left + i * (panel + gap), 0, panel, header), font(20))
    for i, title in enumerate(("Source (A)", "Donor (B)", "Reconstruction", "Edited result")):
        centered(draw, title, (x_right + i * (panel + gap), 0, panel, header), font(20))
    draw.line((x_left + left_w + block_gap // 2, 0,
               x_left + left_w + block_gap // 2, height - outer), fill=(150, 150, 150), width=2)

    manifest = {
        "revision": "v3", "model": "Reader-supervised DeOR-B",
        "selector": "Qwen3-VL-4B-Instruct (text-only)",
        "selector_image_input": False, "selector_gt_input": False,
        "manual_plan_correction": False,
        "layout": "reference-style: three remove examples at left and three donor-guided edits at right",
        "visual_selection": "weak v2 rows replaced after direct review of newly generated fixed-seed candidates",
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
            "replacement_mode": metadata.get("replacement_mode", metadata.get("executed", {}).get("replacement_mode")),
        })
    directory = APPENDIX_ROOT / "e4_language_guided_editing"
    save_figure(canvas, directory, "language_guided_object_editing_v3")
    write_json(directory / "manifest_v3.json", manifest)


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
         APPENDIX_ROOT / "e5_direct_manipulation/_inference_v3_candidates/candidates/add/appendix_v3_add_01_263860"),
        ("Background Transfer",
         APPENDIX_ROOT.parents[1] / "figure_model_editing_retry_add_background/final_native/background/01_568690",
         APPENDIX_ROOT / "e5_direct_manipulation/_inference_v3_candidates/candidates/background/appendix_v3_background_07_364166"),
    )
    panel, gap, header, row_gap, left, block_gap, outer = 300, 6, 52, 22, 142, 28, 8
    block_w = 3 * panel + 2 * gap
    width = outer * 2 + left + 2 * block_w + block_gap
    height = header + 4 * panel + 3 * row_gap + outer
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    x_first, x_second = outer + left, outer + left + block_w + block_gap
    for base in (x_first, x_second):
        for column, title in enumerate(("Source", "Selected representation", "Edited result")):
            centered(draw, title, (base + column * (panel + gap), 0, panel, header), font(21, True))
    draw.line((x_first + block_w + block_gap // 2, 0,
               x_first + block_w + block_gap // 2, height - outer), fill=(160, 160, 160), width=2)
    manifest = {
        "revision": "v3", "model": "Reader-supervised DeOR-B",
        "selection_input": "direct representation IDs; no natural-language selector",
        "layout": "four operation rows by two three-panel examples (six columns)",
        "v3_change": "right Add and right Background Transfer were replaced with fresh visually audited outputs",
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
        line_height = 27
        top = y + (panel - len(lines) * line_height) // 2
        for line_index, line in enumerate(lines):
            centered(draw, line, (outer, top + line_index * line_height, left - 10, line_height),
                     font(18 if operation == "Background Transfer" else 20, True))
    directory = APPENDIX_ROOT / "e5_direct_manipulation"
    save_figure(canvas, directory, "direct_object_manipulation_v3")
    write_json(directory / "manifest_v3.json", manifest)


def audit_v3() -> None:
    figures = {
        "E1": {f"coco:{sample_id}" for group in E1_V3_SELECTIONS.values() for sample_id in group},
        "E2": {
            "coco:22935", "coco:147745", "coco:481573", "coco:409630",
            "coco:112997", "coco:157807", "voc:2007_005845", "voc:2007_007417",
        },
        "E4": {f"coco:{image_id}" for image_id in (
            357941, 170893, 68765, 579321, 555705, 84431, 554735, 112626, 350019)},
        "E5": {f"coco:{image_id}" for image_id in (
            503755, 138819, 312213, 324715, 39480, 227482, 54931, 217425,
            263860, 25139, 568690, 176778, 364166, 210708)},
    }
    audited_main = {
        *(f"coco:{image_id}" for image_id in (
            552371, 559956, 90208, 450686, 339823, 323151, 52891, 205834,
            261161, 193429, 554595, 449909, 802, 243344)),
        "voc:2007_005149", "voc:2007_007795", "voc:2010_004165",
    }
    owners, appendix_conflicts = {}, []
    for figure, cases in figures.items():
        for case in sorted(cases):
            if case in owners:
                appendix_conflicts.append({"case": case, "figures": [owners[case], figure]})
            owners[case] = figure
    main_conflicts = [{"case": case, "appendix_figure": owners[case]}
                      for case in sorted(audited_main & set(owners))]

    alignment_checks = []
    for folder in E2_V3_FOLDERS:
        metadata = json.loads((folder / "metadata.json").read_text())
        masks = np.load(folder / "masks.npz")
        for index, slot in enumerate(metadata["slots"][:2], 1):
            mask = masks[f"object_{index:02d}"]
            if not mask.any():
                raise RuntimeError(f"Empty displayed E2 mask: {folder} object {index}")
            alignment_checks.append({
                "case": f'{metadata["dataset"]}:{metadata["id"]}',
                "object": index, "category": slot["category"],
                "first_sentence": first_sentence(slot["description"]),
                "mask_pixels": int(mask.sum()), "manual_semantic_review": "pass",
            })

    four_to_six = []
    for sample_id in E1_V3_SELECTIONS["4–6"]:
        _, scores = cached_example("coco", sample_id)
        ours = scores["ours"]
        if ours["merge_count"] != 0:
            raise RuntimeError(f"E1 4–6 case {sample_id} contains a detected merge")
        four_to_six.append({"id": sample_id, "ours": ours})

    required = (
        APPENDIX_ROOT / "e1_object_decomposition_coco/object_decomposition_coco_v3.png",
        APPENDIX_ROOT / "e2_description_region_alignment/description_region_alignment_v3.png",
        APPENDIX_ROOT / "e4_language_guided_editing/language_guided_object_editing_v3.png",
        APPENDIX_ROOT / "e5_direct_manipulation/direct_object_manipulation_v3.png",
    )
    missing = [str(path) for path in required if not path.exists()]
    report = {
        "revision": "v3", "passed": not appendix_conflicts and not main_conflicts and not missing,
        "policy": "No displayed source/donor image is reused across E1/E2/E4/E5 or the audited main-paper figures.",
        "figures": {key: sorted(value) for key, value in figures.items()},
        "appendix_conflicts": appendix_conflicts, "main_figure_conflicts": main_conflicts,
        "missing_outputs": missing, "e1_4_to_6_checks": four_to_six,
        "e2_description_mask_checks": alignment_checks,
    }
    write_json(APPENDIX_ROOT / "overlap_and_semantic_audit_v3.json", report)
    if not report["passed"]:
        raise RuntimeError(f"v3 audit failed: {report}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True,
                        choices=("render_e1", "render_e2", "render_e4", "render_e5",
                                 "render_static", "prepare_e5", "audit"))
    args = parser.parse_args()
    if args.stage in ("render_e1", "render_static"):
        render_e1()
    if args.stage in ("render_e2", "render_static"):
        render_e2()
    if args.stage == "prepare_e5":
        prepare_e5_candidates()
    if args.stage == "render_e4":
        render_e4()
    if args.stage == "render_e5":
        render_e5()
    if args.stage == "audit":
        audit_v3()


if __name__ == "__main__":
    main()
