"""Audited v2 revisions of the appendix qualitative figures.

The script only writes ``*_v2`` assets into the existing v1 figure folders.
E.3 is intentionally unsupported because it was explicitly withdrawn.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from pgot.eval.ar_binding_router import parse_ar_slots
from pgot.eval.build_appendix_qualitative_figures import (
    APPENDIX_ROOT,
    DEFAULT_MODEL,
    MODEL_COLUMNS,
    PROJECT,
    SEGMENTATION_ROOT,
    blend,
    cached_example,
    centered,
    first_sentence,
    font,
    label_colors,
    read_jsonl,
    save_figure,
    wrap_pixels,
    write_json,
)


E1_V2_SELECTIONS = {
    "coco": {
        "1–3": ("248631", "402473"),
        "4–6": ("37740", "568710"),
        "7–10": ("209613", "166478"),
        "11+": ("66926", "537506"),
    },
}

E2_V2_FOLDERS = (
    PROJECT / "outputs/figure_deor_ar_grounding/appendix/01_coco_22935",
    PROJECT / "outputs/figure_deor_ar_grounding/appendix_v2_candidates_b/02_coco_147745",
    PROJECT / "outputs/figure_deor_ar_grounding/appendix/03_coco_481573",
    PROJECT / "outputs/figure_deor_ar_grounding/appendix_v2_candidates/03_coco_409630",
    PROJECT / "outputs/figure_deor_ar_grounding/appendix/04_voc_2007_009346",
    PROJECT / "outputs/figure_deor_ar_grounding/appendix/05_voc_2007_005845",
    PROJECT / "outputs/figure_deor_ar_grounding/appendix/06_voc_2007_007417",
    PROJECT / "outputs/figure_deor_ar_grounding/appendix_v2_candidates_b/01_coco_157807",
)


def render_e1_v2(alpha: float = .64) -> None:
    dataset = "coco"
    panel, gap, header, band_gap, margin = 384, 6, 62, 24, 86
    names = ("Image", "GT") + tuple(name for _, name in MODEL_COLUMNS)
    width = margin + len(names) * panel + (len(names) - 1) * gap
    height = header + 8 * panel + 3 * band_gap
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    for column, name in enumerate(names):
        centered(draw, name, (margin + column * (panel + gap), 0, panel, header), font(31, True))

    manifest = {
        "revision": "v2",
        "dataset": dataset,
        "model": "Reader-supervised DeOR-B",
        "checkpoint": str(DEFAULT_MODEL),
        "requested_replacements": {
            "4–6 second row": {"old": "160556", "new": "568710"},
            "7–10 first row": {"old": "86582", "new": "209613"},
        },
        "selection": "GT-count-assisted qualitative selection; two cases per bin; not a benchmark",
        "prediction_labels_modified": False,
        "colour_matching": "Hungarian IoU matching for display colours only",
        "overlay_alpha": alpha,
        "examples": [],
    }
    row = 0
    band_positions = []
    for band_index, (count_range, sample_ids) in enumerate(E1_V2_SELECTIONS[dataset].items()):
        band_top = header + row * panel + band_index * band_gap
        band_positions.append((count_range, band_top))
        for sample_id in sample_ids:
            cache, scores = cached_example(dataset, sample_id)
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
        centered(draw, count_range, (0, band_top, margin - 8, panel * 2), font(25, True))
    directory = APPENDIX_ROOT / "e1_object_decomposition_coco"
    save_figure(canvas, directory, "object_decomposition_coco_v2")
    write_json(directory / "manifest_v2.json", manifest)


def render_e2_v2() -> None:
    panel, panel_gap, tile_gap, outer, header = 300, 5, 30, 12, 150
    tile_w, tile_h = panel * 3 + panel_gap * 2, header + panel
    canvas = Image.new("RGB", (outer * 2 + tile_w * 2 + tile_gap,
                               outer * 2 + tile_h * 4 + tile_gap * 3), "white")
    draw = ImageDraw.Draw(canvas)
    heading_font, text_font = font(23, True), font(20)
    manifest = {
        "revision": "v2",
        "model": "Reader-supervised DeOR-B",
        "input": "image-only autoregressive",
        "external_text_query": False,
        "caption_display": "verbatim generated first sentence",
        "prediction_masks_modified": False,
        "semantic_review": "All displayed descriptions were manually checked against the image and mask.",
        "excluded_case": {
            "id": "coco:454750",
            "reason": "Removed because the generated age claim ('younger zebra') is not visually justified; apparent size can be caused by perspective.",
        },
        "examples": [],
    }
    for index, folder in enumerate(E2_V2_FOLDERS):
        metadata = json.loads((folder / "metadata.json").read_text())
        chosen_slots = metadata["slots"][:2]
        x0 = outer + (index % 2) * (tile_w + tile_gap)
        y0 = outer + (index // 2) * (tile_h + tile_gap)
        paths = (folder / "image.png", folder / "object_01_overlay.png", folder / "object_02_overlay.png")
        record = {"dataset": metadata["dataset"], "id": metadata["id"],
                  "descriptions": [], "sources": [], "semantic_review": "pass"}
        for column, path in enumerate(paths):
            x = x0 + column * (panel + panel_gap)
            if column == 0:
                centered(draw, "Image", (x, y0, panel, header), heading_font)
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
            record["sources"].append({"path": str(path.resolve()),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
        manifest["examples"].append(record)
    directory = APPENDIX_ROOT / "e2_description_region_alignment"
    save_figure(canvas, directory, "description_region_alignment_v2")
    write_json(directory / "manifest_v2.json", manifest)


def outline_overlay(image: Image.Image, mask_path: Path, colour=(28, 205, 76), width=5) -> Image.Image:
    mask_image = Image.open(mask_path).convert("L").resize(image.size, Image.Resampling.NEAREST)
    mask = mask_image.point(lambda value: 255 if value else 0)
    dilated = mask.filter(ImageFilter.MaxFilter(width * 2 + 1))
    eroded = mask.filter(ImageFilter.MinFilter(width * 2 + 1))
    edge = np.asarray(dilated, dtype=np.int16) - np.asarray(eroded, dtype=np.int16)
    rgb = np.asarray(image.convert("RGB")).copy()
    rgb[edge > 0] = np.asarray(colour, dtype=np.uint8)
    return Image.fromarray(rgb)


def _load_edit(root: Path, edit_id: str) -> tuple[Path, dict, dict]:
    folder = root / "edits" / edit_id
    metadata = json.loads((folder / "metadata.json").read_text())
    plan = json.loads((root / "plans" / f"{edit_id}.json").read_text())
    return folder, metadata, plan


def _instruction(draw: ImageDraw.ImageDraw, x: int, y: int, width: int, operation: str, rest: str) -> None:
    red_font, black_font = font(21, True), font(20)
    op_text = f"{operation} "
    total = draw.textlength(op_text, font=red_font) + draw.textlength(rest, font=black_font)
    start = x + max(0, (width - total) / 2)
    draw.text((start, y), op_text, font=red_font, fill=(195, 25, 25))
    draw.text((start + draw.textlength(op_text, font=red_font), y), rest,
              font=black_font, fill="black")


def render_e4_v2() -> None:
    root_v2 = APPENDIX_ROOT / "e4_language_guided_editing/_inference_v2"
    root_v2d = APPENDIX_ROOT / "e4_language_guided_editing/_inference_v2d"
    root_retry = APPENDIX_ROOT / "e4_language_guided_retry"
    left = (
        (root_v2, "appendix_v2_remove_tv", "Remove", "the television from image A."),
        (root_v2, "appendix_v2_remove_dog", "Remove", "the dog from image A."),
        (root_retry, "appendix_retry_004", "Remove", "the umbrella from image A."),
    )
    right = (
        (root_v2d, "appendix_v2_transfer_train_bus", "Transfer", "B's train appearance to A's bus."),
        (root_v2, "appendix_v2_replace_toilet_bench", "Replace", "A's toilet with B's wooden bench."),
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
        "revision": "v2", "model": "Reader-supervised DeOR-B",
        "selector": "Qwen3-VL-4B-Instruct (text-only)",
        "selector_image_input": False, "selector_gt_input": False,
        "manual_plan_correction": False,
        "layout": "reference-style: three remove examples at left and three donor-guided edits at right",
        "cross_category_operations": {
            "appearance_transfer": "visual-memory transfer from train to bus while retaining the bus semantic owner",
            "replacement": "full semantic+visual owner bundle replacement from bench to the toilet region",
        },
        "examples": [],
    }
    for row, spec in enumerate(left):
        root, edit_id, operation, rest = spec
        folder, metadata, plan = _load_edit(root, edit_id)
        y = header + row * (panel + instruction_h + row_gap)
        source = Image.open(folder / "source_A.png").convert("RGB")
        selected = outline_overlay(source, folder / "target_prediction_mask.png")
        images = (selected, Image.open(folder / "reconstruction_native.png").convert("RGB"),
                  Image.open(folder / "edited_native.png").convert("RGB"))
        for column, image in enumerate(images):
            canvas.paste(image.resize((panel, panel), Image.Resampling.LANCZOS),
                         (x_left + column * (panel + gap), y))
        _instruction(draw, x_left, y + panel + 7, left_w, operation, rest)
        manifest["examples"].append({"side": "left", "id": edit_id,
            "request": metadata["query"]["request"], "root": str(root.resolve()),
            "selector_status": plan["status"], "actual_plan": plan["plan"],
            "visual_review": "pass"})
    for row, spec in enumerate(right):
        root, edit_id, operation, rest = spec
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
        manifest["examples"].append({"side": "right", "id": edit_id,
            "request": metadata["query"]["request"], "root": str(root.resolve()),
            "selector_status": plan["status"], "actual_plan": plan["plan"],
            "visual_review": "pass",
            "replacement_mode": metadata.get("replacement_mode", metadata.get("executed", {}).get("replacement_mode"))})
    directory = APPENDIX_ROOT / "e4_language_guided_editing"
    save_figure(canvas, directory, "language_guided_object_editing_v2")
    write_json(directory / "manifest_v2.json", manifest)


def prepare_e5_v2() -> None:
    output = APPENDIX_ROOT / "e5_direct_manipulation/_inference_v2"
    candidates = json.loads((SEGMENTATION_ROOT / "candidates.json").read_text())["entries"]
    by_id = {str(entry["id"]): entry for entry in candidates if entry["dataset"] == "coco"}
    specs = (
        ("replace", "39480", "person", 0, "227482", "person", 0),
        ("add", "333956", "parking meter", 0, "442306", "umbrella", 0),
        ("add", "333956", "parking meter", 0, "555705", "cat", 0),
        ("add", "117744", "person", 0, "555705", "cat", 1),
        ("add", "98839", "tv", 0, "174018", "teddy bear", 0),
        ("add", "112997", "cell phone", 0, "555705", "cat", 0),
    )

    def slot(image_id: str, category: str, occurrence: int):
        entry = by_id[image_id]
        slots = parse_ar_slots(entry["ar_caption"]["text"], entry["ar_caption"]["ovt_count"])
        matches = [value for value in slots if value["normalized_category"] == category]
        if occurrence >= len(matches):
            raise ValueError(f"Missing occurrence {occurrence} of {category!r} in COCO {image_id}")
        return matches[occurrence]

    tasks, entries = [], {}
    for index, (mode, target_id, target_category, target_occurrence,
                donor_id, donor_category, donor_occurrence) in enumerate(specs):
        target = slot(target_id, target_category, target_occurrence)
        donor = slot(donor_id, donor_category, donor_occurrence)
        entries[target_id], entries[donor_id] = by_id[target_id], by_id[donor_id]
        tasks.append({
            "task_id": f"appendix_v2_{mode}_{index:02d}_{target_id}", "mode": mode,
            "target_id": target_id, "target_slot": target["slot_index"],
            "target_category": target_category, "target_prediction_label": target["prediction_label"],
            "donor_id": donor_id, "donor_slot": donor["slot_index"],
            "donor_category": donor_category, "donor_prediction_label": donor["prediction_label"],
            "placement": "donor native location; no text-controlled repositioning",
        })
    write_json(output / "tasks.json", {
        "checkpoint": str(DEFAULT_MODEL), "seed": 1234, "steps": 10,
        "entries": entries, "tasks": tasks,
        "selection": "bounded, fresh, non-overlapping v2 candidates for the second E.5 set",
    })
    print(output / "tasks.json")


def render_e5_v2() -> None:
    # The right-hand examples were generated specifically for v2; the remaining
    # strong examples are unchanged audited outputs from the v1 candidate pool.
    examples = (
        ("Remove",
         APPENDIX_ROOT / "e5_direct_retry/candidates/remove/appendix_remove_00_503755",
         PROJECT / "outputs/figure_model_editing_memory8_register32_noprior/final/remove/07_138819"),
        ("Replace",
         APPENDIX_ROOT / "e5_direct_retry/candidates/replace/appendix_replace_05_312213",
         APPENDIX_ROOT / "e5_direct_manipulation/_inference_v2/candidates/replace/appendix_v2_replace_00_39480"),
        ("Add",
         APPENDIX_ROOT / "e5_direct_retry/candidates/add/appendix_add_09_54931",
         APPENDIX_ROOT / "e5_direct_manipulation/_inference_v2/candidates/add/appendix_v2_add_05_112997"),
        ("Background Transfer",
         PROJECT / "outputs/figure_model_editing_retry_add_background/final_native/background/01_568690",
         PROJECT / "outputs/figure_model_editing_retry_add_background/final_native/background/02_554735"),
    )
    panel, gap, header, row_gap, left, block_gap, outer = 300, 6, 52, 22, 142, 28, 8
    block_w = 3 * panel + 2 * gap
    width = outer * 2 + left + 2 * block_w + block_gap
    height = header + 4 * panel + 3 * row_gap + outer
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    x_first, x_second = outer + left, outer + left + block_w + block_gap
    titles = ("Source", "Selected representation", "Edited result")
    for base in (x_first, x_second):
        for column, title in enumerate(titles):
            centered(draw, title, (base + column * (panel + gap), 0, panel, header), font(21, True))
    draw.line((x_first + block_w + block_gap // 2, 0,
               x_first + block_w + block_gap // 2, height - outer), fill=(160, 160, 160), width=2)
    manifest = {
        "revision": "v2", "model": "Reader-supervised DeOR-B",
        "selection_input": "direct representation IDs; no natural-language selector",
        "layout": "four operation rows by two three-panel examples (six columns)",
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
            centered(draw, line, (outer, top + line_index * line_height,
                                   left - 10, line_height),
                     font(18 if operation == "Background Transfer" else 20, True))
    directory = APPENDIX_ROOT / "e5_direct_manipulation"
    save_figure(canvas, directory, "direct_object_manipulation_v2")
    write_json(directory / "manifest_v2.json", manifest)


def audit_overlap_v2() -> None:
    figures = {
        "E1": {f"coco:{sample_id}" for group in E1_V2_SELECTIONS["coco"].values()
               for sample_id in group},
        "E2": {
            "coco:22935", "coco:147745", "coco:481573", "coco:409630",
            "coco:157807", "voc:2007_009346", "voc:2007_005845", "voc:2007_007417",
        },
        "E4": {f"coco:{image_id}" for image_id in
               (357941, 170893, 32861, 243867, 482585, 245764, 566436, 112626, 350019)},
        "E5": {f"coco:{image_id}" for image_id in
               (503755, 312213, 324715, 54931, 217425, 568690, 176778,
                138819, 39480, 227482, 112997, 555705, 554735, 84431)},
    }
    audited_main = {
        *(f"coco:{image_id}" for image_id in
          (552371, 559956, 90208, 450686, 339823, 323151, 52891, 205834,
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
    report = {
        "passed": not appendix_conflicts and not main_conflicts,
        "policy": "No displayed source/donor image is reused across revised figures or audited main-paper figures.",
        "e3_status": "withdrawn by user; not rebuilt and excluded from revised-figure audit",
        "figures": {key: sorted(value) for key, value in figures.items()},
        "audited_main_figure_cases": sorted(audited_main),
        "appendix_conflicts": appendix_conflicts,
        "main_figure_conflicts": main_conflicts,
    }
    write_json(APPENDIX_ROOT / "overlap_audit_v2.json", report)
    if not report["passed"]:
        raise RuntimeError(f"v2 case-overlap audit failed: {report}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True,
                        choices=("render_e1", "render_e2", "render_e4", "prepare_e5",
                                 "render_e5", "render_static", "audit"))
    args = parser.parse_args()
    if args.stage in ("render_e1", "render_static"):
        render_e1_v2()
    if args.stage in ("render_e2", "render_static"):
        render_e2_v2()
    if args.stage == "render_e4":
        render_e4_v2()
    if args.stage == "prepare_e5":
        prepare_e5_v2()
    if args.stage == "render_e5":
        render_e5_v2()
    if args.stage == "audit":
        audit_overlap_v2()


if __name__ == "__main__":
    main()
