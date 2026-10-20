"""Build the paper appendix qualitative figures from audited cached outputs.

This module keeps qualitative selection manifests explicit.  It never changes
model predictions: Hungarian matching is used only to align display colours.
The dense-scene preparation command creates a small, reproducible replay list
for object-count bins that were absent from the earlier qualitative cache.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from pgot.eval.render_segmentation_paper_figure import blend, label_colors
from pgot.eval.figure_ar_grounding_examples import first_sentence
from pgot.eval.ar_binding_router import parse_ar_slots

PROJECT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL = PROJECT / "checkpoints/pgot_oneshot_memory8_register32_noprior/checkpoint-5000"
FULL_EVAL = PROJECT / "outputs/segeval_pgot_memory8_full_coco_voc_ar"
APPENDIX_ROOT = PROJECT / "outputs/paper_figures/appendix_qualitative_v1"
SEGMENTATION_ROOT = PROJECT / "outputs/figure_segmentation_over_under"
DENSE_ROOT = APPENDIX_ROOT / "dense_scene_eval"

MODEL_COLUMNS = (("adaslot", "AdaSlot"), ("metaslot", "MetaSlot"),
                 ("coda", "CODA"), ("ours", "DeOR"))
E1_SELECTIONS = {
    "coco": {
        "1–3": ("248631", "402473"),
        "4–6": ("37740", "160556"),
        "7–10": ("86582", "166478"),
        "11+": ("66926", "537506"),
    },
    "voc": {
        "1–3": ("2009_002521", "2010_002531"),
        "4–6": ("2010_002546", "2008_000795"),
        "7–10": ("2011_001110", "2011_002358"),
        "11+": ("2007_005173", "2008_008362"),
    },
}


def read_jsonl(path: Path):
    with path.open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False))


def prepare_dense_candidates(args) -> None:
    """Select high-quality 11+ GT-object scenes for bounded model replay."""
    output = Path(args.output_dir)
    candidate_file = output / "candidates.json"
    if candidate_file.exists() and not args.overwrite:
        print(f"Reusing {candidate_file}")
        return

    voc_ids = Path(args.voc_root, "sets/val.txt").read_text().split()
    entries = []
    for dataset in ("coco", "voc"):
        source = FULL_EVAL / f"{dataset}_ar"
        summary = json.loads((source / "summary.json").read_text())
        if Path(summary["ckpt"]).resolve() != Path(args.model_path).resolve():
            raise ValueError(f"Checkpoint mismatch in {source}")
        captions = {row["image_id"]: row for row in read_jsonl(source / "generated_captions.jsonl")}
        rows = [row for row in read_jsonl(source / "per_image.jsonl")
                if row["gt_objects"] >= 11 and row["mIoU"] >= args.min_miou]
        rows.sort(key=lambda row: (row["mIoU"] + .12 * row["fARI"], row["mIoU"]), reverse=True)
        for row in rows[:args.per_dataset]:
            image_id = row["image_id"]
            sample_id = str(image_id) if dataset == "coco" else voc_ids[int(image_id)]
            caption = captions[image_id]
            entries.append({
                "dataset": dataset,
                "id": sample_id,
                "source_image_id": image_id,
                "screening_type": "dense_11plus",
                "prior_metrics": row,
                "ar_caption": caption,
                "caption_source": str(source / "generated_captions.jsonl"),
                "caption_sha256": hashlib.sha256(caption["text"].encode()).hexdigest(),
            })

    write_json(candidate_file, {
        "checkpoint": str(Path(args.model_path).resolve()),
        "seed": args.seed,
        "selection": (
            "GT-count-assisted qualitative dense-scene shortlist; selected by DeOR mIoU "
            "before replaying baselines; not a representative benchmark"
        ),
        "candidate_budget_per_dataset_type": args.per_dataset,
        "entries": entries,
    })
    print(f"Prepared {len(entries)} dense-scene candidates in {candidate_file}")


def prepare_direct_edit_retry(args) -> None:
    """Create a bounded fresh-image retry list for weak E.5 edit types."""
    source = SEGMENTATION_ROOT
    output = APPENDIX_ROOT / "e5_direct_retry"
    candidate_entries = json.loads((source / "candidates.json").read_text())["entries"]
    entries_by_id = {str(entry["id"]): entry for entry in candidate_entries
                     if entry["dataset"] == "coco"}
    # Multiple bounded attempts are retained for visual audit; only one strong,
    # non-overlapping case per operation will be selected for the final figure.
    specs = [
        ("remove", "503755", "person", None, None),
        ("remove", "333956", "parking meter", None, None),
        ("remove", "566436", "bench", None, None),
        ("replace", "241326", "couch", "182162", "couch"),
        ("replace", "357941", "tv", "98839", "tv"),
        ("replace", "312213", "laptop", "324715", "laptop"),
        ("replace", "245764", "toilet", "157807", "toilet"),
        ("add", "112997", "cell phone", "312213", "cat"),
        ("add", "333956", "parking meter", "182162", "couch"),
        ("add", "54931", "person", "217425", "clock"),
        ("add", "503755", "person", "566436", "bench"),
    ]

    def slot(image_id: str, category: str):
        entry = entries_by_id[image_id]
        slots = parse_ar_slots(entry["ar_caption"]["text"], entry["ar_caption"]["ovt_count"])
        matches = [value for value in slots if value["normalized_category"] == category]
        if len(matches) != 1:
            raise ValueError(f"Expected one {category!r} in COCO {image_id}, found {len(matches)}")
        return matches[0]

    tasks, used_entries = [], {}
    for index, (mode, target_id, target_category, donor_id, donor_category) in enumerate(specs):
        target = slot(target_id, target_category)
        donor = slot(donor_id, donor_category) if donor_id else None
        used_entries[target_id] = entries_by_id[target_id]
        if donor_id:
            used_entries[donor_id] = entries_by_id[donor_id]
        tasks.append({
            "task_id": f"appendix_{mode}_{index:02d}_{target_id}", "mode": mode,
            "target_id": target_id, "target_slot": target["slot_index"],
            "target_category": target_category,
            "target_prediction_label": target["prediction_label"],
            "donor_id": donor_id, "donor_slot": donor["slot_index"] if donor else None,
            "donor_category": donor_category,
            "donor_prediction_label": donor["prediction_label"] if donor else None,
            "placement": "donor native location; no text-controlled repositioning",
        })
    write_json(output / "tasks.json", {
        "checkpoint": str(Path(args.model_path)), "seed": args.seed,
        "steps": args.diffusion_inference_steps, "entries": used_entries, "tasks": tasks,
        "selection": "bounded fresh-image qualitative retries for E.5; no GT mask/caption at inference",
    })
    print(output / "tasks.json")


def font(size: int, bold: bool = False):
    name = "DejaVuSerif-Bold.ttf" if bold else "DejaVuSerif.ttf"
    return ImageFont.truetype(f"/usr/share/fonts/truetype/dejavu/{name}", size)


def centered(draw, text, box, text_font, fill="black"):
    left, top, width, height = box
    bb = draw.textbbox((0, 0), text, font=text_font)
    draw.text((left + (width - (bb[2] - bb[0])) / 2 - bb[0],
               top + (height - (bb[3] - bb[1])) / 2 - bb[1]),
              text, font=text_font, fill=fill)


def save_figure(image: Image.Image, directory: Path, stem: str) -> None:
    Image.init()
    directory.mkdir(parents=True, exist_ok=True)
    image.save(directory / f"{stem}.png", dpi=(300, 300))
    image.save(directory / f"{stem}.pdf", resolution=300)
    max_width = 1800
    ratio = min(1.0, max_width / image.width)
    image.resize((round(image.width * ratio), round(image.height * ratio)),
                 Image.Resampling.LANCZOS).save(directory / f"{stem}_preview.jpg", quality=94)


def cached_example(dataset: str, sample_id: str) -> tuple[Path, dict]:
    for root in (SEGMENTATION_ROOT, DENSE_ROOT):
        cache = root / "cache" / dataset / sample_id
        if (cache / "gt.npz").exists():
            scores = {model: json.loads((cache / f"{model}.json").read_text())
                      for model, _ in MODEL_COLUMNS}
            return cache, scores
    raise FileNotFoundError(f"No complete qualitative cache for {dataset}:{sample_id}")


def render_object_decomposition(dataset: str, output_root: Path, alpha: float = .64) -> None:
    """Render eight unchanged predictions, two examples per GT-count range."""
    # Keep labels readable after the full-width raster is reduced in the paper.
    # At the final figure width, 48 px is approximately the caption font size.
    panel, gap, header, band_gap, margin = 384, 6, 82, 24, 104
    names = ("Image", "GT") + tuple(name for _, name in MODEL_COLUMNS)
    width = margin + len(names) * panel + (len(names) - 1) * gap
    height = header + 8 * panel + 3 * band_gap
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    title_font, band_font = font(48, True), font(36, True)
    for column, name in enumerate(names):
        centered(draw, name, (margin + column * (panel + gap), 0, panel, header), title_font)

    manifest = {
        "dataset": dataset,
        "model": "Reader-supervised DeOR-B",
        "checkpoint": str(DEFAULT_MODEL),
        "selection": "GT-count-assisted qualitative selection; two cases per bin; not a benchmark",
        "prediction_labels_modified": False,
        "colour_matching": "Hungarian IoU matching for display colours only",
        "overlay_alpha": alpha,
        "examples": [],
    }
    row = 0
    band_positions = []
    for band_index, (count_range, sample_ids) in enumerate(E1_SELECTIONS[dataset].items()):
        band_top = header + row * panel + band_index * band_gap
        band_positions.append((count_range, band_top))
        for sample_id in sample_ids:
            cache, scores = cached_example(dataset, sample_id)
            data = np.load(cache / "gt.npz")
            gt, ignore = data["gt"], data["ignore"].astype(bool)
            original = np.asarray(Image.open(cache / "input.png").convert("RGB"))
            panels = [original]
            details = {}
            for model in ("gt",) + tuple(key for key, _ in MODEL_COLUMNS):
                source = cache / ("gt.npz" if model == "gt" else f"{model}.npz")
                labels = gt if model == "gt" else np.load(source)["prediction"]
                before = labels.copy()
                colours, mapping = label_colors(labels, gt, ignore, (245, 55, 104))
                panels.append(blend(original, colours, alpha))
                if not np.array_equal(labels, before):
                    raise AssertionError("Rendering modified a prediction")
                details[model] = {"source": str(source.resolve()),
                                  "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                                  "display_colours": {str(k): list(v) for k, v in mapping.items()}}
            y = header + row * panel + band_index * band_gap
            for column, image in enumerate(panels):
                canvas.paste(Image.fromarray(image).resize((panel, panel), Image.Resampling.LANCZOS),
                             (margin + column * (panel + gap), y))
            manifest["examples"].append({"count_range": count_range, "id": sample_id,
                "scores": scores, "panels": details})
            row += 1
    # Draw row-group labels last so later panel pastes can never cover them.
    for count_range, band_top in band_positions:
        centered(draw, count_range, (0, band_top, margin - 8, panel * 2), band_font)
    directory = output_root / f"e1_object_decomposition_{dataset}"
    save_figure(canvas, directory, f"object_decomposition_{dataset}")
    write_json(directory / "manifest.json", manifest)
    print(directory)


def wrap_pixels(draw, text: str, text_font, max_width: int):
    lines = []
    for word in text.split():
        trial = word if not lines else lines[-1] + " " + word
        if lines and draw.textlength(trial, font=text_font) <= max_width:
            lines[-1] = trial
        else:
            lines.append(word)
    return lines


def render_description_alignment(output_root: Path) -> None:
    source = PROJECT / "outputs/figure_deor_ar_grounding/appendix"
    folders = sorted(path for path in source.iterdir() if path.is_dir())
    if len(folders) != 6:
        raise ValueError(f"Expected six alignment examples, found {len(folders)}")
    panel, panel_gap, tile_gap, outer = 300, 5, 30, 12
    header = 150
    tile_w, tile_h = panel * 3 + panel_gap * 2, header + panel
    canvas = Image.new("RGB", (outer * 2 + tile_w * 2 + tile_gap,
                               outer * 2 + tile_h * 3 + tile_gap * 2), "white")
    draw = ImageDraw.Draw(canvas)
    heading_font, text_font = font(23, True), font(20)
    manifest = {"model": "Reader-supervised DeOR-B", "input": "image-only autoregressive",
                "external_text_query": False,
                "caption_display": "verbatim generated first sentence",
                "prediction_masks_modified": False, "examples": []}
    for index, folder in enumerate(folders):
        metadata = json.loads((folder / "metadata.json").read_text())
        chosen_slots = metadata["slots"][:2]
        x0 = outer + (index % 2) * (tile_w + tile_gap)
        y0 = outer + (index // 2) * (tile_h + tile_gap)
        paths = [folder / "image.png", folder / "object_01_overlay.png", folder / "object_02_overlay.png"]
        record = {"dataset": metadata["dataset"], "id": metadata["id"],
                  "descriptions": [], "sources": []}
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
                text_height = len(lines) * line_height
                top = y0 + max(0, header - text_height - 6)
                for line_index, line in enumerate(lines):
                    draw.text((x + 9, top + line_index * line_height), line,
                              font=text_font, fill="black")
            image = Image.open(path).convert("RGB").resize((panel, panel), Image.Resampling.LANCZOS)
            canvas.paste(image, (x, y0 + header))
            record["sources"].append({"path": str(path.resolve()),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
        manifest["examples"].append(record)
    directory = output_root / "e2_description_region_alignment"
    save_figure(canvas, directory, "description_region_alignment")
    write_json(directory / "manifest.json", manifest)
    print(directory)


def crop_grid_tile(grid: Image.Image, row: int, column: int) -> Image.Image:
    width, height = grid.size
    return grid.crop((round(column * width / 4), round(row * height / 3),
                      round((column + 1) * width / 4), round((row + 1) * height / 3))).convert("RGB")


def montage(images, size: int) -> Image.Image:
    images = [image.resize((size, size), Image.Resampling.LANCZOS) for image in images]
    if len(images) == 1:
        return images[0]
    output = Image.new("RGB", (size, size), "white")
    if len(images) == 2:
        output.paste(images[0].crop((0, 0, size // 2, size)), (0, 0))
        output.paste(images[1].crop((size // 2, 0, size, size)), (size // 2, 0))
        ImageDraw.Draw(output).line((size // 2, 0, size // 2, size), fill="white", width=4)
    else:
        half = size // 2
        output.paste(images[0].resize((half, size)), (0, 0))
        output.paste(images[1].resize((size - half, half)), (half, 0))
        output.paste(images[2].resize((size - half, size - half)), (half, half))
        d = ImageDraw.Draw(output)
        d.line((half, 0, half, size), fill="white", width=4)
        d.line((half, half, size, half), fill="white", width=4)
    return output


def render_compositional_generation(output_root: Path) -> None:
    root = PROJECT / "outputs/compgen_pgot_oneshot_memory8_register32_noprior_7b_fp32_fast"
    rows = read_jsonl(root / "compositions.jsonl")[:32]
    chosen = (0, 6, 21, 29)
    caption_rows = {row["image_id"]: row for row in read_jsonl(
        FULL_EVAL / "coco_ar/generated_captions.jsonl")}
    grids = {}
    source_tiles = {}
    for batch in range(8):
        grid = Image.open(root / "examples" / f"batch{batch:03d}_real_recon_composed.png").convert("RGB")
        grids[batch] = grid
        for column in range(4):
            target = rows[batch * 4 + column]["target_image_id"]
            source_tiles[target] = crop_grid_tile(grid, 0, column)

    panel, gap, row_gap, left = 360, 12, 30, 110
    header = 64
    width = left + 3 * panel + 2 * gap
    height = header + 4 * panel + 3 * row_gap
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    heading = font(27, True)
    for column, name in enumerate(("Object sources", "Background source", "Composed image")):
        centered(draw, name, (left + column * (panel + gap), 0, panel, header), heading)
    manifest = {"model": "DeOR-L (7B)", "checkpoint": str(json.loads((root / "summary.json").read_text())["model_path"]),
                "selection": "four visually audited outputs from the first 32 saved random compositions",
                "composition_rule": json.loads((root / "summary.json").read_text())["mixing_rule"],
                "examples": []}
    for row_index, composition_index in enumerate(chosen):
        record = rows[composition_index]
        unique_object_sources = list(dict.fromkeys(item["image_id"] for item in record["objects"]))
        background_source = record["registers"][0]["image_id"]
        categories = []
        for item in record["objects"]:
            caption = caption_rows[item["image_id"]]
            slots = parse_ar_slots(caption["text"], caption["ovt_count"])
            if item["object_index"] < len(slots):
                categories.append(slots[item["object_index"]]["category"])
        object_panel = montage([source_tiles[i] for i in unique_object_sources[:3]], panel)
        background_panel = source_tiles[background_source].resize((panel, panel), Image.Resampling.LANCZOS)
        composed = crop_grid_tile(grids[composition_index // 4], 2, composition_index % 4).resize(
            (panel, panel), Image.Resampling.LANCZOS)
        y = header + row_index * (panel + row_gap)
        for column, image in enumerate((object_panel, background_panel, composed)):
            canvas.paste(image, (left + column * (panel + gap), y))
        centered(draw, str(row_index + 1), (0, y, left - 10, panel), font(27, True))
        manifest["examples"].append({"composition_index": composition_index,
            "object_source_image_ids": unique_object_sources,
            "background_source_image_id": background_source,
            "object_categories": categories, "record": record})
    directory = output_root / "e3_compositional_generation"
    save_figure(canvas, directory, "compositional_generation")
    write_json(directory / "manifest.json", manifest)
    print(directory)


def render_direct_manipulation(output_root: Path) -> None:
    examples = [
        ("Remove", output_root / "e5_direct_retry/candidates/remove/appendix_remove_00_503755"),
        ("Replace", output_root / "e5_direct_retry/candidates/replace/appendix_replace_05_312213"),
        ("Add", output_root / "e5_direct_retry/candidates/add/appendix_add_09_54931"),
        ("Background Transfer", PROJECT / "outputs/figure_model_editing_retry_add_background/final_native/background/01_568690"),
    ]
    panel, gap, header, row_gap, left = 380, 8, 58, 30, 150
    width = left + 3 * panel + 2 * gap
    height = header + 4 * panel + 3 * row_gap
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    for column, title in enumerate(("Source", "Selected representation", "Edited result")):
        centered(draw, title, (left + column * (panel + gap), 0, panel, header), font(27, True))
    manifest = {"model": "Reader-supervised DeOR-B", "selection_input": "direct representation IDs; no natural-language selector",
                "operations": [], "background_operation_name": "Background Transfer"}
    for row_index, (operation, folder) in enumerate(examples):
        metadata = json.loads((folder / "metadata.json").read_text())
        y = header + row_index * (panel + row_gap)
        for column, filename in enumerate(("panel_0.png", "panel_1.png", "panel_2.png")):
            path = folder / filename
            image = Image.open(path).convert("RGB").resize((panel, panel), Image.Resampling.LANCZOS)
            canvas.paste(image, (left + column * (panel + gap), y))
        display_operation = "Background\nTransfer" if operation == "Background Transfer" else operation
        op_font = font(21, True)
        if "\n" in display_operation:
            lines = display_operation.splitlines()
            line_h = 27
            top = y + (panel - len(lines) * line_h) // 2
            for line_index, line in enumerate(lines):
                centered(draw, line, (0, top + line_index * line_h, left - 10, line_h), op_font)
        else:
            centered(draw, display_operation, (0, y, left - 10, panel), op_font)
        manifest["operations"].append({"display_name": operation, "metadata": metadata,
            "directory": str(folder.resolve()), "appendix_visual_review": "pass",
            "source_visual_success_verified": metadata["visual_success_verified"]})
    directory = output_root / "e5_direct_manipulation"
    save_figure(canvas, directory, "direct_object_manipulation")
    write_json(directory / "manifest.json", manifest)
    print(directory)


def mask_overlay(image: Image.Image, mask_path: Path, colour=(55, 220, 100), alpha=.48) -> Image.Image:
    rgb = np.asarray(image.convert("RGB")).copy()
    mask_image = Image.open(mask_path)
    mask = np.asarray(mask_image.convert("L")) > 0
    if mask.shape != rgb.shape[:2]:
        mask = np.asarray(mask_image.convert("L").resize(image.size, Image.Resampling.NEAREST)) > 0
    rgb[mask] = np.rint((1 - alpha) * rgb[mask] + alpha * np.asarray(colour)).astype(np.uint8)
    return Image.fromarray(rgb)


def render_language_guided_editing(output_root: Path) -> None:
    raw = output_root / "e4_language_guided_raw"
    retry = output_root / "e4_language_guided_retry"
    examples = [
        (retry, "appendix_retry_001"),
        (retry, "appendix_retry_002"),
        (retry, "appendix_retry_003"),
        (raw, "appendix_edit_003"),
        (raw, "appendix_edit_004"),
    ]
    panel, gap, outer = 320, 7, 10
    top_header, instruction_h, row_gap = 58, 50, 24
    width = outer * 2 + 4 * panel + 3 * gap
    height = top_header + 5 * (instruction_h + panel) + 4 * row_gap + outer
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    for column, title in enumerate(("Selected target", "Selected donor / description",
                                    "Reconstruction", "Edited result")):
        centered(draw, title, (outer + column * (panel + gap), 0, panel, top_header), font(22, True))
    manifest = {"model": "Reader-supervised DeOR-B", "selector": "Qwen3-VL-4B-Instruct (text-only)",
                "selector_image_input": False, "selector_gt_input": False,
                "manual_plan_correction": False, "examples": []}
    for row_index, (root, edit_id) in enumerate(examples):
        folder = root / "edits" / edit_id
        metadata = json.loads((folder / "metadata.json").read_text())
        plan = json.loads((root / "plans" / f"{edit_id}.json").read_text())
        request = metadata["query"]["request"]
        y = top_header + row_index * (instruction_h + panel + row_gap)
        draw.text((outer + 4, y + 10), request, fill=(190, 25, 25), font=font(22, True))
        source = Image.open(folder / "source_A.png").convert("RGB")
        target = mask_overlay(source, folder / "target_prediction_mask.png").resize(
            (panel, panel), Image.Resampling.LANCZOS)
        canvas.paste(target, (outer, y + instruction_h))
        donor_path = folder / "donor_B.png"
        if donor_path.exists():
            donor = Image.open(donor_path).convert("RGB")
            donor_mask = folder / "donor_prediction_mask.png"
            donor_panel = mask_overlay(donor, donor_mask) if donor_mask.exists() else donor
        else:
            donor_panel = Image.new("RGB", (panel, panel), "white")
            dd = ImageDraw.Draw(donor_panel)
            description = metadata["executed"]["target"]["description"]
            lines = wrap_pixels(dd, first_sentence(description), font(18), panel - 28)
            top = max(14, (panel - len(lines) * 25) // 2)
            for line_index, line in enumerate(lines):
                dd.text((14, top + line_index * 25), line, fill="black", font=font(18))
        canvas.paste(donor_panel.resize((panel, panel), Image.Resampling.LANCZOS),
                     (outer + panel + gap, y + instruction_h))
        reconstruction = Image.open(folder / "reconstruction_native.png").convert("RGB").resize(
            (panel, panel), Image.Resampling.LANCZOS)
        edited = Image.open(folder / "edited_native.png").convert("RGB").resize(
            (panel, panel), Image.Resampling.LANCZOS)
        canvas.paste(reconstruction, (outer + 2 * (panel + gap), y + instruction_h))
        canvas.paste(edited, (outer + 3 * (panel + gap), y + instruction_h))
        manifest["examples"].append({"id": edit_id, "request": request,
            "root": str(root.resolve()), "selector_status": plan["status"],
            "actual_plan": plan["plan"], "executed": metadata["executed"]})
    directory = output_root / "e4_language_guided_editing"
    save_figure(canvas, directory, "language_guided_object_editing")
    write_json(directory / "manifest.json", manifest)
    print(directory)


def coco_crop(image_id: int) -> Image.Image:
    from pgot.train.pgot_dataset import _coda_center_crop_image
    path = Path("/home/jovyan/data/coco/val2017") / f"{image_id:012d}.jpg"
    return _coda_center_crop_image(Image.open(path).convert("RGB"), 512)


def render_failure_cases(output_root: Path) -> None:
    panel, gap, tile_gap, band_gap, outer = 260, 5, 32, 44, 12
    tile_w, tile_h = panel * 3 + gap * 2, 56 + panel
    width = outer * 2 + tile_w * 2 + tile_gap
    height = outer * 2 + tile_h * 3 + band_gap * 2
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    label_font, small_font = font(22, True), font(17)
    records = []

    # Description failures: the generated text itself states an absent or unsupported object.
    captions = {row["image_id"]: row for row in read_jsonl(FULL_EVAL / "coco_ar/generated_captions.jsonl")}
    description_cases = [
        (19402, "Cake", "Generated an object for a background illustration, not a physical cake."),
        (164602, "Toothbrush 2", "Generated a second toothbrush while explicitly stating it is not visible."),
    ]
    for column, (image_id, object_name, note) in enumerate(description_cases):
        x, y = outer + column * (tile_w + tile_gap), outer
        image = coco_crop(image_id).resize((panel, panel), Image.Resampling.LANCZOS)
        canvas.paste(image, (x, y + 56))
        text_box = Image.new("RGB", (panel * 2 + gap, panel), "white")
        td = ImageDraw.Draw(text_box)
        full = captions[image_id]["text"]
        parsed_slots = parse_ar_slots(full, captions[image_id]["ovt_count"])
        base_name = object_name.rsplit(" ", 1)[0] if object_name[-1:].isdigit() else object_name
        matches = [slot for slot in parsed_slots if slot["category"].lower() == base_name.lower()]
        occurrence = int(object_name.rsplit(" ", 1)[1]) - 1 if object_name[-1:].isdigit() else 0
        target = matches[occurrence] if occurrence < len(matches) else None
        sentence = first_sentence(target["description"]) if target else note
        lines = wrap_pixels(td, f'“{sentence}”', small_font, text_box.width - 26)
        top = max(12, (panel - len(lines) * 24) // 2)
        for li, line in enumerate(lines):
            td.text((13, top + li * 24), line, font=small_font, fill="black")
        canvas.paste(text_box, (x + panel + gap, y + 56))
        centered(draw, f"Description failure · {object_name}", (x, y, tile_w, 50), label_font)
        records.append({"type": "description", "image_id": image_id, "object": object_name, "note": note})

    # Decomposition failures: unchanged cached input, GT and DeOR prediction.
    decomposition_cases = [("coco", "42628", "Several small people are missed/merged."),
                           ("voc", "2009_002372", "Crowded people are merged into coarse regions.")]
    for column, (dataset, sample_id, note) in enumerate(decomposition_cases):
        x = outer + column * (tile_w + tile_gap)
        y = outer + tile_h + band_gap
        cache, scores = cached_example(dataset, sample_id)
        data = np.load(cache / "gt.npz")
        gt, ignore = data["gt"], data["ignore"].astype(bool)
        original = np.asarray(Image.open(cache / "input.png").convert("RGB"))
        pred = np.load(cache / "ours.npz")["prediction"]
        gt_rgb, _ = label_colors(gt, gt, ignore, (245, 55, 104))
        pred_rgb, _ = label_colors(pred, gt, ignore, (245, 55, 104))
        panels = (original, blend(original, gt_rgb, .64), blend(original, pred_rgb, .64))
        for i, image in enumerate(panels):
            canvas.paste(Image.fromarray(image).resize((panel, panel), Image.Resampling.LANCZOS),
                         (x + i * (panel + gap), y + 56))
        centered(draw, "Decomposition failure", (x, y, tile_w, 50), label_font)
        records.append({"type": "decomposition", "dataset": dataset, "id": sample_id,
                        "note": note, "scores": scores["ours"]})

    editing_cases = [
        (PROJECT / "outputs/figure_model_editing_retry_add_background/final_native/background/08_530099",
         "Background transfer duplicates/ghosts the cat."),
        (PROJECT / "outputs/figure_model_editing_retry_add_background/final_native/background/04_335658",
         "Transferred desk context distorts and repositions the keyboard and mouse."),
    ]
    for column, (folder, note) in enumerate(editing_cases):
        x = outer + column * (tile_w + tile_gap)
        y = outer + 2 * (tile_h + band_gap)
        metadata = json.loads((folder / "metadata.json").read_text())
        for i, filename in enumerate(("panel_0.png", "panel_1.png", "panel_2.png")):
            image = Image.open(folder / filename).convert("RGB").resize((panel, panel), Image.Resampling.LANCZOS)
            canvas.paste(image, (x + i * (panel + gap), y + 56))
        centered(draw, "Editing failure", (x, y, tile_w, 50), label_font)
        records.append({"type": "editing", "directory": str(folder.resolve()),
                        "note": note, "metadata": metadata})
    directory = output_root / "e6_failure_cases"
    save_figure(canvas, directory, "failure_cases")
    write_json(directory / "manifest.json", {"model": "Reader-supervised DeOR-B",
        "selection": "observed failures; not a frequency estimate", "examples": records})
    print(directory)


def audit_case_overlap(output_root: Path) -> None:
    """Fail if a displayed source image is reused across appendix/main figures."""
    appendix = {
        "E1 COCO": {f"coco:{sample_id}" for bins in E1_SELECTIONS["coco"].values()
                    for sample_id in bins},
        "E1 VOC": {f"voc:{sample_id}" for bins in E1_SELECTIONS["voc"].values()
                   for sample_id in bins},
        "E2": {
            "coco:22935", "coco:454750", "coco:481573",
            "voc:2007_009346", "voc:2007_005845", "voc:2007_007417",
        },
        # Only displayed object/background sources are counted.  The hidden
        # composition template image is not shown anywhere in E3.
        "E3": {f"coco:{image_id}" for image_id in
               (632, 285, 724, 785, 776, 2149, 2006, 1818, 2587, 2431, 2532)},
        "E4": {f"coco:{image_id}" for image_id in
               (565045, 504439, 170893, 84431, 554735, 112626, 350019)},
        "E5": {f"coco:{image_id}" for image_id in
               (503755, 312213, 324715, 54931, 217425, 568690, 176778)},
        "E6": {
            *(f"coco:{image_id}" for image_id in
              (19402, 164602, 42628, 530099, 508602, 335658, 68765)),
            "voc:2009_002372",
        },
    }
    main_paper = {
        *(f"coco:{image_id}" for image_id in
          (552371, 559956, 90208, 450686, 339823,
           323151, 52891, 205834, 261161, 193429, 554595, 449909, 802, 243344)),
        "voc:2007_005149", "voc:2007_007795", "voc:2010_004165",
    }
    owners = {}
    appendix_conflicts = []
    for figure, cases in appendix.items():
        for case in sorted(cases):
            if case in owners:
                appendix_conflicts.append({"case": case, "figures": [owners[case], figure]})
            else:
                owners[case] = figure
    main_conflicts = [{"case": case, "appendix_figure": owners[case]}
                      for case in sorted(main_paper & set(owners))]
    report = {
        "passed": not appendix_conflicts and not main_conflicts,
        "policy": "No displayed source image is reused across appendix figures or the audited main figures.",
        "appendix_figures": {name: sorted(values) for name, values in appendix.items()},
        "audited_main_figure_cases": sorted(main_paper),
        "appendix_conflicts": appendix_conflicts,
        "main_figure_conflicts": main_conflicts,
    }
    write_json(output_root / "overlap_audit.json", report)
    if not report["passed"]:
        raise RuntimeError(f"Case-overlap audit failed: {report}")
    print(output_root / "overlap_audit.json")


def write_index(output_root: Path) -> None:
    figures = [
        ("E.1 COCO", "e1_object_decomposition_coco/object_decomposition_coco.pdf",
         "Object decomposition from sparse to crowded COCO scenes."),
        ("E.1 VOC", "e1_object_decomposition_voc/object_decomposition_voc.pdf",
         "Zero-shot object decomposition from sparse to crowded VOC scenes."),
        ("E.2", "e2_description_region_alignment/description_region_alignment.pdf",
         "Verbatim autoregressive descriptions paired with their predicted regions."),
        ("E.3", "e3_compositional_generation/compositional_generation.pdf",
         "DeOR-L object representations recomposed across images."),
        ("E.4", "e4_language_guided_editing/language_guided_object_editing.pdf",
         "Text-only language selection followed by object removal or appearance transfer."),
        ("E.5", "e5_direct_manipulation/direct_object_manipulation.pdf",
         "Direct remove, replace, add, and background-transfer operations."),
        ("E.6", "e6_failure_cases/failure_cases.pdf",
         "Observed description, decomposition, and editing failures."),
    ]
    lines = [
        "# Appendix qualitative figures", "",
        "E.1, E.2, E.4, E.5, and E.6 use the Reader-supervised DeOR-B checkpoint:",
        f"`{DEFAULT_MODEL}`", "",
        "E.3 uses the saved DeOR-L 7B compositional outputs. Every directory contains a PDF,",
        "a lossless PNG, a lightweight JPG preview, and a provenance manifest.", "",
    ]
    for label, relative, description in figures:
        lines.extend((f"- **{label}:** `{relative}`  ", f"  {description}", ""))
    lines.extend((
        "Case reuse is checked by `overlap_audit.json`. Selection is qualitative and is not",
        "reported as an unbiased quantitative estimate.", "",
    ))
    (output_root / "README.md").write_text("\n".join(lines))


def render_all(output_root: Path) -> None:
    for dataset in ("coco", "voc"):
        render_object_decomposition(dataset, output_root)
    render_description_alignment(output_root)
    render_compositional_generation(output_root)
    render_language_guided_editing(output_root)
    render_direct_manipulation(output_root)
    render_failure_cases(output_root)
    audit_case_overlap(output_root)
    write_index(output_root)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("prepare_dense", "prepare_direct_retry",
                                             "render_e1", "render_e2", "render_e3",
                                             "render_e4", "render_e5", "render_e6", "audit",
                                             "render_all"), required=True)
    parser.add_argument("--output_dir", default=str(
        PROJECT / "outputs/paper_figures/appendix_qualitative_v1/dense_scene_eval"))
    parser.add_argument("--model_path", default=str(DEFAULT_MODEL))
    parser.add_argument("--voc_root", default="/home/jovyan/data/voc")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--diffusion_inference_steps", type=int, default=10)
    parser.add_argument("--per_dataset", type=int, default=8)
    parser.add_argument("--min_miou", type=float, default=.30)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.per_dataset < 1 or not 0 <= args.min_miou <= 1:
        parser.error("Invalid candidate-selection arguments")
    if args.stage == "prepare_dense":
        prepare_dense_candidates(args)
    elif args.stage == "prepare_direct_retry":
        prepare_direct_edit_retry(args)
    elif args.stage == "render_e1":
        for dataset in ("coco", "voc"):
            render_object_decomposition(dataset, APPENDIX_ROOT)
    elif args.stage == "render_e2":
        render_description_alignment(APPENDIX_ROOT)
    elif args.stage == "render_e3":
        render_compositional_generation(APPENDIX_ROOT)
    elif args.stage == "render_e4":
        render_language_guided_editing(APPENDIX_ROOT)
    elif args.stage == "render_e5":
        render_direct_manipulation(APPENDIX_ROOT)
    elif args.stage == "render_e6":
        render_failure_cases(APPENDIX_ROOT)
    elif args.stage == "audit":
        audit_case_overlap(APPENDIX_ROOT)
        write_index(APPENDIX_ROOT)
    elif args.stage == "render_all":
        render_all(APPENDIX_ROOT)


if __name__ == "__main__":
    main()
