"""Rebuild the three fixed qualitative paper figures with the no-Reader-supervision checkpoint.

The examples and baseline predictions are fixed to the previously curated figures.
Only DeOR captions, masks, reconstructions, and edits are recomputed.  This is a
controlled qualitative replay, not a new benchmark or a fresh best-case search.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import textwrap
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from pgot.eval.ar_binding_router import normalize_category, parse_ar_slots
from pgot.eval.render_segmentation_paper_figure import blend, label_colors


PROJECT = Path(__file__).resolve().parents[2]
CHECKPOINT = PROJECT / "checkpoints/pgot_oneshot_memory8_register32_noprior_woreader/checkpoint-5000"
OLD_SOURCE = PROJECT / "outputs/figure_segmentation_over_under"
SOURCE = PROJECT / "outputs/figure_segmentation_woreader_fixed"
COCO_EVAL = PROJECT / "outputs/eval_pgot_oneshot_memory8_register32_noprior_woreader/ar"
VOC_EVAL = PROJECT / "outputs/segeval_pgot_oneshot_memory8_register32_noprior_woreader/voc_ar"
PAPER_ROOT = PROJECT / "outputs/paper_figures"
SEG_OUT = PAPER_ROOT / "segmentation_comparison_woreader"
GROUND_OUT = PAPER_ROOT / "deor_grounding_2x2_woreader"
REMOVE_OUT = PROJECT / "outputs/figure_model_editing_woreader_fixed3"
TRANSFER_OUT = PROJECT / "outputs/object_representation_transfer_woreader_fixed3"
EDIT_OUT = PAPER_ROOT / "object_editing_remove_replace_3x2_woreader"
TRANSFER_PAIRS = PROJECT / "configs/object_representation_transfer_woreader_figure3.jsonl"

SEGMENTATION_ROWS = [
    ("COCO", "over", "coco", "552371"),
    ("COCO", "under", "coco", "559956"),
    ("VOC", "over", "voc", "2007_005149"),
    ("VOC", "under", "voc", "2007_007795"),
]
GROUNDING_ROWS = [
    ("coco", "90208"),
    ("coco", "450686"),
    ("voc", "2010_004165"),
    ("coco", "339823"),
]
REMOVE_TASKS = [
    ("remove_12_323151", "323151", 1, "cup", "Remove the cup from image A."),
    ("remove_08_261161", "261161", 1, "bench", "Remove the bench from image A."),
    ("remove_13_449909", "449909", 1, "giraffe", "Remove the giraffe from image A."),
]
TRANSFER_ROWS = [
    {
        "pair_id": "dog_black_white_to_brown",
        "target_image_id": 52891,
        "target_slot": 0,
        "donor_image_id": 205834,
        "donor_slot": 0,
        "category": "dog",
        "instruction": "Replace A's black-and-white dog with B's brown dog.",
    },
    {
        "pair_id": "snowboard_to_star_surfboard",
        "target_image_id": 193429,
        "target_slot": 0,
        "donor_image_id": 554595,
        "donor_slot": 0,
        "target_category": "snowboard",
        "donor_category": "surfboard",
        "instruction": "Replace A's snowboard with B's star-patterned surfboard.",
    },
    {
        "pair_id": "refrigerator_white_to_steel",
        "target_image_id": 802,
        "target_slot": 0,
        "donor_image_id": 243344,
        "donor_slot": 0,
        "category": "refrigerator",
        "instruction": "Replace A's white refrigerator with B's stainless-steel refrigerator.",
    },
]


def read_json(path: Path):
    return json.loads(path.read_text())


def read_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False))


def sha256(path: Path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def font(size: int, bold: bool = False):
    suffix = "-Bold.ttf" if bold else ".ttf"
    return ImageFont.truetype(f"/usr/share/fonts/truetype/dejavu/DejaVuSerif{suffix}", size)


def center_text(draw, text, box, text_font, fill="black"):
    left, top, width, height = box
    x0, y0, x1, y1 = draw.textbbox((0, 0), text, font=text_font)
    draw.text((left + (width - (x1-x0))/2-x0, top + (height-(y1-y0))/2-y0),
              text, font=text_font, fill=fill)


def compact_thing_slots(entry):
    slots = [dict(slot) for slot in parse_ar_slots(
        entry["ar_caption"]["text"], entry["ar_caption"]["ovt_count"]) if slot["is_thing"]]
    for index, slot in enumerate(slots, 1):
        slot["prediction_label"] = index
    return slots


def prepare_source(_args):
    old = read_json(OLD_SOURCE / "candidates.json")
    old_entries = {(row["dataset"], row["id"]): row for row in old["entries"]}
    wanted = {(dataset, iid) for _, _, dataset, iid in SEGMENTATION_ROWS}
    wanted.update(GROUNDING_ROWS)
    wanted.update(("coco", iid) for _, iid, _, _, _ in REMOVE_TASKS)
    for pair in TRANSFER_ROWS:
        wanted.add(("coco", str(pair["target_image_id"])))
        wanted.add(("coco", str(pair["donor_image_id"])))

    eval_dirs = {"coco": COCO_EVAL, "voc": VOC_EVAL}
    captions = {
        dataset: {int(row["image_id"]): row for row in read_jsonl(root / "generated_captions.jsonl")}
        for dataset, root in eval_dirs.items()
    }
    per_image = {
        dataset: {int(row["image_id"]): row for row in read_jsonl(root / "per_image.jsonl")}
        for dataset, root in eval_dirs.items()
    }
    for dataset, root in eval_dirs.items():
        summary = read_json(root / "summary.json")
        if Path(summary["ckpt"]).resolve() != CHECKPOINT.resolve():
            raise ValueError(f"Unexpected {dataset} checkpoint: {summary['ckpt']}")
        if summary.get("caption_mode") != "autoregressive":
            raise ValueError(f"{dataset} source is not image-only AR")

    entries = []
    for key in sorted(wanted):
        if key not in old_entries:
            raise KeyError(f"Fixed example is absent from original cache: {key}")
        entry = dict(old_entries[key])
        source_index = int(entry["source_image_id"])
        caption = captions[key[0]][source_index]
        entry.update(
            prior_metrics=per_image[key[0]][source_index],
            ar_caption=caption,
            caption_source=str((eval_dirs[key[0]] / "generated_captions.jsonl").resolve()),
            caption_sha256=hashlib.sha256(caption["text"].encode()).hexdigest(),
        )
        entries.append(entry)
        src = OLD_SOURCE / "cache" / key[0] / key[1]
        dst = SOURCE / "cache" / key[0] / key[1]
        dst.mkdir(parents=True, exist_ok=True)
        for name in ("input.png", "gt.npz", "adaslot.npz", "adaslot.json",
                     "metaslot.npz", "metaslot.json", "coda.npz", "coda.json"):
            if (src / name).exists() and not (dst / name).exists():
                shutil.copy2(src / name, dst / name)

    write_json(SOURCE / "candidates.json", {
        "checkpoint": str(CHECKPOINT),
        "seed": 0,
        "selection": "Fixed replay of previously curated IDs; no re-mining for the new checkpoint",
        "candidate_budget_per_dataset_type": None,
        "entries": entries,
    })
    print(f"Prepared {len(entries)} fixed examples in {SOURCE}")


def segmentation_panels(dataset, iid, destination):
    cache = SOURCE / "cache" / dataset / iid
    destination.mkdir(parents=True, exist_ok=True)
    original = np.asarray(Image.open(cache / "input.png").convert("RGB"))
    gt_data = np.load(cache / "gt.npz")
    gt, ignore = gt_data["gt"], gt_data["ignore"].astype(bool)
    Image.fromarray(original).save(destination / "image.png")
    details = {}
    for name in ("gt", "adaslot", "metaslot", "coda", "ours"):
        path = cache / ("gt.npz" if name == "gt" else f"{name}.npz")
        labels = gt if name == "gt" else np.load(path)["prediction"]
        before = labels.copy()
        colors, mapping = label_colors(labels, gt, ignore, background_color=(245, 55, 104))
        rendered = blend(original, colors, .65)
        if not np.array_equal(labels, before):
            raise AssertionError("Display rendering modified a hard prediction")
        Image.fromarray(rendered).save(destination / f"{name}_overlay.png")
        details[name] = {
            "source": str(path.resolve()),
            "source_sha256": sha256(path),
            "label_colors": {str(k): list(v) for k, v in mapping.items()},
            "unique_labels": np.unique(labels).tolist(),
        }
    return details


def compose_segmentation(_args):
    Image.init()
    panel, gap, row_gap, group_gap = 512, 8, 12, 48
    outer, dataset_width, header = 16, 208, 88
    grid_left = outer + dataset_width
    width = grid_left + 6*panel + 5*gap + outer
    group_height = 2*panel + row_gap
    height = outer + header + 2*group_height + group_gap + outer
    figure = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(figure)
    columns = [("image", "Image"), ("gt_overlay", "GT"),
               ("adaslot_overlay", "AdaSlot"), ("metaslot_overlay", "MetaSlot"),
               ("coda_overlay", "CODA"), ("ours_overlay", "DeOR")]
    for index, (_, title) in enumerate(columns):
        center_text(draw, title, (grid_left+index*(panel+gap), outer, panel, header), font(48, True))
    manifest = {
        "checkpoint": str(CHECKPOINT),
        "layout": "4 rows x 6 panels; fixed examples from original figure",
        "controlled_replay": True,
        "baseline_predictions_reused": True,
        "deor_prediction_recomputed": True,
        "overlay_alpha": .65,
        "background_color_rgb": [245, 55, 104],
        "prediction_labels_modified": False,
        "rows": [],
    }
    for row_index, (dataset_title, mode, dataset, iid) in enumerate(SEGMENTATION_ROWS):
        group_index = 0 if dataset == "coco" else 1
        within_group = 0 if mode == "over" else 1
        group_y = outer + header + group_index*(group_height+group_gap)
        if within_group == 0:
            center_text(draw, dataset_title, (outer, group_y, dataset_width, group_height), font(52, True))
        y = group_y + within_group*(panel+row_gap)
        dest = SEG_OUT / "panels" / f"{dataset}_{mode}_{iid}"
        details = segmentation_panels(dataset, iid, dest)
        boxes = []
        for col, (filename, title) in enumerate(columns):
            path = dest / f"{filename}.png"
            x = grid_left + col*(panel+gap)
            figure.paste(Image.open(path).convert("RGB"), (x, y))
            boxes.append({"title": title, "source": str(path.resolve()), "sha256": sha256(path),
                          "box": [x, y, x+panel, y+panel]})
        metrics = {name: read_json(SOURCE / "cache" / dataset / iid / f"{name}.json")
                   for name in ("adaslot", "metaslot", "coda", "ours")}
        manifest["rows"].append({"dataset": dataset, "mode": mode, "id": iid,
                                 "metrics": metrics, "render_details": details, "panels": boxes})
    divider_y = outer + header + group_height + group_gap//2
    draw.line((outer, divider_y, width-outer, divider_y), fill=(188, 192, 198), width=2)
    SEG_OUT.mkdir(parents=True, exist_ok=True)
    figure.save(SEG_OUT / "segmentation_comparison_woreader.png", dpi=(300, 300))
    figure.save(SEG_OUT / "segmentation_comparison_woreader.pdf", resolution=300)
    figure.resize((1722, round(height*1722/width)), Image.Resampling.LANCZOS).save(SEG_OUT / "preview.png")
    write_json(SEG_OUT / "manifest.json", manifest)
    print(SEG_OUT / "segmentation_comparison_woreader.png")


def first_sentence(text):
    return re.split(r"(?<=[.!?])\s+", text)[0]


def wrap_pixels(text, draw, text_font, max_width):
    lines = []
    for word in text.split():
        if lines and draw.textlength(lines[-1] + " " + word, font=text_font) <= max_width:
            lines[-1] += " " + word
        else:
            lines.append(word)
    return lines


def grounding_overlay(image, mask, color, alpha=.60):
    pixels = np.asarray(image).copy()
    pixels[mask] = np.rint((1-alpha)*pixels[mask] + alpha*np.asarray(color)).astype(np.uint8)
    return Image.fromarray(pixels)


def compose_grounding(_args):
    Image.init()
    entries = {(row["dataset"], row["id"]): row for row in read_json(SOURCE / "candidates.json")["entries"]}
    rows = [entries[key] for key in GROUNDING_ROWS]
    colors = [(235, 65, 70), (65, 220, 110)]
    panel, panel_gap, horizontal_gap, vertical_gap, outer = 512, 8, 48, 52, 16
    title_font, desc_font = font(38, True), font(34)
    dummy = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    texts, wrapped, slots_by_row = [], [], []
    for entry in rows:
        slots = compact_thing_slots(entry)
        if len(slots) != 2:
            raise ValueError(f"Grounding layout requires two thing objects: {entry['dataset']}:{entry['id']}")
        current = [first_sentence(slot["description"]) for slot in slots]
        slots_by_row.append(slots)
        texts.append(current)
        wrapped.append([wrap_pixels(text, dummy, desc_font, panel-24) for text in current])
    line_height = 42
    header = max(len(lines) for row in wrapped for lines in row)*line_height + 32
    tile_width, tile_height = panel*3+panel_gap*2, header+panel
    width, height = outer*2+tile_width*2+horizontal_gap, outer*2+tile_height*2+vertical_gap
    figure = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(figure)
    manifest = {
        "checkpoint": str(CHECKPOINT),
        "layout": "2x2 fixed scenes; Image / generated-description region 1 / region 2",
        "input": "image-only autoregressive generation; descriptions are outputs, not prompts",
        "controlled_replay": True,
        "prediction_panels_modified": False,
        "sets": [],
    }
    for index, (entry, slots) in enumerate(zip(rows, slots_by_row)):
        cache = SOURCE / "cache" / entry["dataset"] / entry["id"]
        image = Image.open(cache / "input.png").convert("RGB")
        pred = np.load(cache / "ours.npz")["prediction"]
        tile_x = outer + (index % 2)*(tile_width+horizontal_gap)
        tile_y = outer + (index // 2)*(tile_height+vertical_gap)
        panels = [image] + [grounding_overlay(image, pred == slot["prediction_label"], colors[j])
                            for j, slot in enumerate(slots)]
        record = {"dataset": entry["dataset"], "id": entry["id"],
                  "caption_sha256": entry["caption_sha256"], "descriptions": texts[index], "panels": []}
        panel_dir = GROUND_OUT / "panels" / f"{index+1:02}_{entry['dataset']}_{entry['id']}"
        panel_dir.mkdir(parents=True, exist_ok=True)
        for col, current in enumerate(panels):
            path = panel_dir / ("image.png" if col == 0 else f"object_{col:02}_overlay.png")
            current.save(path)
            x = tile_x + col*(panel+panel_gap)
            if col == 0:
                center_text(draw, "Image", (x, tile_y+header-52, panel, 52), title_font)
            else:
                lines = wrapped[index][col-1]
                last_bottom = draw.textbbox((0, 0), lines[-1], font=desc_font)[3]
                text_top = tile_y+header-12-(len(lines)-1)*line_height-last_bottom
                for line_index, line in enumerate(lines):
                    draw.text((x+12, text_top+line_index*line_height), line, fill="black", font=desc_font)
            figure.paste(current, (x, tile_y+header))
            record["panels"].append({"source": str(path.resolve()), "sha256": sha256(path),
                                     "box": [x, tile_y+header, x+panel, tile_y+header+panel]})
        manifest["sets"].append(record)
    GROUND_OUT.mkdir(parents=True, exist_ok=True)
    figure.save(GROUND_OUT / "deor_grounding_2x2_woreader.png", dpi=(300, 300))
    figure.save(GROUND_OUT / "deor_grounding_2x2_woreader.pdf", resolution=300)
    figure.resize((width//2, height//2), Image.Resampling.LANCZOS).save(GROUND_OUT / "preview.png")
    write_json(GROUND_OUT / "manifest.json", manifest)
    print(GROUND_OUT / "deor_grounding_2x2_woreader.png")


def prepare_remove(_args):
    entries = {(row["dataset"], row["id"]): row for row in read_json(SOURCE / "candidates.json")["entries"]}
    tasks, selected_entries = [], {}
    for task_id, iid, slot_index, expected_category, _prompt in REMOVE_TASKS:
        entry = entries[("coco", iid)]
        slots = compact_thing_slots(entry)
        slot = slots[slot_index]
        if normalize_category(slot["category"]) != normalize_category(expected_category):
            raise ValueError(f"Generated category changed for {task_id}: {slot['category']}")
        selected_entries[iid] = entry
        tasks.append({
            "task_id": task_id,
            "mode": "remove",
            "target_id": iid,
            "target_slot": slot_index,
            "target_category": expected_category,
            "target_prediction_label": slot["prediction_label"],
            "donor_id": None,
            "donor_slot": None,
            "donor_category": None,
            "donor_prediction_label": None,
            "placement": "not applicable",
        })
    write_json(REMOVE_OUT / "tasks.json", {
        "checkpoint": str(CHECKPOINT), "seed": 1234, "steps": 10,
        "entries": selected_entries, "tasks": tasks,
        "selection": "Fixed replay of three previously selected removal examples",
    })
    print(REMOVE_OUT / "tasks.json")


def prepare_transfer(_args):
    TRANSFER_PAIRS.parent.mkdir(parents=True, exist_ok=True)
    TRANSFER_PAIRS.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in TRANSFER_ROWS) + "\n")
    print(TRANSFER_PAIRS)


def green_overlay(image, mask, alpha=.42):
    pixels = np.asarray(image).copy()
    pixels[mask] = np.rint((1-alpha)*pixels[mask] + alpha*np.array([50, 220, 90])).astype(np.uint8)
    return Image.fromarray(pixels)


def green_outline(image, mask):
    from scipy.ndimage import binary_erosion
    pixels = np.asarray(image).copy()
    edge = mask & ~binary_erosion(mask, iterations=2)
    pixels[edge] = [40, 230, 80]
    return Image.fromarray(pixels)


def compose_editing(_args):
    Image.init()
    panel, gap, block_gap, outer, title_h, prompt_h = 256, 6, 28, 8, 36, 48
    left_w, right_w = 3*panel+2*gap, 4*panel+3*gap
    width = outer*2 + left_w + block_gap + right_w
    row_h = title_h + panel + prompt_h
    height = outer*2 + 3*row_h
    figure = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(figure)
    title_font, prompt_font = font(22), font(23, True)
    manifest = {
        "checkpoint": str(CHECKPOINT),
        "layout": "3 rows; fixed Remove examples left and fixed Replace examples right",
        "controlled_replay": True,
        "remove_protocol": "owner removal with predicted-patch exclusion and Writer rewrite",
        "replace_protocol": "visual-memory owner transfer; visual_only condition",
        "freeform_instruction_following_claim": False,
        "examples": [],
    }
    left_titles = ["Source (A)", "Reconstruction", "Edited result"]
    right_titles = ["Source (A)", "Donor (B)", "Reconstruction", "Edited result"]
    for row_index, (remove, pair) in enumerate(zip(REMOVE_TASKS, TRANSFER_ROWS)):
        task_id, iid, _slot, _category, remove_prompt = remove
        remove_dir = REMOVE_OUT / "candidates/remove" / task_id
        remove_meta = read_json(remove_dir / "metadata.json")
        source = Image.open(remove_dir / "original.png").convert("RGB")
        source_cache = SOURCE / "cache/coco" / iid
        pred = np.load(source_cache / "ours.npz")["prediction"]
        selected_source = green_overlay(source, pred == remove_meta["target_prediction_label"])
        left_images = [selected_source,
                       Image.open(remove_dir / "reconstruction_native.png").convert("RGB"),
                       Image.open(remove_dir / "edited_native.png").convert("RGB")]
        y = outer + row_index*row_h
        left_record = {"mode": "remove", "task_id": task_id, "prompt": remove_prompt, "panels": []}
        for col, (title, current) in enumerate(zip(left_titles, left_images)):
            x = outer + col*(panel+gap)
            center_text(draw, title, (x, y, panel, title_h), title_font)
            figure.paste(current.resize((panel, panel), Image.Resampling.LANCZOS), (x, y+title_h))
            left_record["panels"].append({"title": title, "source_size": list(current.size),
                                           "box": [x, y+title_h, x+panel, y+title_h+panel]})
        center_text(draw, remove_prompt, (outer, y+title_h+panel, left_w, prompt_h), prompt_font, fill=(215, 25, 25))
        manifest["examples"].append(left_record)

        pair_dir = TRANSFER_OUT / "pairs" / pair["pair_id"]
        pair_meta = read_json(pair_dir / "metadata.json")
        a = Image.open(pair_dir / "source_A.png").convert("RGB")
        b = Image.open(pair_dir / "donor_B.png").convert("RGB")
        target_mask = np.asarray(Image.open(pair_dir / "target_prediction_mask.png"), bool)
        donor_mask = np.asarray(Image.open(pair_dir / "donor_prediction_mask.png"), bool)
        right_images = [green_outline(a, target_mask), green_outline(b, donor_mask),
                        Image.open(pair_dir / "reconstruction_native.png").convert("RGB"),
                        Image.open(pair_dir / "visual_only_native.png").convert("RGB")]
        right_x = outer + left_w + block_gap
        right_record = {"mode": "replace", "pair_id": pair["pair_id"],
                        "prompt": pair["instruction"], "condition": "visual_only", "panels": []}
        for col, (title, current) in enumerate(zip(right_titles, right_images)):
            x = right_x + col*(panel+gap)
            center_text(draw, title, (x, y, panel, title_h), title_font)
            figure.paste(current.resize((panel, panel), Image.Resampling.LANCZOS), (x, y+title_h))
            right_record["panels"].append({"title": title, "source_size": list(current.size),
                                            "box": [x, y+title_h, x+panel, y+title_h+panel]})
        center_text(draw, pair["instruction"], (right_x, y+title_h+panel, right_w, prompt_h),
                    prompt_font, fill=(215, 25, 25))
        right_record["diagnostics"] = pair_meta.get("change_diagnostics", {}).get("visual_only")
        manifest["examples"].append(right_record)
    EDIT_OUT.mkdir(parents=True, exist_ok=True)
    figure.save(EDIT_OUT / "object_editing_remove_replace_3x2_woreader.png", dpi=(300, 300))
    figure.save(EDIT_OUT / "object_editing_remove_replace_3x2_woreader.pdf", resolution=300)
    figure.resize((1200, round(height*1200/width)), Image.Resampling.LANCZOS).save(EDIT_OUT / "preview.jpg", quality=95)
    write_json(EDIT_OUT / "manifest.json", manifest)
    print(EDIT_OUT / "object_editing_remove_replace_3x2_woreader.png")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True, choices=[
        "prepare_source", "compose_segmentation", "compose_grounding",
        "prepare_remove", "prepare_transfer", "compose_editing", "compose_non_editing",
    ])
    args = parser.parse_args()
    if args.stage == "compose_non_editing":
        compose_segmentation(args)
        compose_grounding(args)
    else:
        globals()[args.stage](args)


if __name__ == "__main__":
    main()
