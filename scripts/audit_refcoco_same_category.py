#!/usr/bin/env python3
"""Create a human-checkable contact sheet for same-category distractors."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pycocotools.coco import COCO


def font(size: int):
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except OSError:
        return ImageFont.load_default()


def image_path(root: Path, image_id: int, file_name: str) -> Path:
    candidates = [
        root / "train2017" / f"{image_id:012d}.jpg",
        root / "val2017" / f"{image_id:012d}.jpg",
        root / "train2014" / file_name,
        root / "val2014" / file_name,
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(candidates)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", type=Path, default=Path("outputs/analysis_refcoco_cached_grounding/per_expression.jsonl"))
    parser.add_argument("--annotations", default="/home/jovyan/data/coco/refcoco/instances.json")
    parser.add_argument("--image_root", type=Path, default=Path("/home/jovyan/data/coco"))
    parser.add_argument("--output", type=Path, default=Path("outputs/analysis_refcoco_cached_grounding/same_category_audit"))
    parser.add_argument("--count", type=int, default=20)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    rows = [json.loads(line) for line in args.records.open()]
    # Avoid repeated sentences about the same target annotation.
    unique = {}
    for row in rows:
        if row.get("same_category_gt_competitors", 0) > 0:
            unique.setdefault(int(row["annotation_id"]), row)
    rng = random.Random(args.seed)
    chosen = rng.sample(list(unique.values()), min(args.count, len(unique)))
    coco = COCO(args.annotations)
    args.output.mkdir(parents=True, exist_ok=True)
    tiles, audited = [], []
    for index, row in enumerate(chosen):
        target = coco.anns[int(row["annotation_id"])]
        category = int(target["category_id"])
        distractors = [
            ann for ann in coco.loadAnns(coco.getAnnIds(imgIds=[target["image_id"]], catIds=[category], iscrowd=False))
            if int(ann["id"]) != int(target["id"]) and coco.annToMask(ann).any()
        ]
        if not distractors:
            raise AssertionError(f"No distractor after excluding target {target['id']}")
        if any(int(ann["id"]) == int(target["id"]) for ann in distractors):
            raise AssertionError("Target leaked into distractor set")
        info = coco.imgs[int(target["image_id"])]
        path = image_path(args.image_root, int(target["image_id"]), info["file_name"])
        with Image.open(path) as handle:
            image = handle.convert("RGB")
        array = np.asarray(image).copy().astype(np.float32)
        distractor_union = np.zeros((image.height, image.width), dtype=bool)
        for ann in distractors:
            distractor_union |= coco.annToMask(ann).astype(bool)
        target_mask = coco.annToMask(target).astype(bool)
        array[distractor_union] = 0.50 * array[distractor_union] + 0.50 * np.array([30, 190, 255])
        array[target_mask] = 0.35 * array[target_mask] + 0.65 * np.array([255, 50, 50])
        overlay = Image.fromarray(array.clip(0, 255).astype(np.uint8)).resize((320, 240), Image.Resampling.BILINEAR)
        tile = Image.new("RGB", (320, 300), "white")
        tile.paste(overlay, (0, 0))
        draw = ImageDraw.Draw(tile)
        name = coco.cats[category]["name"]
        text = f"{index+1}. {name}: target red, {len(distractors)} distractor(s) cyan\n{row['shared_text'][:58]}"
        draw.multiline_text((7, 246), text, fill="black", font=font(14), spacing=2)
        tiles.append(tile)
        audited.append({
            "annotation_id": int(target["id"]), "image_id": int(target["image_id"]),
            "category_id": category, "category": name,
            "distractor_ids": [int(ann["id"]) for ann in distractors],
            "target_excluded": True, "expression": row["shared_text"],
        })
    columns = 4
    rows_count = (len(tiles) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * 320, rows_count * 300), (235, 235, 235))
    for index, tile in enumerate(tiles):
        sheet.paste(tile, ((index % columns) * 320, (index // columns) * 300))
    sheet.save(args.output / "contact_sheet.png")
    (args.output / "audit.json").write_text(json.dumps({
        "definition": "same image/category, noncrowd nonempty GT, target ann_id explicitly excluded",
        "seed": args.seed, "count": len(audited), "samples": audited,
    }, indent=2) + "\n")
    print(f"PASS: {len(audited)} cases; {args.output / 'contact_sheet.png'}")


if __name__ == "__main__":
    main()
