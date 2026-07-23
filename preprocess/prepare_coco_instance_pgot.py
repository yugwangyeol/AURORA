#!/usr/bin/env python
"""Build a compact COCO-instance PGOT manifest.

The manifest deliberately contains *things only*.  Every visible non-crowd
COCO instance becomes exactly one caption chunk and one OVT.  RefCOCO-family
expressions enrich train2017 objects when available; all remaining instances
use a deterministic category-only description.  val2017 has no RefCOCO-family
annotations, so its captions are category-only while its masks remain the
official COCO instance masks.

The expensive COCO JSON is read once here instead of once in every dataloader
worker.  Segmentation polygons/RLE are copied into JSONL and decoded lazily by
the training dataset.
"""

import argparse
import json
import os
import pickle
from collections import defaultdict
from typing import Dict, Iterable, Optional, Tuple


REF_SPECS = (
    # Low to high priority.  Later datasets overwrite earlier expressions for
    # the same ann_id, yielding RefCOCOg > RefCOCO+ > RefCOCO.
    ("refcoco", "refs(unc).p"),
    ("refcoco+", "refs(unc).p"),
    ("refcocog", "refs(umd).p"),
)


def _clean_expression(text: str) -> str:
    return " ".join(str(text).strip().rstrip(".").split()).lower()


def load_ref_expressions(coco_root: str) -> Dict[int, Tuple[str, str]]:
    by_ann: Dict[int, Tuple[str, str]] = {}
    for dataset_name, filename in REF_SPECS:
        path = os.path.join(coco_root, dataset_name, filename)
        if not os.path.exists(path):
            continue
        with open(path, "rb") as f:
            refs = pickle.load(f)
        for ref in refs:
            sentences = ref.get("sentences") or []
            if not sentences:
                continue
            text = _clean_expression(sentences[0].get("sent", sentences[0].get("raw", "")))
            if text:
                by_ann[int(ref["ann_id"])] = (text, dataset_name)
    return by_ann


def _crop_window(width: int, height: int, crop_size: int) -> Tuple[float, float, float, float, float]:
    scale = max(float(crop_size) / float(height), float(crop_size) / float(width))
    resized_w = float(round(width * scale))
    resized_h = float(round(height * scale))
    left = max((resized_w - crop_size) // 2, 0.0)
    top = max((resized_h - crop_size) // 2, 0.0)
    return scale, left, top, left + crop_size, top + crop_size


def _bbox_visible_after_crop(
    bbox: Iterable[float], width: int, height: int, crop_size: int
) -> bool:
    x, y, w, h = (float(v) for v in bbox)
    if w <= 0 or h <= 0:
        return False
    scale, left, top, right, bottom = _crop_window(width, height, crop_size)
    x0, y0 = x * scale, y * scale
    x1, y1 = (x + w) * scale, (y + h) * scale
    return min(x1, right) > max(x0, left) and min(y1, bottom) > max(y0, top)


def build_manifest(args) -> dict:
    split = args.split
    ann_path = args.instances_json or os.path.join(
        args.coco_root, "annotations", f"instances_{split}2017.json"
    )
    image_root = args.image_root or os.path.join(args.coco_root, f"{split}2017")
    with open(ann_path) as f:
        coco = json.load(f)

    categories = {int(c["id"]): str(c["name"]) for c in coco["categories"]}
    images = {int(im["id"]): im for im in coco["images"]}
    annotations = defaultdict(list)
    for ann in coco["annotations"]:
        if int(ann.get("iscrowd", 0)) != 0:
            continue
        im = images[int(ann["image_id"])]
        if not _bbox_visible_after_crop(
            ann.get("bbox", (0, 0, 0, 0)), int(im["width"]), int(im["height"]), args.crop_size
        ):
            continue
        annotations[int(ann["image_id"])].append(ann)

    refs = load_ref_expressions(args.coco_root) if split == "train" and not args.category_only else {}
    stats = defaultdict(int)
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as out:
        for image_id in sorted(images):
            if args.max_images is not None and stats["images_written"] >= args.max_images:
                break
            im = images[image_id]
            anns = annotations.get(image_id, [])
            if not anns:
                stats["drop_no_visible_instance"] += 1
                continue
            # Do not silently turn overflow objects into background registers.
            if len(anns) > args.max_objects:
                stats["drop_over_max_objects"] += 1
                continue

            segments = []
            for ann in anns:
                ann_id = int(ann["id"])
                category = categories[int(ann["category_id"])]
                if ann_id in refs:
                    description, caption_source = refs[ann_id]
                    stats[f"caption_{caption_source}"] += 1
                else:
                    article = "an" if category[:1].lower() in "aeiou" else "a"
                    description = f"{article} {category}"
                    caption_source = "category"
                    stats["caption_category"] += 1
                segments.append(
                    {
                        "ann_id": ann_id,
                        "category_id": int(ann["category_id"]),
                        "category": category,
                        "description": description,
                        "caption_source": caption_source,
                        "segmentation": ann["segmentation"],
                        "bbox": ann["bbox"],
                        "area": float(ann.get("area", 0.0)),
                    }
                )

            record = {
                "dataset_type": "coco_instance",
                "split": split,
                "image_id": image_id,
                "image_path": os.path.join(image_root, im["file_name"]),
                "height": int(im["height"]),
                "width": int(im["width"]),
                "segments": segments,
            }
            out.write(json.dumps(record, separators=(",", ":")) + "\n")
            stats["images_written"] += 1
            stats["instances_written"] += len(segments)

    result = {
        "output": os.path.abspath(args.output),
        "split": split,
        "crop_size": args.crop_size,
        "max_objects": args.max_objects,
        **dict(sorted(stats.items())),
    }
    print(json.dumps(result, indent=2))
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--coco_root", default="/home/jovyan/data/coco")
    p.add_argument("--split", choices=["train", "val"], required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--instances_json", default=None)
    p.add_argument("--image_root", default=None)
    p.add_argument("--crop_size", type=int, default=512)
    p.add_argument("--max_objects", type=int, default=50)
    p.add_argument("--category_only", action="store_true")
    p.add_argument("--max_images", type=int, default=None, help="Testing-only output cap")
    args = p.parse_args()
    build_manifest(args)


if __name__ == "__main__":
    main()
