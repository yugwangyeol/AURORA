#!/usr/bin/env python3
"""Audit rich-caption coverage from Pix2Cap and the RefCOCO family.

COCO instance annotations are the canonical object inventory. Pix2Cap
panoptic things are matched to COCO instances by image/category and greedy
bbox IoU. RefCOCO-family expressions already carry the canonical COCO
``ann_id``. The script only writes a JSON report; it does not create or modify
training manifests, images, or masks.
"""

import argparse
import json
import math
import pickle
import re
from collections import Counter, defaultdict
from pathlib import Path


REFUSAL_PATTERNS = (
    re.compile(r"^(?:i(?:'m| am)\s+)?sorry\b", re.IGNORECASE),
    re.compile(
        r"\b(?:cannot|can't|could not|couldn't|unable to)\s+"
        r"(?:find|identify|see|locate|detect|describe|generate)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:there (?:is|are) no|no visible|not visible|does(?:n't| not) "
        r"seem to be|do(?:es)? not appear)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:bounding box|query object|queried object|specified object|"
        r"highlighted object|current frame|provide (?:a )?(?:different|another) "
        r"(?:image|query))\b",
        re.IGNORECASE,
    ),
)


def clean_caption(value):
    return " ".join(str(value or "").strip().split())


def valid_caption(value):
    text = clean_caption(value)
    return bool(text) and not any(pattern.search(text) for pattern in REFUSAL_PATTERNS)


def bbox_iou(a, b):
    ax, ay, aw, ah = (float(v) for v in a)
    bx, by, bw, bh = (float(v) for v in b)
    ax2, ay2 = ax + aw, ay + ah
    bx2, by2 = bx + bw, by + bh
    inter = max(0.0, min(ax2, bx2) - max(ax, bx)) * max(
        0.0, min(ay2, by2) - max(ay, by)
    )
    union = max(aw * ah + bw * bh - inter, 0.0)
    return inter / union if union > 0.0 else 0.0


def crop_visible(bbox, width, height, crop_size):
    x, y, w, h = (float(v) for v in bbox)
    if w <= 0.0 or h <= 0.0:
        return False
    scale = max(float(crop_size) / float(height), float(crop_size) / float(width))
    resized_w = float(round(width * scale))
    resized_h = float(round(height * scale))
    left = max((resized_w - crop_size) // 2, 0.0)
    top = max((resized_h - crop_size) // 2, 0.0)
    right, bottom = left + crop_size, top + crop_size
    x0, y0 = x * scale, y * scale
    x1, y1 = (x + w) * scale, (y + h) * scale
    return min(x1, right) > max(x0, left) and min(y1, bottom) > max(y0, top)


def load_ref_objects(coco_root, allowed_splits):
    specs = (
        ("refcoco", "refs(unc).p"),
        ("refcoco+", "refs(unc).p"),
        ("refcocog", "refs(umd).p"),
    )
    by_dataset = {}
    union = set()
    expressions = defaultdict(list)
    split_histograms = {}
    for dataset, filename in specs:
        path = coco_root / dataset / filename
        with path.open("rb") as handle:
            refs = pickle.load(handle)
        split_histograms[dataset] = dict(
            sorted(Counter(str(ref.get("split", "")) for ref in refs).items())
        )
        selected = set()
        for ref in refs:
            if allowed_splits and str(ref.get("split", "")) not in allowed_splits:
                continue
            ann_id = int(ref["ann_id"])
            selected.add(ann_id)
            for sentence in ref.get("sentences", []):
                text = clean_caption(sentence.get("sent") or sentence.get("raw"))
                if text:
                    expressions[ann_id].append((dataset, text))
        by_dataset[dataset] = selected
        union.update(selected)
    return by_dataset, union, expressions, split_histograms


def load_current_manifest(path):
    image_ids = set()
    object_count = 0
    if path is None or not path.exists():
        return image_ids, object_count
    with path.open() as handle:
        for line in handle:
            record = json.loads(line)
            image_ids.add(int(record["image_id"]))
            object_count += int(record.get("n_objects", len(record.get("segments", []))))
    return image_ids, object_count


def audit_split(args, split):
    coco_root = Path(args.coco_root)
    instances_path = coco_root / "annotations" / f"instances_{split}2017.json"
    pix2cap_path = coco_root / f"pix2cap_coco_{split}.json"
    current_manifest = (
        Path(args.current_train_manifest)
        if split == "train"
        else Path(args.current_val_manifest)
    )

    with instances_path.open() as handle:
        instances = json.load(handle)
    images = {int(image["id"]): image for image in instances["images"]}
    anns = {int(ann["id"]): ann for ann in instances["annotations"]}
    anns_by_image_category = defaultdict(list)
    visible_ann_ids = set()
    visible_by_image = defaultdict(set)
    for ann_id, ann in anns.items():
        image = images[int(ann["image_id"])]
        if crop_visible(
            ann.get("bbox", (0, 0, 0, 0)),
            int(image["width"]),
            int(image["height"]),
            args.crop_size,
        ):
            visible_ann_ids.add(ann_id)
            visible_by_image[int(ann["image_id"])].add(ann_id)
            anns_by_image_category[
                (int(ann["image_id"]), int(ann["category_id"]))
            ].append(ann)

    ref_by_dataset, ref_union_raw, expressions, ref_split_histograms = load_ref_objects(
        coco_root,
        set(args.ref_splits.split(",")) if split == "train" else set(),
    )
    ref_union = ref_union_raw & visible_ann_ids
    ref_by_dataset_visible = {
        name: values & visible_ann_ids for name, values in ref_by_dataset.items()
    }

    pix_valid_ids = set()
    pix_all_matched_ids = set()
    pix_image_ids = set()
    match_ious = []
    match_stats = Counter()
    if pix2cap_path.exists():
        with pix2cap_path.open() as handle:
            pix = json.load(handle)
        pix_images = {int(image["id"]): image for image in pix["images"]}
        thing_ids = {
            int(category["id"])
            for category in pix["categories"]
            if int(category.get("isthing", 0)) == 1
        }
        for annotation in pix["annotations"]:
            image_id = int(annotation["image_id"])
            if image_id not in images:
                continue
            pix_image_ids.add(image_id)
            used = set()
            segments = [
                segment
                for segment in annotation.get("segments_info", [])
                if int(segment.get("category_id", -1)) in thing_ids
                and crop_visible(
                    segment.get("bbox", (0, 0, 0, 0)),
                    int(pix_images[image_id]["width"]),
                    int(pix_images[image_id]["height"]),
                    args.crop_size,
                )
            ]
            segments.sort(
                key=lambda segment: float(segment.get("area", 0.0)),
                reverse=True,
            )
            for segment in segments:
                candidates = anns_by_image_category.get(
                    (image_id, int(segment["category_id"])),
                    (),
                )
                scored = sorted(
                    (
                        (
                            bbox_iou(
                                segment.get("bbox", (0, 0, 0, 0)),
                                candidate.get("bbox", (0, 0, 0, 0)),
                            ),
                            int(candidate["id"]),
                        )
                        for candidate in candidates
                        if int(candidate["id"]) not in used
                    ),
                    reverse=True,
                )
                if not scored or scored[0][0] < args.min_bbox_iou:
                    match_stats["unmatched_pix2cap_segments"] += 1
                    continue
                iou, ann_id = scored[0]
                used.add(ann_id)
                pix_all_matched_ids.add(ann_id)
                match_ious.append(iou)
                match_stats["matched_pix2cap_segments"] += 1
                if valid_caption(segment.get("description")):
                    pix_valid_ids.add(ann_id)
                else:
                    match_stats["matched_invalid_pix2cap_caption"] += 1

    current_images, current_objects = load_current_manifest(current_manifest)
    union_ids = pix_valid_ids | ref_union
    overlap_ids = pix_valid_ids & ref_union

    def image_coverage(captioned_ids):
        any_images = 0
        complete_images = 0
        nonempty_images = 0
        for object_ids in visible_by_image.values():
            if not object_ids:
                continue
            nonempty_images += 1
            hit = object_ids & captioned_ids
            any_images += int(bool(hit))
            complete_images += int(object_ids.issubset(captioned_ids))
        return {
            "crop_visible_nonempty_images": nonempty_images,
            "images_with_any_captioned_object": any_images,
            "images_with_all_visible_objects_captioned": complete_images,
        }

    ref_images = {
        int(anns[ann_id]["image_id"]) for ann_id in ref_union if ann_id in anns
    }
    union_images = {
        int(anns[ann_id]["image_id"]) for ann_id in union_ids if ann_id in anns
    }
    denominator = max(len(visible_ann_ids), 1)
    return {
        "split": split,
        "crop_size": args.crop_size,
        "coco": {
            "images": len(images),
            "crop_visible_instances": len(visible_ann_ids),
        },
        "current_strict_pix2cap_manifest": {
            "path": str(current_manifest),
            "images": len(current_images),
            "objects": current_objects,
        },
        "pix2cap_source": {
            "source_images_in_coco_split": len(pix_image_ids),
            "matched_visible_instances": len(pix_all_matched_ids),
            "valid_caption_instances": len(pix_valid_ids),
            "visible_instance_coverage": len(pix_valid_ids) / denominator,
            "bbox_match_iou_mean": (
                sum(match_ious) / len(match_ious) if match_ious else None
            ),
            "bbox_match_iou_min": min(match_ious) if match_ious else None,
            "match_stats": dict(sorted(match_stats.items())),
            **image_coverage(pix_valid_ids),
        },
        "refcoco_family": {
            "selected_ref_splits": (
                sorted(set(args.ref_splits.split(","))) if split == "train" else "all"
            ),
            "split_histograms": ref_split_histograms,
            "visible_unique_instances_by_dataset": {
                name: len(values)
                for name, values in sorted(ref_by_dataset_visible.items())
            },
            "visible_unique_instances_union": len(ref_union),
            "visible_instance_coverage": len(ref_union) / denominator,
            "images": len(ref_images),
            "expressions": sum(len(expressions[ann_id]) for ann_id in ref_union),
            **image_coverage(ref_union),
        },
        "pix2cap_refcoco_union": {
            "visible_unique_instances": len(union_ids),
            "visible_instance_coverage": len(union_ids) / denominator,
            "overlap_instances": len(overlap_ids),
            "added_by_refcoco_over_pix2cap": len(ref_union - pix_valid_ids),
            "added_by_pix2cap_over_refcoco": len(pix_valid_ids - ref_union),
            "images": len(union_images),
            "strict_manifest_plus_refcoco_image_union": len(
                current_images | ref_images
            ),
            **image_coverage(union_ids),
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--coco_root", default="/home/jovyan/data/coco")
    parser.add_argument(
        "--current_train_manifest",
        default="/home/jovyan/PGOT/data/pgot_pix2cap_thing_train.jsonl",
    )
    parser.add_argument(
        "--current_val_manifest",
        default="/home/jovyan/PGOT/data/pgot_pix2cap_thing_val.jsonl",
    )
    parser.add_argument("--crop_size", type=int, default=512)
    parser.add_argument("--min_bbox_iou", type=float, default=0.5)
    parser.add_argument(
        "--ref_splits",
        default="train",
        help="Comma-separated RefCOCO splits allowed for the training audit.",
    )
    parser.add_argument("--splits", default="train,val")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    report = {
        "policy": (
            "COCO instances are canonical; Pix2Cap matches by image/category/bbox "
            "IoU and RefCOCO-family captions match by ann_id."
        ),
        "min_bbox_iou": args.min_bbox_iou,
        "splits": [
            audit_split(args, split.strip())
            for split in args.splits.split(",")
            if split.strip()
        ],
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
