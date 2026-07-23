#!/usr/bin/env python
"""Create exhaustive, thing-only PGOT manifests from Pix2Cap-COCO.

This writes JSONL references only; JPEGs and panoptic PNGs are never copied or
modified.  A sample is retained only when every crop-visible COCO thing has a
usable Pix2Cap caption and the complete object/OVT sequence fits the configured
object and token budgets.  Stuff is deliberately omitted and becomes the
residual register target.
"""

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

from transformers import AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pgot.constants import NEW_SPECIAL_TOKENS
from preprocess.prepare_lviscap_pgot import (
    bbox_crop_intersection,
    clean_caption,
    invalid_caption_reason,
)


def build_caption(segments, n_ovt_per_object):
    counts = Counter()
    ovts = "<ovt>" * n_ovt_per_object
    chunks = []
    for segment in segments:
        category = str(segment["category"])
        counts[category] += 1
        label = f"{category.capitalize()} {counts[category]}"
        description = str(segment["description"]).rstrip(".").rstrip()
        chunks.append(f"<thing> {label}: {description}. {ovts}.")
    return " ".join(chunks) + " <scene_end>"


def build_manifest(args):
    source_path = Path(
        args.input
        or Path(args.coco_root) / f"pix2cap_coco_{args.split}.json"
    )
    output_path = Path(args.output)
    stats_path = Path(args.stats_output or f"{args.output}.stats.json")
    image_root = Path(args.coco_root) / f"{args.split}2017"
    panoptic_root = Path(args.coco_root) / "annotations" / f"panoptic_{args.split}2017"

    print(f"Loading {source_path} ...", flush=True)
    with source_path.open() as handle:
        source = json.load(handle)
    images = {int(image["id"]): image for image in source["images"]}
    thing_ids = {
        int(category["id"])
        for category in source["categories"]
        if int(category.get("isthing", 0)) == 1
    }
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, use_fast=False)
    tokenizer.add_special_tokens({"additional_special_tokens": NEW_SPECIAL_TOKENS})

    stats = Counter()
    invalid_reasons = Counter()
    object_histogram = Counter()
    token_histogram = Counter()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(output_path.name + ".tmp")
    with temporary_path.open("w") as output:
        for annotation in source["annotations"]:
            stats["source_images"] += 1
            image_id = int(annotation["image_id"])
            image = images[image_id]
            image_path = image_root / image["file_name"]
            panoptic_name = annotation.get("file_name") or f"{image_id:012d}.png"
            panoptic_path = panoptic_root / panoptic_name
            if not image_path.exists() or not panoptic_path.exists():
                stats["drop_missing_file"] += 1
                continue

            visible = []
            invalid = None
            for segment in annotation.get("segments_info", []):
                if int(segment.get("category_id", -1)) not in thing_ids:
                    stats["stuff_segments_removed"] += 1
                    continue
                is_visible, crop_area, crop_x, crop_y = bbox_crop_intersection(
                    segment.get("bbox", (0, 0, 0, 0)),
                    int(image["width"]),
                    int(image["height"]),
                    args.crop_size,
                )
                if not is_visible:
                    stats["thing_segments_outside_crop"] += 1
                    continue
                description = clean_caption(segment.get("description"))
                reason = invalid_caption_reason(description)
                if reason is not None:
                    invalid = reason
                    invalid_reasons[reason] += 1
                    break
                enriched = {
                    "segment_id": int(segment["id"]),
                    "category_id": int(segment["category_id"]),
                    "category": str(segment.get("category", "object")),
                    "description": description,
                    "caption_source": "pix2cap",
                    "is_thing": True,
                    "bbox": segment.get("bbox", []),
                    "area": float(segment.get("area", 0.0)),
                    "visible_crop_area_bbox": float(crop_area),
                    "crop_x_bbox": float(crop_x),
                    "crop_y_bbox": float(crop_y),
                }
                visible.append(enriched)

            if invalid is not None:
                stats["drop_invalid_visible_caption"] += 1
                continue
            if not visible:
                stats["drop_no_visible_thing"] += 1
                continue
            if len(visible) > args.max_objects:
                stats["drop_over_max_objects"] += 1
                continue
            visible.sort(
                key=lambda segment: (
                    -segment["visible_crop_area_bbox"],
                    segment["crop_x_bbox"],
                    segment["crop_y_bbox"],
                    segment["segment_id"],
                )
            )
            caption = build_caption(visible, args.n_ovt_per_object)
            caption_tokens = len(tokenizer.encode(caption, add_special_tokens=False))
            if caption_tokens > args.max_caption_tokens:
                stats["drop_over_caption_tokens"] += 1
                continue

            record = {
                "source_dataset": "pix2cap_thing_only",
                "split": args.split,
                "image_id": image_id,
                "image_path": str(image_path),
                "panoptic_mask_path": str(panoptic_path),
                "height": int(image["height"]),
                "width": int(image["width"]),
                "caption": caption,
                "segments": visible,
                "n_objects": len(visible),
            }
            output.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
            stats["images_written"] += 1
            stats["things_written"] += len(visible)
            object_histogram[len(visible)] += 1
            token_histogram[(caption_tokens // 100) * 100] += 1
            if args.max_images is not None and stats["images_written"] >= args.max_images:
                stats["stopped_at_max_images"] = 1
                break

    os.replace(temporary_path, output_path)
    result = {
        "input": str(source_path.resolve()),
        "output": str(output_path.resolve()),
        "split": args.split,
        "policy": "all crop-visible things have Pix2Cap captions and OVTs; stuff -> registers",
        "crop_size": args.crop_size,
        "max_objects": args.max_objects,
        "max_caption_tokens": args.max_caption_tokens,
        "n_ovt_per_object": args.n_ovt_per_object,
        "stats": dict(sorted(stats.items())),
        "invalid_caption_reasons": dict(sorted(invalid_reasons.items())),
        "object_count_histogram": {str(k): v for k, v in sorted(object_histogram.items())},
        "caption_token_histogram": {str(k): v for k, v in sorted(token_histogram.items())},
    }
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    with stats_path.open("w") as handle:
        json.dump(result, handle, indent=2)
        handle.write("\n")
    print(json.dumps(result, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--coco_root", default="/home/jovyan/data/coco")
    parser.add_argument("--split", choices=("train", "val"), required=True)
    parser.add_argument("--input", default=None)
    parser.add_argument("--output", required=True)
    parser.add_argument("--stats_output", default=None)
    parser.add_argument(
        "--tokenizer_path", default="/home/jovyan/data/Scale-RAE-Qwen1.5B_DiT2.4B"
    )
    parser.add_argument("--crop_size", type=int, default=512)
    parser.add_argument("--max_objects", type=int, default=50)
    parser.add_argument("--max_caption_tokens", type=int, default=1024)
    parser.add_argument("--n_ovt_per_object", type=int, default=1)
    parser.add_argument("--max_images", type=int, default=None)
    args = parser.parse_args()
    build_manifest(args)


if __name__ == "__main__":
    main()
