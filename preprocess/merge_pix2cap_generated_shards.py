#!/usr/bin/env python
"""Validate and merge resumable Pix2Cap prediction shards.

The merged file follows the released Pix2Cap COCO JSON schema closely enough
to be consumed by prepare_pix2cap_thing_pgot.py. It deliberately keeps all
images, including images with no predicted segments; the subsequent PGOT
manifest reports how many have at least one usable crop-visible thing.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard_root", required=True)
    parser.add_argument("--instances_json", required=True)
    parser.add_argument("--panoptic_categories_json", required=True)
    parser.add_argument("--mask_root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary_output", default=None)
    parser.add_argument("--expected_images", type=int, default=5000)
    parser.add_argument("--max_images", type=int, default=None)
    return parser.parse_args()


def rgb_to_id(rgb: np.ndarray) -> np.ndarray:
    return (
        rgb[..., 0].astype(np.int64)
        + 256 * rgb[..., 1].astype(np.int64)
        + 256 * 256 * rgb[..., 2].astype(np.int64)
    )


def main() -> None:
    args = parse_args()
    shard_paths = sorted(Path(args.shard_root).glob("predictions_shard_*.jsonl"))
    if not shard_paths:
        raise FileNotFoundError(f"No prediction shards found in {args.shard_root}")

    with Path(args.instances_json).open(encoding="utf-8") as handle:
        instances = json.load(handle)
    with Path(args.panoptic_categories_json).open(encoding="utf-8") as handle:
        panoptic_source = json.load(handle)
    expected_images = sorted(instances["images"], key=lambda image: int(image["id"]))
    if args.max_images is not None:
        expected_images = expected_images[: args.max_images]
    expected_by_id = {int(image["id"]): image for image in expected_images}
    records = {}
    duplicate_ids = []
    malformed = []
    for shard_path in shard_paths:
        with shard_path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                record = json.loads(line)
                image_id = int(record["image_id"])
                if image_id in records:
                    old = records[image_id]
                    if old.get("status") == "ok" and record.get("status") == "ok":
                        duplicate_ids.append(image_id)
                        continue
                    if record.get("status") == "ok":
                        records[image_id] = record
                else:
                    records[image_id] = record
                if image_id not in expected_by_id:
                    malformed.append(
                        f"{shard_path.name}:{line_number}: unknown image_id={image_id}"
                    )

    ok_records = {
        image_id: record
        for image_id, record in records.items()
        if record.get("status") == "ok"
    }
    error_records = {
        image_id: record
        for image_id, record in records.items()
        if record.get("status") != "ok"
    }
    missing_ids = sorted(set(expected_by_id) - set(records))
    if duplicate_ids or malformed or missing_ids or error_records:
        details = {
            "duplicate_success_ids": duplicate_ids[:20],
            "malformed": malformed[:20],
            "missing_ids": missing_ids[:20],
            "error_ids": sorted(error_records)[:20],
        }
        raise RuntimeError(f"Prediction shards are incomplete or invalid: {details}")
    if len(ok_records) != args.expected_images:
        raise RuntimeError(
            f"Expected {args.expected_images} successful images, got {len(ok_records)}"
        )

    mask_root = Path(args.mask_root)
    stats = Counter()
    annotations = []
    for image_id in sorted(ok_records):
        record = ok_records[image_id]
        expected = expected_by_id[image_id]
        mask_path = mask_root / record["mask_file_name"]
        if not mask_path.exists():
            raise FileNotFoundError(mask_path)
        with Image.open(mask_path) as mask_handle:
            rgb = np.asarray(mask_handle.convert("RGB"))
        if rgb.shape[:2] != (int(expected["height"]), int(expected["width"])):
            raise RuntimeError(
                f"Mask shape mismatch for image_id={image_id}: {rgb.shape[:2]} vs "
                f"{(expected['height'], expected['width'])}"
            )
        id_map = rgb_to_id(rgb)
        segment_ids = {int(segment["id"]) for segment in record["segments_info"]}
        mask_ids = set(int(value) for value in np.unique(id_map) if int(value) != 0)
        if mask_ids != segment_ids:
            raise RuntimeError(
                f"Segment/mask id mismatch for image_id={image_id}: "
                f"json={sorted(segment_ids)}, mask={sorted(mask_ids)}"
            )
        for segment in record["segments_info"]:
            actual_area = int((id_map == int(segment["id"])).sum())
            if actual_area != int(segment["area"]):
                raise RuntimeError(
                    f"Area mismatch image_id={image_id}, segment={segment['id']}: "
                    f"json={segment['area']}, mask={actual_area}"
                )
            stats["thing_segments" if segment["isthing"] else "stuff_segments"] += 1
            if not str(segment.get("description", "")).strip():
                stats["empty_captions"] += 1
        has_thing = any(segment["isthing"] for segment in record["segments_info"])
        stats["images"] += 1
        stats["images_with_things"] += int(has_thing)
        stats["images_without_things"] += int(not has_thing)
        annotations.append(
            {
                "image_id": image_id,
                "file_name": str(record["mask_file_name"]),
                "segments_info": record["segments_info"],
            }
        )

    merged = {
        "info": {
            "description": "Pix2Cap model-generated dense captions and panoptic masks",
            "source_shards": [str(path.resolve()) for path in shard_paths],
        },
        "images": [expected_by_id[image_id] for image_id in sorted(expected_by_id)],
        "annotations": annotations,
        "categories": panoptic_source["categories"],
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f".{output_path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(merged, handle, ensure_ascii=False, separators=(",", ":"))
        handle.write("\n")
    os.replace(temporary, output_path)

    summary = {
        "output": str(output_path.resolve()),
        "mask_root": str(mask_root.resolve()),
        "shards": [str(path.resolve()) for path in shard_paths],
        "expected_images": args.expected_images,
        "successful_images": len(ok_records),
        "stats": dict(sorted(stats.items())),
    }
    summary_path = Path(args.summary_output or f"{args.output}.summary.json")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
