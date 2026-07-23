#!/usr/bin/env python
"""Build a clean PGOT manifest from LVIScap annotations.

No image or mask raster is generated.  The output JSONL references the
existing COCO JPEGs and carries the original LVIS polygon annotations.  A
whole image is rejected whenever a visible annotation cannot become an OVT;
this prevents omitted foreground objects from silently becoming register
background.

Default clean policy:
  * LVIScap train only;
  * reject images with non-exhaustive category annotations;
  * ignore annotations fully outside the CODA 512 center crop;
  * reject the image if any crop-visible caption is empty or a VLM refusal;
  * reject images with more than 50 crop-visible instances;
  * reject images whose complete caption/OVT sequence exceeds 1024 tokens;
  * reject any image present in COCO val2017.
"""

import argparse
import json
import os
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from transformers import AutoTokenizer


_REFUSAL_PATTERNS: Sequence[Tuple[str, re.Pattern]] = (
    ("apology", re.compile(r"^(?:i(?:'m| am)\s+)?sorry\b", re.IGNORECASE)),
    (
        "cannot_describe",
        re.compile(
            r"\b(?:cannot|can't|could not|couldn't|unable to)\s+"
            r"(?:find|identify|see|locate|detect|describe|generate)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "object_not_visible",
        re.compile(
            r"\b(?:there (?:is|are) no|no visible|not visible|does(?:n't| not) "
            r"seem to be|do(?:es)? not appear)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "prompt_leak",
        re.compile(
            r"\b(?:bounding box|query object|queried object|specified object|"
            r"highlighted object|current frame|provide (?:a )?(?:different|another) "
            r"(?:image|query))\b",
            re.IGNORECASE,
        ),
    ),
)


def clean_caption(value: object) -> str:
    """Normalize whitespace while preserving the released sentence."""
    return " ".join(str(value or "").strip().split())


def invalid_caption_reason(caption: str) -> Optional[str]:
    if not caption:
        return "empty_caption"
    for name, pattern in _REFUSAL_PATTERNS:
        if pattern.search(caption):
            return name
    return None


def crop_window(width: int, height: int, crop_size: int) -> Tuple[float, float, float, float, float]:
    scale = max(float(crop_size) / float(height), float(crop_size) / float(width))
    resized_w = float(round(width * scale))
    resized_h = float(round(height * scale))
    left = max((resized_w - crop_size) // 2, 0.0)
    top = max((resized_h - crop_size) // 2, 0.0)
    return scale, left, top, left + crop_size, top + crop_size


def bbox_crop_intersection(
    bbox: Iterable[float], width: int, height: int, crop_size: int
) -> Tuple[bool, float, float, float]:
    """Return visibility, approximate visible area, crop-x, and crop-y."""
    x, y, w, h = (float(v) for v in bbox)
    if w <= 0 or h <= 0:
        return False, 0.0, 0.0, 0.0
    scale, left, top, right, bottom = crop_window(width, height, crop_size)
    x0, y0 = x * scale, y * scale
    x1, y1 = (x + w) * scale, (y + h) * scale
    ix0, iy0 = max(x0, left), max(y0, top)
    ix1, iy1 = min(x1, right), min(y1, bottom)
    iw, ih = max(ix1 - ix0, 0.0), max(iy1 - iy0, 0.0)
    return iw > 0 and ih > 0, iw * ih, ix0 - left, iy0 - top


def coco_image_path(coco_root: Path, image: dict) -> Path:
    url = str(image.get("coco_url") or image.get("file_name") or "")
    filename = os.path.basename(url)
    if "train2017" in url:
        split_dir = "train2017"
    elif "val2017" in url:
        split_dir = "val2017"
    else:
        train_candidate = coco_root / "train2017" / filename
        val_candidate = coco_root / "val2017" / filename
        if train_candidate.exists():
            return train_candidate
        if val_candidate.exists():
            return val_candidate
        raise ValueError(f"Cannot resolve COCO split for image_id={image.get('id')}: {url}")
    return coco_root / split_dir / filename


def build_caption_text(segments: Sequence[dict], n_ovt_per_object: int) -> str:
    category_counts: Counter = Counter()
    ovt_text = "<ovt>" * n_ovt_per_object
    parts: List[str] = []
    for segment in segments:
        category = str(segment["category"])
        category_counts[category] += 1
        label = f"{category.capitalize()} {category_counts[category]}"
        description = str(segment["description"]).rstrip(".").rstrip()
        parts.append(f"<thing> {label}: {description}. {ovt_text}.")
    return " ".join(parts) + " <scene_end>"


def build_manifest(args: argparse.Namespace) -> Dict[str, object]:
    source_path = Path(args.input)
    coco_root = Path(args.coco_root)
    output_path = Path(args.output)
    stats_path = Path(args.stats_output or f"{output_path}.stats.json")

    print(f"Loading {source_path} ...", flush=True)
    with source_path.open() as handle:
        source = json.load(handle)

    images = {int(image["id"]): image for image in source["images"]}
    categories = {int(category["id"]): category for category in source["categories"]}
    annotations: Dict[int, List[dict]] = defaultdict(list)
    for annotation in source["annotations"]:
        annotations[int(annotation["image_id"])].append(annotation)

    tokenizer = None
    if args.max_caption_tokens > 0:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, use_fast=False)

    coco_val_names = {path.name for path in (coco_root / "val2017").glob("*.jpg")}
    stats: Counter = Counter()
    invalid_reasons: Counter = Counter()
    object_count_histogram: Counter = Counter()
    caption_token_histogram: Counter = Counter()
    bad_caption_examples: List[dict] = []

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(output_path.name + ".tmp")
    with temporary_path.open("w") as output:
        for image_id in sorted(images):
            image = images[image_id]
            stats["source_images"] += 1

            image_path = coco_image_path(coco_root, image)
            if not image_path.exists():
                stats["drop_missing_image"] += 1
                continue
            if args.only_coco_val and image_path.name not in coco_val_names:
                stats["drop_non_coco_val"] += 1
                continue
            if args.exclude_coco_val and image_path.name in coco_val_names:
                stats["drop_coco_val_leakage"] += 1
                continue
            if args.drop_non_exhaustive and image.get("not_exhaustive_category_ids"):
                stats["drop_non_exhaustive"] += 1
                continue

            width, height = int(image["width"]), int(image["height"])
            visible: List[Tuple[dict, str, float, float, float]] = []
            failed_reason = None
            for annotation in annotations.get(image_id, []):
                is_visible, approx_area, crop_x, crop_y = bbox_crop_intersection(
                    annotation.get("bbox", (0, 0, 0, 0)), width, height, args.crop_size
                )
                if not is_visible:
                    stats["instances_outside_crop"] += 1
                    continue
                caption = clean_caption(annotation.get("caption"))
                reason = invalid_caption_reason(caption)
                if reason is not None:
                    failed_reason = reason
                    invalid_reasons[reason] += 1
                    if len(bad_caption_examples) < args.max_bad_caption_examples:
                        bad_caption_examples.append(
                            {
                                "image_id": image_id,
                                "ann_id": int(annotation["id"]),
                                "category": str(categories[int(annotation["category_id"])]["name"]),
                                "reason": reason,
                                "caption": caption,
                            }
                        )
                    break
                visible.append((annotation, caption, approx_area, crop_x, crop_y))

            if failed_reason is not None:
                stats["drop_invalid_visible_caption"] += 1
                continue
            if not visible:
                stats["drop_no_visible_instance"] += 1
                continue
            if len(visible) > args.max_objects:
                stats["drop_over_max_objects"] += 1
                stats["instances_in_dropped_overflow_images"] += len(visible)
                continue

            visible.sort(key=lambda item: (-item[2], item[3], item[4], int(item[0]["id"])))
            segments: List[dict] = []
            for annotation, caption, _area, _crop_x, _crop_y in visible:
                category_id = int(annotation["category_id"])
                segments.append(
                    {
                        "ann_id": int(annotation["id"]),
                        "category_id": category_id,
                        "category": str(categories[category_id]["name"]),
                        "description": caption,
                        "caption_source": "lviscap_gemini2_flash",
                        "is_thing": True,
                        "segmentation": annotation["segmentation"],
                        "bbox": annotation["bbox"],
                        "area": float(annotation.get("area", 0.0)),
                    }
                )

            caption_tokens = 0
            if tokenizer is not None:
                caption_text = build_caption_text(segments, args.n_ovt_per_object)
                caption_tokens = len(tokenizer.encode(caption_text, add_special_tokens=False))
                if caption_tokens > args.max_caption_tokens:
                    stats["drop_over_caption_tokens"] += 1
                    stats["instances_in_dropped_long_caption_images"] += len(segments)
                    continue

            record = {
                "dataset_type": "coco_instance",
                "source_dataset": "lviscap_v1",
                "split": args.split,
                "image_id": image_id,
                "image_path": str(image_path),
                "height": height,
                "width": width,
                "segments": segments,
            }
            output.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
            stats["images_written"] += 1
            stats["instances_written"] += len(segments)
            object_count_histogram[len(segments)] += 1
            if caption_tokens:
                caption_token_histogram[(caption_tokens // 100) * 100] += 1
            if args.max_images is not None and stats["images_written"] >= args.max_images:
                stats["stopped_at_max_images"] = 1
                break

    os.replace(temporary_path, output_path)

    result: Dict[str, object] = {
        "input": str(source_path.resolve()),
        "output": str(output_path.resolve()),
        "split": args.split,
        "crop_size": args.crop_size,
        "max_objects": args.max_objects,
        "drop_non_exhaustive": bool(args.drop_non_exhaustive),
        "exclude_coco_val": bool(args.exclude_coco_val),
        "only_coco_val": bool(args.only_coco_val),
        "max_caption_tokens": args.max_caption_tokens,
        "tokenizer_path": args.tokenizer_path if tokenizer is not None else None,
        "n_ovt_per_object": args.n_ovt_per_object,
        "stats": dict(sorted(stats.items())),
        "invalid_caption_reasons_encountered": dict(sorted(invalid_reasons.items())),
        "object_count_histogram": {str(k): v for k, v in sorted(object_count_histogram.items())},
        "caption_token_histogram_100wide": {
            f"{k}-{k + 99}": v for k, v in sorted(caption_token_histogram.items())
        },
        "bad_caption_examples": bad_caption_examples,
    }
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    with stats_path.open("w") as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default="/home/jovyan/data/coco/lviscap/lviscap_v1_train.json",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--stats_output", default=None)
    parser.add_argument("--coco_root", default="/home/jovyan/data/coco")
    parser.add_argument("--split", default="train")
    parser.add_argument("--crop_size", type=int, default=512)
    parser.add_argument("--max_objects", type=int, default=50)
    parser.add_argument("--n_ovt_per_object", type=int, default=1)
    parser.add_argument("--max_caption_tokens", type=int, default=1024)
    parser.add_argument(
        "--tokenizer_path",
        default="/home/jovyan/PGOT/checkpoints/pgot_instance_e1/checkpoint-10000",
    )
    parser.add_argument("--drop_non_exhaustive", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--exclude_coco_val", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--only_coco_val",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Keep only images whose JPEG belongs to COCO val2017.",
    )
    parser.add_argument("--max_bad_caption_examples", type=int, default=30)
    parser.add_argument("--max_images", type=int, default=None, help="Testing-only output cap")
    args = parser.parse_args()
    if args.only_coco_val and args.exclude_coco_val:
        parser.error("--only_coco_val requires --no-exclude_coco_val")
    build_manifest(args)


if __name__ == "__main__":
    main()
