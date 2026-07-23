#!/usr/bin/env python
"""Attach LVISCap val captions to matching COCO-instance evaluation objects.

COCO and LVIS annotate the same val2017 images, but their instance sets and
taxonomies are not identical.  This script therefore keeps the existing COCO
instances/masks (and hence the CODA evaluation protocol) and replaces a COCO
object's category-only description only when a same-image LVIS instance has
sufficient mask IoU.  Unmatched COCO objects retain their category caption.

The output contains exactly the same images, COCO masks, and object order as
the input JSONL.  Only ``description``/caption provenance fields change.
"""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from pycocotools import mask as mask_utils
from scipy.optimize import linear_sum_assignment
from transformers import AutoTokenizer

from prepare_lviscap_pgot import build_caption_text, clean_caption, invalid_caption_reason


def decode_mask(segmentation, height: int, width: int) -> np.ndarray:
    if isinstance(segmentation, list):
        rle = mask_utils.merge(mask_utils.frPyObjects(segmentation, height, width))
    elif isinstance(segmentation, dict) and isinstance(segmentation.get("counts"), list):
        rle = mask_utils.frPyObjects(segmentation, height, width)
    else:
        rle = segmentation
    mask = mask_utils.decode(rle)
    if mask.ndim == 3:
        mask = mask.any(axis=2)
    return mask.astype(bool, copy=False)


def bbox_iou(a, b) -> float:
    ax, ay, aw, ah = (float(x) for x in a)
    bx, by, bw, bh = (float(x) for x in b)
    ix0, iy0 = max(ax, bx), max(ay, by)
    ix1, iy1 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    inter = max(ix1 - ix0, 0.0) * max(iy1 - iy0, 0.0)
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = np.logical_and(a, b).sum(dtype=np.int64)
    union = np.logical_or(a, b).sum(dtype=np.int64)
    return float(inter / union) if union else 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--coco_jsonl", required=True)
    parser.add_argument("--lviscap_json", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--stats_output", default=None)
    parser.add_argument("--min_mask_iou", type=float, default=0.5)
    parser.add_argument("--max_caption_tokens", type=int, default=1024)
    parser.add_argument(
        "--tokenizer_path",
        default="/home/jovyan/PGOT/checkpoints/pgot_lviscap_hardreg_e2/checkpoint-10000",
    )
    parser.add_argument(
        "--min_bbox_iou",
        type=float,
        default=0.05,
        help="Cheap candidate pruning before mask decoding; not the acceptance criterion.",
    )
    args = parser.parse_args()

    coco_path = Path(args.coco_jsonl)
    lvis_path = Path(args.lviscap_json)
    output_path = Path(args.output)
    stats_path = Path(args.stats_output or f"{output_path}.stats.json")

    print(f"Loading {lvis_path} ...", flush=True)
    with lvis_path.open() as handle:
        lvis = json.load(handle)
    lvis_categories = {int(c["id"]): str(c["name"]) for c in lvis["categories"]}
    lvis_by_image = defaultdict(list)
    invalid_lvis_captions = Counter()
    for annotation in lvis["annotations"]:
        caption = clean_caption(annotation.get("caption"))
        reason = invalid_caption_reason(caption)
        if reason is not None:
            invalid_lvis_captions[reason] += 1
            continue
        enriched = dict(annotation)
        enriched["clean_caption"] = caption
        lvis_by_image[int(annotation["image_id"])].append(enriched)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, use_fast=False)

    stats = Counter()
    match_ious = []
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(output_path.name + ".tmp")
    with coco_path.open() as source, temporary_path.open("w") as destination:
        for line in source:
            sample = json.loads(line)
            stats["images"] += 1
            height, width = int(sample["height"]), int(sample["width"])
            coco_segments = sample["segments"]
            lvis_anns = lvis_by_image.get(int(sample["image_id"]), [])
            stats["coco_objects"] += len(coco_segments)
            if not lvis_anns:
                stats["images_without_lviscap_annotations"] += 1
                stats["fallback_category_objects"] += len(coco_segments)
                destination.write(json.dumps(sample, ensure_ascii=False, separators=(",", ":")) + "\n")
                continue

            # Decode each mask at most once per image. Candidate bbox pruning
            # prevents decoding unrelated LVIS objects in densely annotated images.
            coco_masks = [None] * len(coco_segments)
            lvis_masks = [None] * len(lvis_anns)
            scores = np.zeros((len(coco_segments), len(lvis_anns)), dtype=np.float32)
            for i, coco_seg in enumerate(coco_segments):
                for j, lvis_ann in enumerate(lvis_anns):
                    if bbox_iou(coco_seg["bbox"], lvis_ann["bbox"]) < args.min_bbox_iou:
                        continue
                    if coco_masks[i] is None:
                        coco_masks[i] = decode_mask(coco_seg["segmentation"], height, width)
                    if lvis_masks[j] is None:
                        lvis_masks[j] = decode_mask(lvis_ann["segmentation"], height, width)
                    scores[i, j] = mask_iou(coco_masks[i], lvis_masks[j])

            matched_indices = []
            matched_ious = {}
            original_caption_fields = {}
            if scores.size:
                rows, cols = linear_sum_assignment(-scores)
                for i, j in zip(rows.tolist(), cols.tolist()):
                    iou = float(scores[i, j])
                    if iou < float(args.min_mask_iou):
                        continue
                    segment = coco_segments[i]
                    annotation = lvis_anns[j]
                    original_caption_fields[i] = (
                        segment["description"],
                        segment.get("caption_source", "category"),
                    )
                    segment["description"] = annotation["clean_caption"]
                    segment["caption_source"] = "lviscap_gemini2_flash_val_mask_match"
                    segment["lviscap_ann_id"] = int(annotation["id"])
                    segment["lviscap_category_id"] = int(annotation["category_id"])
                    segment["lviscap_category"] = lvis_categories[int(annotation["category_id"])]
                    segment["lviscap_mask_iou"] = iou
                    matched_indices.append(i)
                    matched_ious[i] = iou
            # Rich per-instance sentences can make a crowded image exceed the
            # training-time 1024-token caption budget. Revert the longest
            # matched captions first rather than silently dropping COCO masks.
            # This preserves exactly the original evaluation object set.
            token_count = len(
                tokenizer.encode(
                    build_caption_text(coco_segments, n_ovt_per_object=1),
                    add_special_tokens=False,
                )
            )
            if token_count > args.max_caption_tokens:
                candidates = sorted(
                    matched_indices,
                    key=lambda idx: len(
                        tokenizer.encode(
                            str(coco_segments[idx]["description"]),
                            add_special_tokens=False,
                        )
                    ),
                    reverse=True,
                )
                for i in candidates:
                    old_description, old_source = original_caption_fields[i]
                    segment = coco_segments[i]
                    segment["description"] = old_description
                    segment["caption_source"] = old_source
                    for key in (
                        "lviscap_ann_id",
                        "lviscap_category_id",
                        "lviscap_category",
                        "lviscap_mask_iou",
                    ):
                        segment.pop(key, None)
                    matched_indices.remove(i)
                    matched_ious.pop(i, None)
                    stats["budget_reverted_objects"] += 1
                    token_count = len(
                        tokenizer.encode(
                            build_caption_text(coco_segments, n_ovt_per_object=1),
                            add_special_tokens=False,
                        )
                    )
                    if token_count <= args.max_caption_tokens:
                        break
            matched = len(matched_indices)
            match_ious.extend(matched_ious.values())
            stats["max_caption_tokens_observed"] = max(
                stats["max_caption_tokens_observed"], token_count
            )
            stats["matched_objects"] += matched
            stats["fallback_category_objects"] += len(coco_segments) - matched
            if matched:
                stats["images_with_any_match"] += 1
            if matched == len(coco_segments):
                stats["images_with_all_objects_matched"] += 1
            sample["caption_mix"] = "lviscap_val_with_category_fallback"
            destination.write(json.dumps(sample, ensure_ascii=False, separators=(",", ":")) + "\n")

    temporary_path.replace(output_path)
    result = {
        "coco_jsonl": str(coco_path.resolve()),
        "lviscap_json": str(lvis_path.resolve()),
        "output": str(output_path.resolve()),
        "min_mask_iou": args.min_mask_iou,
        "min_bbox_iou_for_candidates": args.min_bbox_iou,
        "max_caption_tokens": args.max_caption_tokens,
        "tokenizer_path": args.tokenizer_path,
        "stats": dict(sorted(stats.items())),
        "matched_object_fraction": stats["matched_objects"] / max(stats["coco_objects"], 1),
        "match_iou_mean": float(np.mean(match_ious)) if match_ious else None,
        "match_iou_min": float(np.min(match_ious)) if match_ious else None,
        "invalid_lvis_captions": dict(sorted(invalid_lvis_captions.items())),
    }
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    with stats_path.open("w") as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
