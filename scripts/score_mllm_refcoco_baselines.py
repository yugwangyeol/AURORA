"""Score cached DeOR/LISA/GLaMM masks with one shared RefCOCO protocol."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from pycocotools.coco import COCO

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mllm_refcoco_common import load_samples, prediction_folder, write_json  # noqa: E402


THRESHOLDS = (0.5, 0.6, 0.7, 0.8, 0.9)


def parse_model(value):
    if "=" not in value:
        raise argparse.ArgumentTypeError("model must be NAME=PREDICTION_DIRECTORY")
    name, directory = value.split("=", 1)
    if not name or not directory:
        raise argparse.ArgumentTypeError("model must be NAME=PREDICTION_DIRECTORY")
    return name, Path(directory)


def mask_iou(mask, targets, target_areas):
    pred_area = int(mask.sum())
    intersections = np.logical_and(targets, mask[None]).sum(axis=(1, 2), dtype=np.int64)
    unions = pred_area + target_areas - intersections
    ious = np.divide(intersections, unions, out=np.zeros(len(targets), dtype=np.float64), where=unions > 0)
    return intersections, unions, ious


def audit_image(coco, samples, models):
    image_id = int(samples[0]["image_id"])
    annotations = sorted(
        (
            coco.anns[aid]
            for aid in coco.getAnnIds(imgIds=[image_id])
            if not coco.anns[aid].get("iscrowd", 0)
        ),
        key=lambda ann: int(ann["id"]),
    )
    masks_and_annotations = [
        (ann, coco.annToMask(ann).astype(bool)) for ann in annotations
    ]
    masks_and_annotations = [(ann, mask) for ann, mask in masks_and_annotations if mask.any()]
    annotation_ids = [int(ann["id"]) for ann, _ in masks_and_annotations]
    categories = np.asarray([int(ann["category_id"]) for ann, _ in masks_and_annotations])
    gt_masks = np.stack([mask for _, mask in masks_and_annotations])
    gt_areas = gt_masks.sum(axis=(1, 2), dtype=np.int64)
    rows = []
    for sample in samples:
        target = annotation_ids.index(int(sample["annotation_id"]))
        row = {key: sample[key] for key in (
            "split", "ref_id", "sent_id", "image_id", "annotation_id", "shared_text"
        )}
        same_category = np.flatnonzero(categories == categories[target])
        same_category = same_category[same_category != target]
        row["same_category_competitors"] = int(len(same_category))
        for name, directory in models:
            folder = prediction_folder(directory, sample)
            if not (folder / "prediction.npz").is_file():
                raise FileNotFoundError(f"Missing {name} prediction: {folder}")
            metadata_path = folder / "metadata.json"
            if metadata_path.is_file():
                metadata = json.loads(metadata_path.read_text())
                for key in ("query_sha256", "image_sha256"):
                    if metadata.get(key) != sample[key]:
                        raise ValueError(f"{name} prediction provenance mismatch: {folder}")
            with np.load(folder / "prediction.npz") as saved:
                prediction = saved["mask"].astype(bool)
            if prediction.shape != gt_masks.shape[1:]:
                raise ValueError(f"{name} mask has wrong shape: {folder}")
            intersections, unions, ious = mask_iou(prediction, gt_masks, gt_areas)
            target_iou = float(ious[target])
            other_ious = np.delete(ious, target)
            target_hit = target_iou > 0 and (
                len(other_ious) == 0 or target_iou > float(other_ious.max()) + 1e-12
            )
            if len(same_category):
                same_category_hit = target_iou > 0 and target_iou > float(ious[same_category].max()) + 1e-12
            else:
                same_category_hit = None
            row[name] = {
                "intersection": int(intersections[target]),
                "union": int(unions[target]),
                "iou": target_iou,
                "empty_mask": not bool(prediction.any()),
                "target_hit_at_1": bool(target_hit),
                "same_category_target_hit_at_1": same_category_hit,
            }
        rows.append(row)
    return rows


def aggregate(rows, model_names):
    result = {
        "expressions": len(rows),
        "images": len({row["image_id"] for row in rows}),
        "references": len({row["ref_id"] for row in rows}),
    }
    for name in model_names:
        values = [row[name] for row in rows]
        union = sum(value["union"] for value in values)
        same = [
            value["same_category_target_hit_at_1"]
            for value in values
            if value["same_category_target_hit_at_1"] is not None
        ]
        result[name] = {
            "mIoU": 100.0 * float(np.mean([value["iou"] for value in values])),
            "oIoU": 100.0 * sum(value["intersection"] for value in values) / union if union else 0.0,
            "precision_at_iou": {
                f"{threshold:.1f}": 100.0 * float(np.mean([
                    value["iou"] >= threshold for value in values
                ]))
                for threshold in THRESHOLDS
            },
            "target_hit_at_1": 100.0 * float(np.mean([
                value["target_hit_at_1"] for value in values
            ])),
            "same_category_target_hit_at_1": (
                100.0 * float(np.mean(same)) if same else None
            ),
            "same_category_expressions": len(same),
            "empty_masks": sum(value["empty_mask"] for value in values),
        }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--train", required=True)
    parser.add_argument("--annotations", default="/home/jovyan/data/coco/refcoco/instances.json")
    parser.add_argument("--output", required=True)
    parser.add_argument("--subset", choices=("full", "nonoverlap"), default="nonoverlap")
    parser.add_argument("--splits", nargs="+", default=("val", "testA", "testB"))
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--model", action="append", type=parse_model, required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if len({name for name, _ in args.model}) != len(args.model):
        parser.error("duplicate model name")
    manifest, samples = load_samples(
        args.manifest, args.subset, args.train, args.max_samples, args.splits
    )
    coco = COCO(args.annotations)
    grouped = defaultdict(list)
    for sample in samples:
        grouped[int(sample["image_id"])].append(sample)
    started = time.monotonic()
    rows = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(audit_image, coco, group, args.model) for group in grouped.values()]
        for index, future in enumerate(as_completed(futures), 1):
            rows.extend(future.result())
            if index % 500 == 0:
                print(f"scored {index}/{len(futures)} images", flush=True)
    rows.sort(key=lambda row: (args.splits.index(row["split"]), row["ref_id"], row["sent_id"]))
    model_names = [name for name, _ in args.model]
    by_split = {}
    for split in args.splits:
        selected = [row for row in rows if row["split"] == split]
        if selected:
            by_split[split] = aggregate(selected, model_names)
    by_split["all"] = aggregate(rows, model_names)
    result = {
        "protocol": {
            "task": "RefCOCO referring-expression segmentation",
            "input": "one image and one original human expression; model-native fixed wrapper",
            "gt_inference_inputs": "none",
            "mask": "released mask logit thresholded at zero and resized by each native SAM decoder",
            "aggregation": "expression-weighted mIoU/P@t/TargetHit@1; foreground pixel-aggregated oIoU",
            "target_hit": "positive unique highest IoU against all nonempty noncrowd GT instances",
            "same_category_target_hit": "positive unique highest IoU among target-category GT instances",
            "subset": args.subset,
        },
        "manifest": str(Path(args.manifest).resolve()),
        "source_fingerprint": manifest["fingerprint"],
        "models": {name: str(directory.resolve()) for name, directory in args.model},
        "by_split": by_split,
        "elapsed_seconds": time.monotonic() - started,
    }
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "summary.json", result)
    with (output / "per_expression.jsonl").open("w") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    with (output / "comparison.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow([
            "split", "images", "expressions", "method", "mIoU", "oIoU", "P@0.5",
            "TargetHit@1", "Same-category TargetHit@1", "empty_masks",
        ])
        for split, stats in by_split.items():
            for name in model_names:
                value = stats[name]
                writer.writerow([
                    split, stats["images"], stats["expressions"], name, value["mIoU"],
                    value["oIoU"], value["precision_at_iou"]["0.5"], value["target_hit_at_1"],
                    value["same_category_target_hit_at_1"], value["empty_masks"],
                ])
    with (output / "table.md").open("w") as stream:
        stream.write("| Split | Method | mIoU | oIoU | P@0.5 | TargetHit@1 | Same-cat. TH@1 |\n")
        stream.write("|---|---|---:|---:|---:|---:|---:|\n")
        for split, stats in by_split.items():
            for name in model_names:
                value = stats[name]
                same = value["same_category_target_hit_at_1"]
                same_text = "—" if same is None else f"{same:.2f}"
                stream.write(
                    f"| {split} | {name} | {value['mIoU']:.2f} | {value['oIoU']:.2f} | "
                    f"{value['precision_at_iou']['0.5']:.2f} | {value['target_hit_at_1']:.2f} | "
                    f"{same_text} |\n"
                )
    print((output / "table.md").read_text(), flush=True)


if __name__ == "__main__":
    main()
