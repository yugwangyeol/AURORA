#!/usr/bin/env python
"""Merge sharded Stage A object caption outputs."""

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge sharded Stage A object caption outputs.")
    parser.add_argument("--input-dirs", nargs="+", required=True, help="Per-shard caption output dirs.")
    parser.add_argument("--output-dir", required=True, help="Merged output dir.")
    parser.add_argument("--overwrite", action="store_true", help="Allow writing into a non-empty output dir.")
    return parser.parse_args()


def save_json(path: str, payload: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def sort_key(text: str):
    return (0, int(text)) if str(text).isdigit() else (1, str(text))


def main() -> None:
    args = parse_args()
    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    if any(Path(output_dir).iterdir()) and not args.overwrite:
        raise SystemExit(f"Output dir is not empty: {output_dir}\nUse --overwrite or a new output dir.")

    merged_predictions: List[Dict[str, Any]] = []
    merged_annotations: Dict[str, Dict[str, Any]] = {}
    merged_examples: List[Dict[str, Any]] = []
    source_summaries: List[Dict[str, Any]] = []

    for input_dir in args.input_dirs:
        src_dir = os.path.abspath(input_dir)
        predictions_path = os.path.join(src_dir, "predictions.jsonl")
        annotations_path = os.path.join(src_dir, "object_caption_annotations.json")
        summary_path = os.path.join(src_dir, "summary.json")
        example_candidates = sorted(Path(src_dir).glob("saved_examples_first_*.json"))
        examples_path = str(example_candidates[0]) if example_candidates else None

        if not os.path.isfile(predictions_path):
            raise SystemExit(f"Missing predictions.jsonl in {src_dir}")
        if not os.path.isfile(annotations_path):
            raise SystemExit(f"Missing object_caption_annotations.json in {src_dir}")

        preds = load_jsonl(predictions_path)
        anns = load_json(annotations_path)
        summary = load_json(summary_path) if os.path.isfile(summary_path) else {"output_dir": src_dir}
        source_summaries.append(summary)

        for row in preds:
            image_id = str(row.get("image_id", ""))
            if not image_id:
                continue
            merged_predictions.append(row)

        for image_id, ann in anns.items():
            image_id = str(image_id)
            if image_id in merged_annotations:
                raise SystemExit(f"Duplicate image_id during merge: {image_id}")
            merged_annotations[image_id] = ann

        if examples_path and os.path.isfile(examples_path):
            merged_examples.extend(load_json(examples_path))

    merged_predictions.sort(key=lambda row: sort_key(str(row.get("image_id", ""))))
    merged_annotations = {
        image_id: merged_annotations[image_id]
        for image_id in sorted(merged_annotations.keys(), key=sort_key)
    }

    predictions_out = os.path.join(output_dir, "predictions.jsonl")
    annotations_out = os.path.join(output_dir, "object_caption_annotations.json")
    examples_out = os.path.join(output_dir, "saved_examples_first_100.json")
    summary_out = os.path.join(output_dir, "summary.json")

    with open(predictions_out, "w", encoding="utf-8") as f:
        for row in merged_predictions:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    save_json(annotations_out, merged_annotations)
    save_json(examples_out, merged_examples[:100])

    annotation_values = list(merged_annotations.values())
    summary = {
        "output_dir": output_dir,
        "input_dirs": [os.path.abspath(path) for path in args.input_dirs],
        "annotation_entry_count": len(merged_annotations),
        "images_with_any_object": sum(int((ann.get("clean_num_objects") or 0) > 0) for ann in annotation_values),
        "images_with_background": sum(int(bool(ann.get("background"))) for ann in annotation_values),
        "total_clean_objects": sum(int(ann.get("clean_num_objects") or 0) for ann in annotation_values),
        "predictions_jsonl": predictions_out,
        "annotations_json": annotations_out,
        "examples_json": examples_out,
        "source_summaries": source_summaries,
        "notes": [
            "Merged from per-shard Stage A object caption outputs.",
            "This is the caption-only stage; no SteerViT maps are included.",
        ],
    }
    save_json(summary_out, summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
