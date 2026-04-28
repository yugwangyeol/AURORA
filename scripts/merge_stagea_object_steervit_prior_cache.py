#!/usr/bin/env python
"""Merge sharded Stage A object SteerViT prior caches into a single output dir."""

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Any, Dict, List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge sharded object SteerViT prior caches.")
    parser.add_argument(
        "--input-dirs",
        nargs="+",
        required=True,
        help="Per-shard cache directories to merge.",
    )
    parser.add_argument("--output-dir", required=True, help="Merged output directory.")
    parser.add_argument("--overwrite", action="store_true", help="Allow writing into a non-empty output dir.")
    return parser.parse_args()


def ensure_output_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def save_json(path: str, payload: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def sort_key(text: str):
    return (0, int(text)) if str(text).isdigit() else (1, str(text))


def main() -> None:
    args = parse_args()

    output_dir = os.path.abspath(args.output_dir)
    shards_out = os.path.join(output_dir, "shards")
    if os.path.exists(output_dir):
        has_existing = any(Path(output_dir).iterdir())
        if has_existing and not args.overwrite:
            raise SystemExit(
                f"Output dir is not empty: {output_dir}\n"
                "Use --overwrite or choose a new output directory."
            )
    ensure_output_dir(output_dir)
    ensure_output_dir(shards_out)

    merged_annotations: Dict[str, Dict[str, Any]] = {}
    merged_metadata: List[Dict[str, Any]] = []
    merged_shards: List[str] = []
    shard_counter = 0
    source_summaries: List[Dict[str, Any]] = []

    for input_dir in args.input_dirs:
        src_dir = os.path.abspath(input_dir)
        metadata_path = os.path.join(src_dir, "metadata.jsonl")
        annotations_path = os.path.join(src_dir, "captionslot_annotations.json")
        summary_path = os.path.join(src_dir, "summary.json")
        shards_dir = os.path.join(src_dir, "shards")

        if not os.path.isfile(metadata_path):
            raise SystemExit(f"Missing metadata.jsonl in {src_dir}")
        if not os.path.isfile(annotations_path):
            raise SystemExit(f"Missing captionslot_annotations.json in {src_dir}")
        if not os.path.isdir(shards_dir):
            raise SystemExit(f"Missing shards dir in {src_dir}")

        metadata_rows = load_jsonl(metadata_path)
        annotations = load_json(annotations_path)
        summary = load_json(summary_path) if os.path.isfile(summary_path) else {"output_dir": src_dir}
        source_summaries.append(summary)

        shard_map: Dict[str, str] = {}
        shard_files = sorted(
            [
                os.path.join(shards_dir, name)
                for name in os.listdir(shards_dir)
                if name.startswith("cache_") and name.endswith(".pt")
            ]
        )

        for shard_path in shard_files:
            new_name = f"cache_{shard_counter:05d}.pt"
            new_path = os.path.abspath(os.path.join(shards_out, new_name))
            shutil.copy2(shard_path, new_path)
            shard_map[os.path.abspath(shard_path)] = new_path
            merged_shards.append(new_path)
            shard_counter += 1

        for image_id, ann in annotations.items():
            image_id_str = str(image_id)
            if image_id_str in merged_annotations:
                raise SystemExit(f"Duplicate image_id during merge: {image_id_str}")
            ann_copy = dict(ann)
            old_shard = ann_copy.get("map_shard")
            if old_shard:
                old_shard = os.path.abspath(old_shard)
                if old_shard not in shard_map:
                    raise SystemExit(f"Annotation shard path not found in copied shards: {old_shard}")
                ann_copy["map_shard"] = shard_map[old_shard]
            merged_annotations[image_id_str] = ann_copy

        for row in metadata_rows:
            row_copy = dict(row)
            image_id_str = str(row_copy.get("image_id"))
            if not image_id_str:
                continue
            if row_copy.get("map_shard"):
                old_shard = os.path.abspath(str(row_copy["map_shard"]))
                if old_shard not in shard_map:
                    raise SystemExit(f"Metadata shard path not found in copied shards: {old_shard}")
                row_copy["map_shard"] = shard_map[old_shard]
            merged_metadata.append(row_copy)

    merged_annotations = {
        image_id: merged_annotations[image_id]
        for image_id in sorted(merged_annotations.keys(), key=sort_key)
    }
    merged_metadata.sort(key=lambda row: sort_key(str(row.get("image_id", ""))))

    annotations_out = os.path.join(output_dir, "captionslot_annotations.json")
    metadata_out = os.path.join(output_dir, "metadata.jsonl")
    summary_out = os.path.join(output_dir, "summary.json")

    save_json(annotations_out, merged_annotations)
    with open(metadata_out, "w", encoding="utf-8") as f:
        for row in merged_metadata:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    annotation_values = list(merged_annotations.values())
    summary = {
        "output_dir": output_dir,
        "input_dirs": [os.path.abspath(path) for path in args.input_dirs],
        "annotation_entry_count": len(merged_annotations),
        "images_with_any_object": sum(int((ann.get("clean_num_objects") or 0) > 0) for ann in annotation_values),
        "images_with_any_head_prior": sum(
            int(any(bool(x) for x in (ann.get("head_valid_mask") or []))) for ann in annotation_values
        ),
        "total_cached_objects": sum(int(ann.get("clean_num_objects") or 0) for ann in annotation_values),
        "total_valid_head_maps": sum(
            sum(1 for x in (ann.get("head_valid_mask") or []) if x) for ann in annotation_values
        ),
        "metadata_jsonl": metadata_out,
        "annotations_json": annotations_out,
        "map_shards": merged_shards,
        "source_summaries": source_summaries,
        "notes": [
            "Merged from per-shard object SteerViT prior caches.",
            "map_shard paths were rewritten to the merged output directory.",
        ],
    }
    save_json(summary_out, summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
