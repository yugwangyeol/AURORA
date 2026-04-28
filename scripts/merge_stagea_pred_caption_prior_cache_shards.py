#!/usr/bin/env python
import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Any, Dict, Iterable, List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge sharded Stage A predicted-caption prior caches.")
    parser.add_argument("--root-output-dir", required=True, help="Final merged output directory.")
    parser.add_argument(
        "--shard-dir",
        dest="shard_dirs",
        action="append",
        required=True,
        help="Shard output directory. Pass this flag once per shard.",
    )
    return parser.parse_args()


def read_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: str, payload: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def iter_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(path: str, records: List[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def merge_samples(shard_dirs: List[str], root_output_dir: str) -> None:
    root_samples_dir = os.path.join(root_output_dir, "samples")
    ensure_dir(root_samples_dir)
    for shard_dir in shard_dirs:
        shard_samples_dir = os.path.join(shard_dir, "samples")
        if not os.path.isdir(shard_samples_dir):
            continue
        for sample_name in sorted(os.listdir(shard_samples_dir)):
            src_dir = os.path.join(shard_samples_dir, sample_name)
            dst_dir = os.path.join(root_samples_dir, sample_name)
            if not os.path.isdir(src_dir):
                continue
            ensure_dir(dst_dir)
            for file_name in sorted(os.listdir(src_dir)):
                src_file = os.path.join(src_dir, file_name)
                dst_file = os.path.join(dst_dir, file_name)
                if os.path.isfile(src_file):
                    shutil.copy2(src_file, dst_file)


def main() -> None:
    args = parse_args()
    root_output_dir = os.path.abspath(args.root_output_dir)
    shard_dirs = [os.path.abspath(path) for path in args.shard_dirs]
    ensure_dir(root_output_dir)

    shard_summaries: List[Dict[str, Any]] = []
    metadata_records: List[Dict[str, Any]] = []
    map_shards: List[str] = []

    for shard_dir in shard_dirs:
        summary_path = os.path.join(shard_dir, "summary.json")
        if not os.path.isfile(summary_path):
            raise FileNotFoundError(f"Missing shard summary: {summary_path}")
        summary = read_json(summary_path)
        shard_summaries.append(summary)
        map_shards.extend(summary.get("map_shards", []))

        metadata_path = os.path.join(shard_dir, "metadata.jsonl")
        if os.path.isfile(metadata_path):
            metadata_records.extend(iter_jsonl(metadata_path))

    if not shard_summaries:
        raise SystemExit("No shard summaries found.")

    metadata_records.sort(key=lambda item: item.get("image", ""))
    merged_metadata_path = os.path.join(root_output_dir, "metadata.jsonl")
    write_jsonl(merged_metadata_path, metadata_records)
    merge_samples(shard_dirs, root_output_dir)

    first_non_empty = next(
        (summary for summary in shard_summaries if summary.get("requested_samples", 0)),
        shard_summaries[0],
    )

    notes = list(first_non_empty.get("notes", []))
    notes.append("This summary was merged from sharded Stage A cache runs and sorted by image path.")

    summary: Dict[str, Any] = {
        "image_dir": first_non_empty.get("image_dir"),
        "output_dir": root_output_dir,
        "model_path": first_non_empty.get("model_path"),
        "caption_prompt": first_non_empty.get("caption_prompt"),
        "caption_prompt_built": first_non_empty.get("caption_prompt_built"),
        "spacy_model": first_non_empty.get("spacy_model"),
        "device": first_non_empty.get("device"),
        "dtype": first_non_empty.get("dtype"),
        "caption_batch_size": first_non_empty.get("caption_batch_size"),
        "caption_max_new_tokens": first_non_empty.get("caption_max_new_tokens"),
        "max_caption_tokens": first_non_empty.get("max_caption_tokens"),
        "attention_temperature": first_non_empty.get("attention_temperature"),
        "normalize_attention_tokens": first_non_empty.get("normalize_attention_tokens"),
        "seed": first_non_empty.get("seed"),
        "num_shards": len(shard_summaries),
        "shard_dirs": shard_dirs,
        "shard_summary_paths": [os.path.join(shard_dir, "summary.json") for shard_dir in shard_dirs],
        "requested_samples": int(sum(int(summary.get("requested_samples", 0)) for summary in shard_summaries)),
        "global_requested_samples": int(first_non_empty.get("global_requested_samples", 0)),
        "prior_valid_count": int(sum(int(summary.get("prior_valid_count", 0)) for summary in shard_summaries)),
        "prior_invalid_count": int(sum(int(summary.get("prior_invalid_count", 0)) for summary in shard_summaries)),
        "error_count": int(sum(int(summary.get("error_count", 0)) for summary in shard_summaries)),
        "fallback_or_missing_count": int(sum(int(summary.get("fallback_or_missing_count", 0)) for summary in shard_summaries)),
        "debug_saved_count": int(sum(int(summary.get("debug_saved_count", 0)) for summary in shard_summaries)),
        "metadata_jsonl": os.path.abspath(merged_metadata_path),
        "map_shards": sorted(set(os.path.abspath(path) for path in map_shards)),
        "notes": notes,
    }

    summary_path = os.path.join(root_output_dir, "summary.json")
    write_json(summary_path, summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
