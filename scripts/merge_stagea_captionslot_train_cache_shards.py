#!/usr/bin/env python
import argparse
import json
import os
import shutil
from typing import Any, Dict, Iterable, List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge sharded Stage A CaptionSlot train cache outputs.")
    parser.add_argument("--root-output-dir", required=True)
    parser.add_argument("--shard-dir", dest="shard_dirs", action="append", required=True)
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
    annotations: Dict[str, Any] = {}
    map_shards: List[str] = []

    for shard_dir in shard_dirs:
        summary_path = os.path.join(shard_dir, "summary.json")
        if not os.path.isfile(summary_path):
            raise FileNotFoundError(summary_path)
        summary = read_json(summary_path)
        shard_summaries.append(summary)
        map_shards.extend(summary.get("map_shards", []))

        metadata_path = os.path.join(shard_dir, "metadata.jsonl")
        if os.path.isfile(metadata_path):
            metadata_records.extend(iter_jsonl(metadata_path))

        annotations_path = os.path.join(shard_dir, "captionslot_annotations.json")
        if os.path.isfile(annotations_path):
            annotations.update(read_json(annotations_path))

    metadata_records.sort(key=lambda item: item.get("image", ""))
    ordered_annotations = {
        image_id: annotations[image_id]
        for image_id in sorted(annotations.keys(), key=lambda key: int(key))
    }

    merged_metadata_path = os.path.join(root_output_dir, "metadata.jsonl")
    merged_annotations_path = os.path.join(root_output_dir, "captionslot_annotations.json")
    write_jsonl(merged_metadata_path, metadata_records)
    write_json(merged_annotations_path, ordered_annotations)
    merge_samples(shard_dirs, root_output_dir)

    first_non_empty = next((item for item in shard_summaries if item.get("requested_samples", 0)), shard_summaries[0])
    notes = list(first_non_empty.get("notes", []))
    notes.append("This summary was merged from sharded CaptionSlot train-cache runs and sorted by image id.")

    summary = {
        "image_dir": first_non_empty.get("image_dir"),
        "output_dir": root_output_dir,
        "model_path": first_non_empty.get("model_path"),
        "caption_prompt": first_non_empty.get("caption_prompt"),
        "caption_prompt_built": first_non_empty.get("caption_prompt_built"),
        "spacy_model": first_non_empty.get("spacy_model"),
        "device": first_non_empty.get("device"),
        "dtype": first_non_empty.get("dtype"),
        "caption_max_new_tokens": first_non_empty.get("caption_max_new_tokens"),
        "max_caption_tokens": first_non_empty.get("max_caption_tokens"),
        "trace_last_n_layers": first_non_empty.get("trace_last_n_layers"),
        "caption_batch_size": first_non_empty.get("caption_batch_size"),
        "trace_batch_size": first_non_empty.get("trace_batch_size"),
        "spacy_batch_size": first_non_empty.get("spacy_batch_size"),
        "max_slots": first_non_empty.get("max_slots"),
        "use_kv_cache": first_non_empty.get("use_kv_cache"),
        "force_eager_attention": first_non_empty.get("force_eager_attention"),
        "seed": first_non_empty.get("seed"),
        "num_shards": len(shard_summaries),
        "shard_dirs": shard_dirs,
        "shard_summary_paths": [os.path.join(shard_dir, "summary.json") for shard_dir in shard_dirs],
        "requested_samples": int(sum(int(item.get("requested_samples", 0)) for item in shard_summaries)),
        "global_requested_samples": int(first_non_empty.get("global_requested_samples", 0)),
        "annotation_entry_count": int(len(ordered_annotations)),
        "images_with_any_noun_chunk": int(sum(int(item.get("images_with_any_noun_chunk", 0)) for item in shard_summaries)),
        "images_with_any_head_prior": int(sum(int(item.get("images_with_any_head_prior", 0)) for item in shard_summaries)),
        "images_with_any_phrase_prior": int(sum(int(item.get("images_with_any_phrase_prior", 0)) for item in shard_summaries)),
        "total_cached_noun_chunks": int(sum(int(item.get("total_cached_noun_chunks", 0)) for item in shard_summaries)),
        "total_valid_head_maps": int(sum(int(item.get("total_valid_head_maps", 0)) for item in shard_summaries)),
        "total_valid_phrase_maps": int(sum(int(item.get("total_valid_phrase_maps", 0)) for item in shard_summaries)),
        "error_count": int(sum(int(item.get("error_count", 0)) for item in shard_summaries)),
        "caption_batch_fallback_count": int(sum(int(item.get("caption_batch_fallback_count", 0)) for item in shard_summaries)),
        "trace_batch_fallback_count": int(sum(int(item.get("trace_batch_fallback_count", 0)) for item in shard_summaries)),
        "debug_saved_count": int(sum(int(item.get("debug_saved_count", 0)) for item in shard_summaries)),
        "metadata_jsonl": os.path.abspath(merged_metadata_path),
        "annotations_json": os.path.abspath(merged_annotations_path),
        "map_shards": sorted(set(os.path.abspath(path) for path in map_shards)),
        "notes": notes,
    }

    summary_path = os.path.join(root_output_dir, "summary.json")
    write_json(summary_path, summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
