#!/usr/bin/env python
import argparse
import json
import os
import shutil
from typing import Any, Dict, Iterable, List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge sharded Stage A generation-trace diagnostics.")
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


def build_ranking_entry(record: Dict[str, Any], metric_key: str, sample_root: str) -> Dict[str, Any]:
    metric = record.get(metric_key, {}) if isinstance(record.get(metric_key), dict) else {}
    image_id = str(record.get("image_id", ""))
    sample_dir = os.path.join(sample_root, image_id)
    return {
        "image_id": image_id,
        "image": record.get("image"),
        "caption": record.get("caption"),
        "first_noun_phrase": record.get("first_noun_phrase"),
        "noun_head": record.get("noun_head"),
        "selection_strategy": record.get("selection_strategy"),
        "matched_coco_category_names": record.get("matched_coco_category_names", []),
        "inside_mass": metric.get("inside_mass"),
        "argmax_in_box": metric.get("argmax_in_box"),
        "top10_patch_in_box_rate": metric.get("top10_patch_in_box_rate"),
        "record_path": os.path.abspath(os.path.join(sample_dir, "record.json")),
        "trace_overlay_path": os.path.abspath(
            os.path.join(sample_dir, "head_trace_overlay.png" if metric_key == "head_bbox_eval" else "phrase_trace_overlay.png")
        ),
        "trace_map_path": os.path.abspath(
            os.path.join(sample_dir, "head_trace_map.png" if metric_key == "head_bbox_eval" else "phrase_trace_map.png")
        ),
        "bbox_overlay_path": os.path.abspath(os.path.join(sample_dir, "bbox_overlay.png")),
    }


def write_debug_rankings(root_output_dir: str, metadata_records: List[Dict[str, Any]], top_k: int = 10) -> Dict[str, str]:
    analysis_dir = os.path.join(root_output_dir, "analysis")
    ensure_dir(analysis_dir)
    sample_root = os.path.join(root_output_dir, "samples")
    written_paths: Dict[str, str] = {}
    top_k = max(1, int(top_k))

    for metric_key, prefix in (("head_bbox_eval", "head"), ("phrase_bbox_eval", "phrase")):
        ranked = [
            build_ranking_entry(record, metric_key, sample_root)
            for record in metadata_records
            if isinstance(record.get(metric_key), dict) and record[metric_key].get("valid")
        ]
        ranked.sort(key=lambda item: float(item["inside_mass"]), reverse=True)
        best = ranked[:top_k]
        worst = list(reversed(ranked[-top_k:])) if ranked else []
        best_path = os.path.join(analysis_dir, f"{prefix}_best_{top_k}.json")
        worst_path = os.path.join(analysis_dir, f"{prefix}_worst_{top_k}.json")
        write_json(best_path, best)
        write_json(worst_path, worst)
        written_paths[f"{prefix}_best"] = os.path.abspath(best_path)
        written_paths[f"{prefix}_worst"] = os.path.abspath(worst_path)

    guide_path = os.path.join(analysis_dir, "README.txt")
    with open(guide_path, "w", encoding="utf-8") as f:
        f.write("How to review Stage A generation-trace diagnostics\n\n")
        f.write("1. Open head_best_*.json and head_worst_*.json first.\n")
        f.write("2. Compare trace_overlay_path against bbox_overlay_path.\n")
        f.write("3. Use head maps as the primary prior candidate.\n")
    written_paths["guide"] = os.path.abspath(guide_path)
    return written_paths


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
            raise FileNotFoundError(summary_path)
        summary = read_json(summary_path)
        shard_summaries.append(summary)
        map_shards.extend(summary.get("map_shards", []))

        metadata_path = os.path.join(shard_dir, "metadata.jsonl")
        if os.path.isfile(metadata_path):
            metadata_records.extend(iter_jsonl(metadata_path))

    metadata_records.sort(key=lambda item: item.get("image", ""))
    merged_metadata_path = os.path.join(root_output_dir, "metadata.jsonl")
    write_jsonl(merged_metadata_path, metadata_records)
    merge_samples(shard_dirs, root_output_dir)
    analysis_paths = write_debug_rankings(root_output_dir, metadata_records, top_k=10)

    first_non_empty = next((item for item in shard_summaries if item.get("requested_samples", 0)), shard_summaries[0])
    notes = list(first_non_empty.get("notes", []))
    notes.append("This summary was merged from sharded generation-trace runs and sorted by image path.")

    phrase_bbox_records = [rec["phrase_bbox_eval"] for rec in metadata_records if isinstance(rec.get("phrase_bbox_eval"), dict) and rec["phrase_bbox_eval"].get("valid")]
    head_bbox_records = [rec["head_bbox_eval"] for rec in metadata_records if isinstance(rec.get("head_bbox_eval"), dict) and rec["head_bbox_eval"].get("valid")]

    summary = {
        "image_dir": first_non_empty.get("image_dir"),
        "output_dir": root_output_dir,
        "coco_instances_json": first_non_empty.get("coco_instances_json"),
        "model_path": first_non_empty.get("model_path"),
        "caption_prompt": first_non_empty.get("caption_prompt"),
        "caption_prompt_built": first_non_empty.get("caption_prompt_built"),
        "spacy_model": first_non_empty.get("spacy_model"),
        "device": first_non_empty.get("device"),
        "dtype": first_non_empty.get("dtype"),
        "caption_max_new_tokens": first_non_empty.get("caption_max_new_tokens"),
        "max_caption_tokens": first_non_empty.get("max_caption_tokens"),
        "trace_last_n_layers": first_non_empty.get("trace_last_n_layers"),
        "use_kv_cache": first_non_empty.get("use_kv_cache"),
        "force_eager_attention": first_non_empty.get("force_eager_attention"),
        "seed": first_non_empty.get("seed"),
        "num_shards": len(shard_summaries),
        "shard_dirs": shard_dirs,
        "shard_summary_paths": [os.path.join(shard_dir, "summary.json") for shard_dir in shard_dirs],
        "requested_samples": int(sum(int(item.get("requested_samples", 0)) for item in shard_summaries)),
        "global_requested_samples": int(first_non_empty.get("global_requested_samples", 0)),
        "trace_valid_count": int(sum(int(item.get("trace_valid_count", 0)) for item in shard_summaries)),
        "trace_invalid_count": int(sum(int(item.get("trace_invalid_count", 0)) for item in shard_summaries)),
        "error_count": int(sum(int(item.get("error_count", 0)) for item in shard_summaries)),
        "phrase_bbox_eval_count": int(len(phrase_bbox_records)),
        "head_bbox_eval_count": int(len(head_bbox_records)),
        "phrase_inside_mass_mean": (sum(float(item["inside_mass"]) for item in phrase_bbox_records) / len(phrase_bbox_records)) if phrase_bbox_records else None,
        "head_inside_mass_mean": (sum(float(item["inside_mass"]) for item in head_bbox_records) / len(head_bbox_records)) if head_bbox_records else None,
        "phrase_argmax_in_box_rate": (sum(1 for item in phrase_bbox_records if item.get("argmax_in_box")) / len(phrase_bbox_records)) if phrase_bbox_records else None,
        "head_argmax_in_box_rate": (sum(1 for item in head_bbox_records if item.get("argmax_in_box")) / len(head_bbox_records)) if head_bbox_records else None,
        "phrase_top10_patch_in_box_rate_mean": (sum(float(item["top10_patch_in_box_rate"]) for item in phrase_bbox_records) / len(phrase_bbox_records)) if phrase_bbox_records else None,
        "head_top10_patch_in_box_rate_mean": (sum(float(item["top10_patch_in_box_rate"]) for item in head_bbox_records) / len(head_bbox_records)) if head_bbox_records else None,
        "metadata_jsonl": os.path.abspath(merged_metadata_path),
        "map_shards": sorted(set(os.path.abspath(path) for path in map_shards)),
        "analysis_paths": analysis_paths,
        "notes": notes,
    }

    summary_path = os.path.join(root_output_dir, "summary.json")
    write_json(summary_path, summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
