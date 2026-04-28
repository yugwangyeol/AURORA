#!/usr/bin/env python
"""Post-process existing Stage A object captions with updated cleaning rules.

This rewrites:
- predictions.jsonl
- object_caption_annotations.json
- saved_examples_first_N.json
- summary.json

using the stored raw_generation field from an existing merged caption output dir.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from transformers import AutoTokenizer
from tqdm import tqdm


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.append(str(SCRIPT_DIR))

from build_stagea_object_steervit_prior_cache import (  # type: ignore
    build_caption_from_object_texts,
    extract_clean_object_texts,
    parse_objectwise_output,
    save_json,
)
from generate_stagea_object_captions import build_annotation_entry  # type: ignore


DEFAULT_MODEL_PATH = "/home/jovyan/data/Scale-RAE-Qwen1.5B_DiT2.4B"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Post-process Stage A object caption outputs.")
    parser.add_argument("--input-dir", required=True, help="Existing merged caption output dir.")
    parser.add_argument("--output-dir", required=True, help="New output dir for rewritten results.")
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH, help="Tokenizer source for rebuilding token_ids/spans.")
    parser.add_argument("--max-slots", type=int, default=15)
    parser.add_argument("--max-caption-tokens", type=int, default=192)
    parser.add_argument("--save-limit", type=int, default=100)
    parser.add_argument("--max-records", type=int, default=None, help="Optional cap for smoke tests.")
    parser.add_argument("--overwrite", action="store_true", help="Allow writing into a non-empty output dir.")
    return parser.parse_args()


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def iter_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(path: str, rows: Iterable[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def ensure_output_dir(path: str, overwrite: bool) -> None:
    os.makedirs(path, exist_ok=True)
    if any(Path(path).iterdir()) and not overwrite:
        raise SystemExit(f"Output dir is not empty: {path}\nUse --overwrite or choose a new output dir.")


def normalize_sort_key(text: str):
    return (0, int(text)) if str(text).isdigit() else (1, str(text))


def estimate_total_records(predictions_path: str, source_summary: Dict[str, Any], max_records: Optional[int]) -> int:
    total = source_summary.get("annotation_entry_count")
    if isinstance(total, int) and total >= 0:
        return min(total, max_records) if max_records is not None else total

    with open(predictions_path, "r", encoding="utf-8") as f:
        total = sum(1 for line in f if line.strip())
    return min(total, max_records) if max_records is not None else total


def main() -> None:
    args = parse_args()
    input_dir = os.path.abspath(args.input_dir)
    output_dir = os.path.abspath(args.output_dir)
    ensure_output_dir(output_dir, overwrite=bool(args.overwrite))

    predictions_in = os.path.join(input_dir, "predictions.jsonl")
    summary_in = os.path.join(input_dir, "summary.json")
    if not os.path.isfile(predictions_in):
        raise SystemExit(f"Missing predictions.jsonl in {input_dir}")

    source_summary = load_json(summary_in) if os.path.isfile(summary_in) else {}
    total_records = estimate_total_records(predictions_in, source_summary, args.max_records)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=False)

    rewritten_records: List[Dict[str, Any]] = []
    annotations: Dict[str, Dict[str, Any]] = {}
    saved_examples: List[Dict[str, Any]] = []

    total_removed_duplicates = 0
    total_removed_truncated = 0
    total_removed_generic = 0
    images_with_any_object = 0
    images_with_background = 0
    images_with_extra_lines = 0
    total_clean_objects = 0
    changed_records = 0
    decreased_object_count_records = 0

    with tqdm(
        total=total_records,
        desc="Postprocess StageA Captions",
        unit="img",
        dynamic_ncols=True,
    ) as pbar:
        for idx, record in enumerate(iter_jsonl(predictions_in)):
            if args.max_records is not None and idx >= args.max_records:
                break

            original_clean = int(record.get("clean_num_objects", 0))
            raw_generation = str(record.get("raw_generation") or "")
            if not raw_generation:
                raw_lines = record.get("raw_lines") or []
                raw_generation = "\n".join(str(line) for line in raw_lines if line)

            parsed = parse_objectwise_output(raw_generation)
            clean_object_texts, cleaning_stats = extract_clean_object_texts(parsed, max_slots=args.max_slots)
            caption_serialized = build_caption_from_object_texts(
                tokenizer=tokenizer,
                object_texts=clean_object_texts,
                max_caption_tokens=args.max_caption_tokens,
                max_slots=args.max_slots,
            )

            record["raw_generation"] = raw_generation
            record["raw_lines"] = parsed["raw_lines"]
            record["normalized_lines"] = parsed["normalized_lines"]
            record["objects"] = parsed["objects"]
            record["background"] = parsed["background"]
            record["extra_lines"] = parsed["extra_lines"]
            record["caption"] = caption_serialized["caption"]
            record["token_ids"] = caption_serialized["token_ids"]
            record["noun_chunks"] = caption_serialized["noun_chunks"]
            record["object_texts"] = caption_serialized["object_texts"]
            record["clean_num_objects"] = int(len(caption_serialized["object_texts"]))
            record["raw_num_objects"] = int(parsed.get("num_objects", 0))
            record["dropped_for_token_budget"] = int(caption_serialized["dropped_for_token_budget"])
            record["object_cleaning"] = cleaning_stats
            record["has_background"] = bool(parsed.get("background"))

            if record["clean_num_objects"] != original_clean:
                changed_records += 1
            if record["clean_num_objects"] < original_clean:
                decreased_object_count_records += 1

            if record["clean_num_objects"] > 0:
                images_with_any_object += 1
                total_clean_objects += int(record["clean_num_objects"])
            if record["has_background"]:
                images_with_background += 1
            if record["extra_lines"]:
                images_with_extra_lines += 1

            total_removed_duplicates += int(cleaning_stats.get("removed_duplicates", 0))
            total_removed_truncated += int(cleaning_stats.get("removed_truncated", 0))
            total_removed_generic += int(cleaning_stats.get("removed_generic", 0))

            rewritten_records.append(record)
            annotations[str(record["image_id"])] = build_annotation_entry(record)
            if len(saved_examples) < args.save_limit:
                saved_examples.append(record)

            pbar.update(1)
            if (idx + 1) % 500 == 0 or (idx + 1) == total_records:
                pbar.set_postfix(
                    changed=changed_records,
                    clean_objs=total_clean_objects,
                    trunc_rm=total_removed_truncated,
                    dup_rm=total_removed_duplicates,
                )

    rewritten_records.sort(key=lambda row: normalize_sort_key(str(row.get("image_id", ""))))
    annotations = {
        image_id: annotations[image_id]
        for image_id in sorted(annotations.keys(), key=normalize_sort_key)
    }

    predictions_out = os.path.join(output_dir, "predictions.jsonl")
    annotations_out = os.path.join(output_dir, "object_caption_annotations.json")
    examples_out = os.path.join(output_dir, f"saved_examples_first_{min(args.save_limit, len(saved_examples))}.json")
    summary_out = os.path.join(output_dir, "summary.json")

    write_jsonl(predictions_out, rewritten_records)
    save_json(annotations_out, annotations)
    save_json(examples_out, saved_examples)

    summary = {
        "input_dir": input_dir,
        "output_dir": output_dir,
        "model_path": args.model_path,
        "max_slots": int(args.max_slots),
        "max_caption_tokens": int(args.max_caption_tokens),
        "processed_records": len(rewritten_records),
        "annotation_entry_count": len(annotations),
        "images_with_any_object": images_with_any_object,
        "images_with_background": images_with_background,
        "images_with_extra_lines": images_with_extra_lines,
        "total_clean_objects": total_clean_objects,
        "total_removed_duplicates": total_removed_duplicates,
        "total_removed_truncated": total_removed_truncated,
        "total_removed_generic": total_removed_generic,
        "changed_records": changed_records,
        "decreased_object_count_records": decreased_object_count_records,
        "predictions_jsonl": predictions_out,
        "annotations_json": annotations_out,
        "examples_json": examples_out,
        "source_summary": source_summary,
        "notes": [
            "Post-processed from existing raw_generation outputs; no caption regeneration was performed.",
            "Updated cleaning includes exact dedup, truncated-line removal, generic-junk removal, and conservative prefix duplicate filtering.",
        ],
    }
    save_json(summary_out, summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
