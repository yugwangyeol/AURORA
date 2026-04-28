#!/usr/bin/env python
"""Generate Stage A object-wise refexp/plain captions only.

This is the caption-only half of the object-caption -> grounding-prior pipeline.
It keeps the efficient Scale-RAE batched generation path, but skips SteerViT.

Outputs:
- predictions.jsonl: per-image detailed prediction records
- object_caption_annotations.json: compact per-image records keyed by image_id
- saved_examples_first_N.json: first few records for inspection
- summary.json: run summary + runtime placement info
"""

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.append(str(SCRIPT_DIR))

from build_stagea_object_steervit_prior_cache import (  # type: ignore
    DEFAULT_MODEL_PATH,
    DEFAULT_OBJECT_PROMPT,
    build_caption_from_object_texts,
    build_prompt,
    ensure_output_dir,
    extract_clean_object_texts,
    generate_captions_batched_kv,
    get_runtime_devices,
    get_runtime_dtypes,
    move_vision_towers_to_device,
    parse_objectwise_output,
    prepare_special_token_ids,
    resolve_dtype,
    resolve_requested_image_paths,
    save_json,
    set_fast_math,
    set_seed,
    take_shard,
    tokenize_prompt,
)
from utils.load_model import load_scale_rae_model  # type: ignore

SCALE_RAE_ASSETS_DIR = Path("/home/jovyan/Scale-RAE/inference/assets")
DEFAULT_OBJECT_PROMPT_STRUCTURED = """List the visible objects in the image, one object per line.
After the object lines, add one background line.
Output only lines in this format:
[OBJ1] refexp: ... | name: ... | location: ... | attributes: ... | action: ...
[OBJ2] refexp: ... | name: ... | location: ... | attributes: ... | action: ...
...
[BACKGROUND] background description
Rules:
- Each OBJ line should describe one distinct object.
- refexp should be the clearest phrase for identifying that object.
- name should be the object category or a short noun phrase.
- location may be any natural spatial phrase.
- attributes may mention appearance, color, size, material, or state.
- If action is unclear, write none.
- Do not impose a fixed maximum number of objects, but only include objects you can clearly see.
- Do not write an introduction, summary, or extra explanation.
- Write the descriptions in English.
"""
PROMPT_PRESET_FILES = {
    "object_refexp_detailed_en": SCALE_RAE_ASSETS_DIR / "object_refexp_detailed_en.txt",
    "object_structured_en": SCALE_RAE_ASSETS_DIR / "object_structured_en.txt",
}
PROMPT_PRESET_FALLBACKS = {
    "object_refexp_detailed_en": DEFAULT_OBJECT_PROMPT,
    "object_structured_en": DEFAULT_OBJECT_PROMPT_STRUCTURED,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate object-wise Stage A captions only.")
    parser.add_argument("--image-dir", default=None, help="Directory containing images.")
    parser.add_argument(
        "--image-list-json",
        default=None,
        help="Optional JSON containing image paths. Supports a list or {'image_paths': [...]} payload.",
    )
    parser.add_argument("--output-dir", required=True, help="Output directory.")
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=["fp16", "bf16", "fp32"], default="bf16")
    parser.add_argument("--caption-batch-size", type=int, default=64)
    parser.add_argument("--caption-max-new-tokens", type=int, default=128)
    parser.add_argument("--max-caption-tokens", type=int, default=192)
    parser.add_argument("--max-slots", type=int, default=15)
    parser.add_argument(
        "--prompt-preset",
        choices=sorted(PROMPT_PRESET_FILES.keys()),
        default="object_refexp_detailed_en",
        help="Built-in prompt preset. Ignored when --prompt or --prompt-file is provided.",
    )
    parser.add_argument("--prompt", default=None, help="Inline custom prompt text.")
    parser.add_argument("--prompt-file", default=None, help="Path to a custom prompt text file.")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--save-limit", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--disable-kv-cache", action="store_true")
    parser.add_argument("--tf32", action="store_true")
    parser.add_argument("--loader-num-workers", type=int, default=8)
    parser.add_argument("--fsync-every-n-batches", type=int, default=20)
    parser.add_argument(
        "--repair-existing-outputs",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Repair predictions/annotations/examples before resuming by removing malformed lines and duplicate image_ids.",
    )
    return parser.parse_args()


class _PrefetchCaptionDataset(torch.utils.data.Dataset):
    """Worker-side PIL decode + Scale-RAE preprocess only."""

    def __init__(self, image_paths: Sequence[str], image_processor) -> None:
        self.image_paths = list(image_paths)
        self.image_processor = image_processor

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        path = self.image_paths[idx]
        image_id = Path(path).stem
        try:
            pil = Image.open(path).convert("RGB")
            caption_pixel = self.image_processor[0].preprocess(
                pil, return_tensors="pt"
            )["pixel_values"][0].unsqueeze(0).contiguous()
            return {
                "ok": True,
                "image_path": path,
                "image_id": image_id,
                "caption_pixel": caption_pixel,
            }
        except Exception as exc:  # noqa: BLE001
            return {
                "ok": False,
                "image_path": path,
                "image_id": image_id,
                "error": f"{type(exc).__name__}: {exc}",
            }


def _identity_collate(batch: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return batch


def read_prompt_text(args: argparse.Namespace) -> tuple[str, str]:
    if args.prompt is not None:
        return args.prompt.strip(), "inline"
    if args.prompt_file is not None:
        prompt_path = Path(args.prompt_file).expanduser().resolve()
        with open(prompt_path, "r", encoding="utf-8") as f:
            return f.read().strip(), str(prompt_path)

    preset = str(args.prompt_preset)
    prompt_path = PROMPT_PRESET_FILES[preset]
    if prompt_path.is_file():
        with open(prompt_path, "r", encoding="utf-8") as f:
            return f.read().strip(), f"preset:{preset}"
    return PROMPT_PRESET_FALLBACKS[preset].strip(), f"preset_fallback:{preset}"


def atomic_write_json(path: str, payload: Any) -> None:
    parent = Path(path).resolve().parent
    fd, tmp_path = tempfile.mkstemp(prefix=".tmp_json_", suffix=".json", dir=str(parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def atomic_write_jsonl(path: str, rows: Sequence[Dict[str, Any]]) -> None:
    parent = Path(path).resolve().parent
    fd, tmp_path = tempfile.mkstemp(prefix=".tmp_jsonl_", suffix=".jsonl", dir=str(parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def sort_key(text: str):
    return (0, int(text)) if str(text).isdigit() else (1, str(text))


def record_quality_score(record: Dict[str, Any], line_idx: int) -> tuple:
    clean_num_objects = int(record.get("clean_num_objects", 0) or 0)
    raw_num_objects = int(record.get("raw_num_objects", 0) or 0)
    object_text_count = len(record.get("object_texts") or [])
    noun_chunk_count = len(record.get("noun_chunks") or [])
    raw_generation_len = len(str(record.get("raw_generation") or ""))
    caption_len = len(str(record.get("caption") or ""))
    has_error = bool(record.get("error"))
    has_background = bool(record.get("background")) or bool(record.get("has_background"))
    return (
        0 if has_error else 1,
        int(clean_num_objects > 0),
        clean_num_objects,
        int(object_text_count > 0),
        object_text_count,
        noun_chunk_count,
        raw_num_objects,
        int(has_background),
        raw_generation_len,
        caption_len,
        line_idx,
    )


def repair_existing_outputs(
    output_dir: str,
    predictions_path: str,
    annotations_path: str,
    save_limit: int,
) -> Dict[str, Any]:
    stats = {
        "predictions_present": int(os.path.isfile(predictions_path)),
        "annotations_present": int(os.path.isfile(annotations_path)),
        "total_prediction_lines": 0,
        "valid_prediction_lines": 0,
        "malformed_prediction_lines_removed": 0,
        "missing_image_id_lines_removed": 0,
        "duplicate_prediction_records_removed": 0,
        "stale_annotation_entries_removed": 0,
        "rewrote_predictions": 0,
        "rewrote_annotations": 0,
        "rewrote_examples": 0,
    }

    best_records: Dict[str, Dict[str, Any]] = {}
    best_scores: Dict[str, tuple] = {}
    if os.path.isfile(predictions_path):
        with open(predictions_path, "r", encoding="utf-8") as f:
            for line_idx, raw_line in enumerate(f, start=1):
                if not raw_line.strip():
                    continue
                stats["total_prediction_lines"] += 1
                try:
                    record = json.loads(raw_line)
                except json.JSONDecodeError:
                    stats["malformed_prediction_lines_removed"] += 1
                    continue
                image_id = str(record.get("image_id", "")).strip()
                if not image_id:
                    stats["missing_image_id_lines_removed"] += 1
                    continue
                stats["valid_prediction_lines"] += 1
                score = record_quality_score(record, line_idx)
                prev_score = best_scores.get(image_id)
                if prev_score is None:
                    best_scores[image_id] = score
                    best_records[image_id] = record
                    continue
                stats["duplicate_prediction_records_removed"] += 1
                if score > prev_score:
                    best_scores[image_id] = score
                    best_records[image_id] = record

    ordered_records = [
        best_records[image_id]
        for image_id in sorted(best_records.keys(), key=sort_key)
    ]

    existing_annotations: Dict[str, Dict[str, Any]] = {}
    if os.path.isfile(annotations_path):
        try:
            with open(annotations_path, "r", encoding="utf-8") as f:
                loaded = json.load(f)
            if isinstance(loaded, dict):
                existing_annotations = {str(key): value for key, value in loaded.items()}
        except (json.JSONDecodeError, OSError):
            existing_annotations = {}

    rebuilt_annotations = {
        str(record["image_id"]): build_annotation_entry(record)
        for record in ordered_records
    }

    if existing_annotations:
        stats["stale_annotation_entries_removed"] = max(
            0,
            len(existing_annotations) - len(set(existing_annotations.keys()) & set(rebuilt_annotations.keys())),
        )

    should_rewrite_predictions = (
        bool(os.path.isfile(predictions_path))
        and (
            stats["malformed_prediction_lines_removed"] > 0
            or stats["missing_image_id_lines_removed"] > 0
            or stats["duplicate_prediction_records_removed"] > 0
            or stats["valid_prediction_lines"] != len(ordered_records)
        )
    )
    if should_rewrite_predictions:
        atomic_write_jsonl(predictions_path, ordered_records)
        stats["rewrote_predictions"] = 1

    if existing_annotations != rebuilt_annotations:
        atomic_write_json(annotations_path, rebuilt_annotations)
        stats["rewrote_annotations"] = 1

    example_candidates = list(Path(output_dir).glob("saved_examples_first_*.json"))
    examples_path = os.path.join(output_dir, f"saved_examples_first_{save_limit}.json")
    if example_candidates or ordered_records:
        for candidate in example_candidates:
            if str(candidate.resolve()) != str(Path(examples_path).resolve()):
                candidate.unlink(missing_ok=True)
        atomic_write_json(examples_path, ordered_records[: max(0, int(save_limit))])
        stats["rewrote_examples"] = 1

    stats["unique_completed_records"] = len(ordered_records)
    return stats


def load_resume_state(predictions_path: str, annotations_path: str) -> Dict[str, Any]:
    completed_ids = set()
    historical_records: List[Dict[str, Any]] = []
    existing_annotations: Dict[str, Dict[str, Any]] = {}

    if os.path.isfile(predictions_path):
        with open(predictions_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                image_id = str(rec.get("image_id", ""))
                if not image_id:
                    continue
                completed_ids.add(image_id)
                historical_records.append(rec)

    if os.path.isfile(annotations_path):
        try:
            with open(annotations_path, "r", encoding="utf-8") as f:
                existing_annotations = json.load(f)
        except (json.JSONDecodeError, OSError):
            existing_annotations = {}

    if not existing_annotations and historical_records:
        rebuilt_annotations: Dict[str, Dict[str, Any]] = {}
        for rec in historical_records:
            image_id = str(rec.get("image_id", ""))
            if not image_id:
                continue
            rebuilt_annotations[image_id] = build_annotation_entry(rec)
        existing_annotations = rebuilt_annotations

    return {
        "completed_ids": completed_ids,
        "historical_records": historical_records,
        "existing_annotations": existing_annotations,
    }


def build_annotation_entry(record: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "image": record.get("image"),
        "prompt_source": record.get("prompt_source"),
        "caption_prompt": record.get("caption_prompt"),
        "caption_prompt_built": record.get("caption_prompt_built"),
        "raw_generation": record.get("raw_generation", ""),
        "raw_lines": record.get("raw_lines", []),
        "background": record.get("background"),
        "caption": record.get("caption", ""),
        "token_ids": record.get("token_ids", []),
        "noun_chunks": record.get("noun_chunks", []),
        "object_texts": record.get("object_texts", []),
        "clean_num_objects": int(record.get("clean_num_objects", 0)),
        "raw_num_objects": int(record.get("raw_num_objects", 0)),
        "dropped_for_token_budget": int(record.get("dropped_for_token_budget", 0)),
        "object_cleaning": record.get("object_cleaning", {}),
        "stop_reason": record.get("stop_reason"),
        "cache_mode": record.get("cache_mode"),
        "error": record.get("error"),
    }


def aggregate_historical_counts(records: Sequence[Dict[str, Any]]) -> Dict[str, int]:
    counts = {
        "total_errors": 0,
        "images_with_any_object": 0,
        "images_with_background": 0,
        "images_with_extra_lines": 0,
        "total_clean_objects": 0,
        "total_removed_duplicates": 0,
        "total_removed_truncated": 0,
        "total_removed_generic": 0,
    }
    for rec in records:
        if rec.get("error"):
            counts["total_errors"] += 1
        clean_num = int(rec.get("clean_num_objects", 0))
        if clean_num > 0:
            counts["images_with_any_object"] += 1
            counts["total_clean_objects"] += clean_num
        if rec.get("background"):
            counts["images_with_background"] += 1
        if rec.get("extra_lines"):
            counts["images_with_extra_lines"] += 1
        cleaning = rec.get("object_cleaning") or {}
        counts["total_removed_duplicates"] += int(cleaning.get("removed_duplicates", 0))
        counts["total_removed_truncated"] += int(cleaning.get("removed_truncated", 0))
        counts["total_removed_generic"] += int(cleaning.get("removed_generic", 0))
    return counts


def finalize_records(
    finalized_records: Sequence[Dict[str, Any]],
    predictions_file,
    annotations: Dict[str, Dict[str, Any]],
    example_records: List[Dict[str, Any]],
    save_limit: int,
) -> None:
    for record_out in finalized_records:
        predictions_file.write(json.dumps(record_out, ensure_ascii=False) + "\n")
        annotations[str(record_out["image_id"])] = build_annotation_entry(record_out)
        if len(example_records) < save_limit:
            example_records.append(record_out)


def main() -> None:
    args = parse_args()
    ensure_output_dir(args.output_dir)
    set_seed(args.seed)
    set_fast_math(args.tf32)

    image_paths, image_source = resolve_requested_image_paths(args)
    if args.max_samples is not None:
        image_paths = image_paths[: args.max_samples]
    global_requested_samples = len(image_paths)
    if global_requested_samples == 0:
        raise SystemExit(f"No images found for source: {image_source}")
    image_paths_all_for_shard = take_shard(image_paths, args.num_shards, args.shard_index)

    predictions_path = os.path.join(args.output_dir, "predictions.jsonl")
    annotations_path = os.path.join(args.output_dir, "object_caption_annotations.json")
    examples_path = os.path.join(args.output_dir, f"saved_examples_first_{args.save_limit}.json")
    summary_path = os.path.join(args.output_dir, "summary.json")

    repair_stats = {}
    if bool(args.repair_existing_outputs):
        repair_stats = repair_existing_outputs(
            output_dir=args.output_dir,
            predictions_path=predictions_path,
            annotations_path=annotations_path,
            save_limit=int(args.save_limit),
        )

    resume_state = load_resume_state(predictions_path, annotations_path)
    hist_counts = aggregate_historical_counts(resume_state["historical_records"])
    completed_ids = resume_state["completed_ids"]
    image_paths = [p for p in image_paths_all_for_shard if Path(p).stem not in completed_ids]
    resumed_sample_count = len(image_paths_all_for_shard) - len(image_paths)

    if not image_paths_all_for_shard:
        if not os.path.isfile(predictions_path):
            Path(predictions_path).write_text("", encoding="utf-8")
        if not os.path.isfile(annotations_path):
            save_json(annotations_path, {})
        empty_summary = {
            "image_source": os.path.abspath(image_source),
            "output_dir": os.path.abspath(args.output_dir),
            "requested_samples": 0,
            "global_requested_samples": int(global_requested_samples),
            "annotation_entry_count": 0,
            "predictions_jsonl": os.path.abspath(predictions_path),
            "annotations_json": os.path.abspath(annotations_path),
            "examples_json": os.path.abspath(examples_path),
            "notes": ["This shard had no assigned samples after sharding."],
        }
        save_json(summary_path, empty_summary)
        print(json.dumps(empty_summary, indent=2, ensure_ascii=False))
        return

    if not image_paths:
        save_json(annotations_path, resume_state["existing_annotations"])
        save_json(examples_path, [])
        resume_summary = {
            "image_source": os.path.abspath(image_source),
            "output_dir": os.path.abspath(args.output_dir),
            "requested_samples": int(len(image_paths_all_for_shard)),
            "global_requested_samples": int(global_requested_samples),
            "resumed_sample_count": int(resumed_sample_count),
            "annotation_entry_count": int(len(resume_state["existing_annotations"])),
            "images_with_any_object": int(hist_counts["images_with_any_object"]),
            "images_with_background": int(hist_counts["images_with_background"]),
            "images_with_extra_lines": int(hist_counts["images_with_extra_lines"]),
            "total_clean_objects": int(hist_counts["total_clean_objects"]),
            "error_count": int(hist_counts["total_errors"]),
            "predictions_jsonl": os.path.abspath(predictions_path),
            "annotations_json": os.path.abspath(annotations_path),
            "examples_json": os.path.abspath(examples_path),
            "notes": ["Resume complete: no new samples to process in this shard."],
        }
        save_json(summary_path, resume_summary)
        print(json.dumps(resume_summary, indent=2, ensure_ascii=False))
        return

    dtype = resolve_dtype(args.dtype, args.device)
    tokenizer, model, image_processor, _context_len = load_scale_rae_model(
        model_path=args.model_path,
        device=args.device,
        dtype=dtype,
    )
    model_device = torch.device(
        args.device if torch.cuda.is_available() or not str(args.device).startswith("cuda") else "cpu"
    )
    move_vision_towers_to_device(model, model_device, dtype)
    runtime_dtypes = get_runtime_dtypes(model)
    runtime_devices = get_runtime_devices(model)
    runtime_backbone_dtype = next(model.get_model().parameters()).dtype

    start_id, end_id, eos_id = prepare_special_token_ids(tokenizer)
    prompt_text, prompt_source = read_prompt_text(args)
    prompt_built = build_prompt(prompt_text, model_config=model.config, with_image=True, num_frames=1)
    prompt_input_ids = tokenize_prompt(prompt_built, tokenizer, device=model.device)

    runtime_report = {
        "scale_rae": {
            "device": runtime_devices,
            "dtype": runtime_dtypes,
        }
    }
    print("Runtime placement:")
    print(json.dumps(runtime_report, indent=2, ensure_ascii=False))

    total_errors = 0
    images_with_any_object = 0
    images_with_background = 0
    images_with_extra_lines = 0
    total_clean_objects = 0
    total_removed_duplicates = 0
    total_removed_truncated = 0
    total_removed_generic = 0
    caption_batch_fallback_count = 0
    annotations: Dict[str, Dict[str, Any]] = dict(resume_state["existing_annotations"])
    example_records: List[Dict[str, Any]] = []

    prefetch_dataset = _PrefetchCaptionDataset(image_paths=image_paths, image_processor=image_processor)
    loader_num_workers = max(0, int(args.loader_num_workers))
    prefetch_loader = torch.utils.data.DataLoader(
        prefetch_dataset,
        batch_size=int(args.caption_batch_size),
        shuffle=False,
        num_workers=loader_num_workers,
        collate_fn=_identity_collate,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=loader_num_workers > 0,
        prefetch_factor=4 if loader_num_workers > 0 else None,
    )

    fsync_every = max(1, int(args.fsync_every_n_batches))
    use_kv_cache = not args.disable_kv_cache

    def _error_record(base_record: Dict[str, Any]) -> Dict[str, Any]:
        base_record["caption"] = ""
        base_record["token_ids"] = []
        base_record["noun_chunks"] = []
        base_record["object_texts"] = []
        base_record["clean_num_objects"] = 0
        base_record["raw_num_objects"] = 0
        base_record["dropped_for_token_budget"] = 0
        base_record["object_cleaning"] = {}
        base_record["raw_lines"] = []
        base_record["normalized_lines"] = []
        base_record["objects"] = []
        base_record["background"] = None
        base_record["extra_lines"] = []
        return base_record

    predictions_open_mode = "a" if os.path.isfile(predictions_path) else "w"
    with open(predictions_path, predictions_open_mode, encoding="utf-8") as predictions_file, tqdm(
        total=len(image_paths),
        desc="StageA Object Captions",
        unit="img",
        dynamic_ncols=True,
    ) as pbar:
        for batch_idx, batch in enumerate(prefetch_loader, start=1):
            batch_items: List[Dict[str, Any]] = []
            for entry in batch:
                record: Dict[str, Any] = {
                    "image": os.path.abspath(entry["image_path"]),
                    "image_id": entry["image_id"],
                    "caption_prompt": prompt_text,
                    "caption_prompt_built": prompt_built,
                    "prompt_source": prompt_source,
                    "num_shards": int(args.num_shards),
                    "shard_index": int(args.shard_index),
                }
                if not entry.get("ok", False):
                    total_errors += 1
                    record["error"] = entry.get("error", "load_failed")
                    finalize_records(
                        [_error_record(record)],
                        predictions_file,
                        annotations,
                        example_records,
                        int(args.save_limit),
                    )
                    pbar.update(1)
                    continue
                batch_items.append(
                    {
                        "image_path": entry["image_path"],
                        "image_id": entry["image_id"],
                        "record": record,
                        "caption_pixel": entry["caption_pixel"],
                    }
                )

            if not batch_items:
                continue

            images_tensor = torch.stack(
                [item["caption_pixel"] for item in batch_items], dim=0
            ).to(device=model.device, dtype=runtime_backbone_dtype, non_blocking=True)
            batch_input_ids = prompt_input_ids.expand(len(batch_items), -1).contiguous()

            try:
                caption_batch = generate_captions_batched_kv(
                    model=model,
                    tokenizer=tokenizer,
                    prompt_input_ids=batch_input_ids,
                    images_tensor=images_tensor,
                    eos_token_id=eos_id,
                    max_new_tokens=args.caption_max_new_tokens,
                    start_image_token_id=start_id,
                    end_image_token_id=end_id,
                    use_kv_cache=use_kv_cache,
                )
                cache_mode = caption_batch["cache_mode"]
                for row_idx, item in enumerate(batch_items):
                    item["caption_info"] = {
                        "caption": caption_batch["captions"][row_idx],
                        "generated_ids": caption_batch["generated_ids"][row_idx],
                        "stop_reason": caption_batch["stop_reasons"][row_idx],
                        "cache_mode": cache_mode,
                    }
            except Exception as batch_exc:
                caption_batch_fallback_count += 1
                print(f"[caption-fallback] batch {batch_idx} -> single: {type(batch_exc).__name__}: {batch_exc}")
                recovered_items: List[Dict[str, Any]] = []
                for item in batch_items:
                    try:
                        single_pixels = item["caption_pixel"].unsqueeze(0).to(
                            device=model.device, dtype=runtime_backbone_dtype, non_blocking=True
                        )
                        single_result = generate_captions_batched_kv(
                            model=model,
                            tokenizer=tokenizer,
                            prompt_input_ids=prompt_input_ids,
                            images_tensor=single_pixels,
                            eos_token_id=eos_id,
                            max_new_tokens=args.caption_max_new_tokens,
                            start_image_token_id=start_id,
                            end_image_token_id=end_id,
                            use_kv_cache=use_kv_cache,
                        )
                        item["caption_info"] = {
                            "caption": single_result["captions"][0],
                            "generated_ids": single_result["generated_ids"][0],
                            "stop_reason": single_result["stop_reasons"][0],
                            "cache_mode": "fallback_single",
                        }
                        recovered_items.append(item)
                    except Exception as exc:
                        total_errors += 1
                        rec = item["record"]
                        rec["error"] = f"{type(exc).__name__}: {exc}"
                        finalize_records(
                            [_error_record(rec)],
                            predictions_file,
                            annotations,
                            example_records,
                            int(args.save_limit),
                        )
                        pbar.update(1)
                batch_items = recovered_items

            if not batch_items:
                continue

            finalized_records: List[Dict[str, Any]] = []
            for item in batch_items:
                record = item["record"]
                caption_info = item["caption_info"]
                parsed = parse_objectwise_output(caption_info["caption"])
                clean_object_texts, cleaning_stats = extract_clean_object_texts(parsed, max_slots=args.max_slots)
                caption_serialized = build_caption_from_object_texts(
                    tokenizer=tokenizer,
                    object_texts=clean_object_texts,
                    max_caption_tokens=args.max_caption_tokens,
                    max_slots=args.max_slots,
                )

                record["raw_generation"] = caption_info["caption"]
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
                record["stop_reason"] = caption_info["stop_reason"]
                record["cache_mode"] = caption_info["cache_mode"]
                record["has_background"] = bool(parsed.get("background"))

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
                finalized_records.append(record)

            finalize_records(
                finalized_records,
                predictions_file,
                annotations,
                example_records,
                int(args.save_limit),
            )
            for _ in finalized_records:
                pbar.update(1)

            if batch_idx % fsync_every == 0:
                predictions_file.flush()
                os.fsync(predictions_file.fileno())
            pbar.set_postfix(
                batch=batch_idx,
                obj_imgs=images_with_any_object,
                errors=total_errors,
                caption_bs=args.caption_batch_size,
            )

        predictions_file.flush()
        os.fsync(predictions_file.fileno())

    ordered_annotations = {
        image_id: annotations[image_id]
        for image_id in sorted(annotations.keys(), key=lambda key: int(key) if str(key).isdigit() else str(key))
    }
    save_json(annotations_path, ordered_annotations)
    save_json(examples_path, example_records)

    summary = {
        "image_source": os.path.abspath(image_source),
        "output_dir": os.path.abspath(args.output_dir),
        "model_path": args.model_path,
        "prompt": prompt_text,
        "prompt_source": prompt_source,
        "prompt_preset": args.prompt_preset,
        "prompt_built": prompt_built,
        "device": str(model.device),
        "model_device_property": runtime_devices["model_device_property"],
        "backbone_device": runtime_devices["backbone_device"],
        "lm_head_device": runtime_devices["lm_head_device"],
        "vision_tower_device": runtime_devices["vision_tower_device"],
        "mm_projector_device": runtime_devices["mm_projector_device"],
        "requested_dtype": str(dtype),
        "dtype": runtime_dtypes["model_dtype_property"],
        "backbone_dtype": runtime_dtypes["backbone_dtype"],
        "lm_head_dtype": runtime_dtypes["lm_head_dtype"],
        "vision_tower_dtype": runtime_dtypes["vision_tower_dtype"],
        "caption_batch_size": int(args.caption_batch_size),
        "caption_max_new_tokens": int(args.caption_max_new_tokens),
        "max_caption_tokens": int(args.max_caption_tokens),
        "max_slots": int(args.max_slots),
        "use_kv_cache": bool(use_kv_cache),
        "tf32": bool(args.tf32),
        "seed": int(args.seed),
        "num_shards": int(args.num_shards),
        "shard_index": int(args.shard_index),
        "requested_samples": int(len(image_paths_all_for_shard)),
        "newly_processed_samples": int(len(image_paths)),
        "resumed_sample_count": int(resumed_sample_count),
        "global_requested_samples": int(global_requested_samples),
        "annotation_entry_count": int(len(ordered_annotations)),
        "images_with_any_object": int(images_with_any_object + hist_counts["images_with_any_object"]),
        "images_with_background": int(images_with_background + hist_counts["images_with_background"]),
        "images_with_extra_lines": int(images_with_extra_lines + hist_counts["images_with_extra_lines"]),
        "total_clean_objects": int(total_clean_objects + hist_counts["total_clean_objects"]),
        "total_removed_duplicates": int(total_removed_duplicates + hist_counts["total_removed_duplicates"]),
        "total_removed_truncated": int(total_removed_truncated + hist_counts["total_removed_truncated"]),
        "total_removed_generic": int(total_removed_generic + hist_counts["total_removed_generic"]),
        "error_count": int(total_errors + hist_counts["total_errors"]),
        "caption_batch_fallback_count": int(caption_batch_fallback_count),
        "repair_existing_outputs": bool(args.repair_existing_outputs),
        "repair_stats": repair_stats,
        "predictions_jsonl": os.path.abspath(predictions_path),
        "annotations_json": os.path.abspath(annotations_path),
        "examples_json": os.path.abspath(examples_path),
        "notes": [
            "This stage stores object-wise refexp/plain captions only; no SteerViT maps are produced.",
            "object_texts are cleaned per-object captions truncated to max_slots for downstream slot use.",
            "caption stores the serialized object_texts and noun_chunks stores their token spans.",
        ],
    }
    save_json(summary_path, summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
