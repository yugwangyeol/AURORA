#!/usr/bin/env python
"""Build SteerViT priors from existing postprocessed object captions."""

import argparse
import json
import os
import sys
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
    SlotMapShardWriter,
    SteerViTExtractor,
    aggregate_historical_counts,
    build_annotation_entry,
    finalize_records,
    heatmaps_to_patch_vectors,
    load_resume_state,
    save_json,
    set_fast_math,
    set_seed,
    take_shard,
)


DEFAULT_STEERVIT_SRC = "/home/jovyan/SteerViT/src"
DEFAULT_STEERVIT_CHECKPOINT = "steervit_dinov2_base.pth"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Stage A SteerViT priors from object caption annotations.")
    parser.add_argument("--caption-input-dir", required=True, help="Directory with object_caption_annotations.json.")
    parser.add_argument("--output-dir", required=True, help="Output directory for prior cache.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=["fp16", "bf16", "fp32"], default="bf16")
    parser.add_argument("--grid-side", type=int, default=16)
    parser.add_argument("--max-slots", type=int, default=15)
    parser.add_argument("--steervit-src", default=DEFAULT_STEERVIT_SRC)
    parser.add_argument("--steervit-checkpoint", default=DEFAULT_STEERVIT_CHECKPOINT)
    parser.add_argument("--steervit-map-type", choices=("attention", "heatmap"), default="attention")
    parser.add_argument("--steervit-head-pooling", choices=("mean", "max", "min", "median"), default="mean")
    parser.add_argument("--steervit-gate-factor", type=float, default=1.0)
    parser.add_argument("--steervit-batch-size", type=int, default=128)
    parser.add_argument("--map-shard-size", type=int, default=1000)
    parser.add_argument("--max-records", type=int, default=None)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--loader-num-workers", type=int, default=8)
    parser.add_argument("--fsync-every-n-batches", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tf32", action="store_true")
    parser.add_argument("--save-debug-images", action="store_true")
    parser.add_argument("--debug-limit", type=int, default=25)
    return parser.parse_args()


def ensure_output_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def resolve_dtype(dtype_name: str, device_name: str) -> torch.dtype:
    if dtype_name == "fp16":
        dtype = torch.float16
    elif dtype_name == "bf16":
        dtype = torch.bfloat16
    else:
        dtype = torch.float32
    if str(device_name).startswith("cpu") and dtype != torch.float32:
        return torch.float32
    return dtype


def load_caption_annotations(caption_input_dir: str) -> List[Dict[str, Any]]:
    ann_path = os.path.join(caption_input_dir, "object_caption_annotations.json")
    if not os.path.isfile(ann_path):
        raise SystemExit(f"Missing object_caption_annotations.json in {caption_input_dir}")
    with open(ann_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    rows: List[Dict[str, Any]] = []
    for image_id in sorted(payload.keys(), key=lambda key: int(key) if str(key).isdigit() else str(key)):
        row = dict(payload[image_id])
        row["image_id"] = str(image_id)
        rows.append(row)
    return rows


def tensor_to_pil(image_tensor: torch.Tensor) -> Image.Image:
    array = image_tensor.detach().float().cpu().clamp(0, 1).permute(1, 2, 0).numpy()
    array = np.clip(array * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(array)


def make_attention_overlay(source: torch.Tensor, attn: torch.Tensor, color=(0.15, 0.65, 1.0)) -> torch.Tensor:
    n_patches = int(attn.numel())
    grid_side = int(n_patches ** 0.5)
    attn_2d = attn.view(grid_side, grid_side).unsqueeze(0).unsqueeze(0)
    attn_map = torch.nn.functional.interpolate(
        attn_2d, size=source.shape[-2:], mode="bilinear", align_corners=False
    )[0, 0]
    attn_map = attn_map.clamp(0.0, 1.0)
    color_t = torch.tensor(color, dtype=torch.float32).view(3, 1, 1)
    overlay = source.float() * (1.0 - 0.6 * attn_map.unsqueeze(0)) + color_t * 0.6 * attn_map.unsqueeze(0)
    return overlay.clamp(0.0, 1.0)


def make_attention_heatmap(attn: torch.Tensor) -> Image.Image:
    n_patches = int(attn.numel())
    grid_side = int(n_patches ** 0.5)
    attn_2d = attn.view(grid_side, grid_side).detach().float().cpu().numpy()
    attn_2d = np.clip(attn_2d * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(attn_2d, mode="L").resize((224, 224), resample=Image.BILINEAR)


class _SteerViTPrefetchDataset(torch.utils.data.Dataset):
    def __init__(self, entries: Sequence[Dict[str, Any]], steervit_transform) -> None:
        self.entries = list(entries)
        self.steervit_transform = steervit_transform

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        entry = self.entries[idx]
        image_path = str(entry.get("image") or "")
        image_id = str(entry.get("image_id") or Path(image_path).stem)
        try:
            if not image_path:
                raise FileNotFoundError("Missing image path in caption annotation.")
            pil = Image.open(image_path).convert("RGB")
            steervit_pixel = self.steervit_transform(pil).contiguous()
            return {
                "ok": True,
                "image_path": image_path,
                "image_id": image_id,
                "entry": entry,
                "steervit_pixel": steervit_pixel,
            }
        except Exception as exc:  # noqa: BLE001
            return {
                "ok": False,
                "image_path": image_path,
                "image_id": image_id,
                "entry": entry,
                "error": f"{type(exc).__name__}: {exc}",
            }


def _identity_collate(batch: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return batch


def main() -> None:
    args = parse_args()
    if args.grid_side != 16:
        raise ValueError("Current CaptionSlot trainer expects 16x16 = 256 patch priors.")

    ensure_output_dir(args.output_dir)
    ensure_output_dir(os.path.join(args.output_dir, "shards"))
    if args.save_debug_images:
        ensure_output_dir(os.path.join(args.output_dir, "samples"))
    set_seed(args.seed)
    set_fast_math(args.tf32)

    all_entries = load_caption_annotations(args.caption_input_dir)
    if args.max_records is not None:
        all_entries = all_entries[: args.max_records]
    global_requested_samples = len(all_entries)
    if global_requested_samples == 0:
        raise SystemExit(f"No caption annotations found in {args.caption_input_dir}")

    entries_all_for_shard = take_shard(all_entries, args.num_shards, args.shard_index)
    metadata_path = os.path.join(args.output_dir, "metadata.jsonl")
    annotations_path = os.path.join(args.output_dir, "captionslot_annotations.json")
    summary_path = os.path.join(args.output_dir, "summary.json")
    shards_dir = os.path.join(args.output_dir, "shards")

    resume_state = load_resume_state(metadata_path, annotations_path, shards_dir)
    hist_counts_merged = aggregate_historical_counts(resume_state["historical_records"])
    completed_ids = resume_state["completed_ids"]
    entries = [row for row in entries_all_for_shard if str(row["image_id"]) not in completed_ids]
    resumed_sample_count = len(entries_all_for_shard) - len(entries)

    if not entries_all_for_shard:
        if not os.path.isfile(metadata_path):
            Path(metadata_path).write_text("", encoding="utf-8")
        if not os.path.isfile(annotations_path):
            save_json(annotations_path, {})
        empty_summary = {
            "caption_input_dir": os.path.abspath(args.caption_input_dir),
            "output_dir": os.path.abspath(args.output_dir),
            "requested_samples": 0,
            "global_requested_samples": int(global_requested_samples),
            "annotation_entry_count": 0,
            "images_with_any_object": 0,
            "images_with_any_head_prior": 0,
            "total_cached_objects": 0,
            "total_valid_head_maps": 0,
            "error_count": 0,
            "metadata_jsonl": os.path.abspath(metadata_path),
            "annotations_json": os.path.abspath(annotations_path),
            "map_shards": [],
            "notes": ["This shard had no assigned samples after sharding."],
        }
        save_json(summary_path, empty_summary)
        print(json.dumps(empty_summary, indent=2, ensure_ascii=False))
        return

    if not entries:
        save_json(annotations_path, resume_state["existing_annotations"])
        resume_summary = {
            "caption_input_dir": os.path.abspath(args.caption_input_dir),
            "output_dir": os.path.abspath(args.output_dir),
            "requested_samples": int(len(entries_all_for_shard)),
            "global_requested_samples": int(global_requested_samples),
            "resumed_sample_count": int(resumed_sample_count),
            "annotation_entry_count": int(len(resume_state["existing_annotations"])),
            "images_with_any_object": int(hist_counts_merged["images_with_any_object"]),
            "images_with_any_head_prior": int(hist_counts_merged["images_with_any_head_prior"]),
            "total_cached_objects": int(hist_counts_merged["total_cached_objects"]),
            "total_valid_head_maps": int(hist_counts_merged["total_valid_head_maps"]),
            "error_count": int(hist_counts_merged["total_errors"]),
            "metadata_jsonl": os.path.abspath(metadata_path),
            "annotations_json": os.path.abspath(annotations_path),
            "map_shards": list(resume_state["existing_shard_paths"]),
            "notes": ["Resume complete: no new samples to process in this shard."],
        }
        save_json(summary_path, resume_summary)
        print(json.dumps(resume_summary, indent=2, ensure_ascii=False))
        return

    dtype = resolve_dtype(args.dtype, args.device)
    extractor = SteerViTExtractor(
        steervit_src=args.steervit_src,
        checkpoint_name=args.steervit_checkpoint,
        device=torch.device(args.device),
        dtype=dtype,
        map_type=args.steervit_map_type,
        head_pooling=args.steervit_head_pooling,
        gate_factor=args.steervit_gate_factor,
    )
    steervit_runtime = extractor.get_runtime_info()
    print("Runtime placement:")
    print(json.dumps({"steervit": steervit_runtime}, indent=2, ensure_ascii=False))

    total_errors = 0
    images_with_any_object = 0
    images_with_any_head_prior = 0
    total_cached_objects = 0
    total_valid_head_maps = 0
    debug_saved = 0

    annotations: Dict[str, Dict[str, Any]] = dict(resume_state["existing_annotations"])
    shard_writer = SlotMapShardWriter(
        os.path.join(args.output_dir, "shards"),
        shard_size=args.map_shard_size,
        start_shard_idx=int(resume_state["next_shard_idx"]),
    )
    shard_writer.shard_paths.extend(resume_state["existing_shard_paths"])

    dataset = _SteerViTPrefetchDataset(entries=entries, steervit_transform=extractor.transform)
    loader_num_workers = max(0, int(args.loader_num_workers))
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=int(args.steervit_batch_size),
        shuffle=False,
        num_workers=loader_num_workers,
        collate_fn=_identity_collate,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=loader_num_workers > 0,
        prefetch_factor=4 if loader_num_workers > 0 else None,
    )

    fsync_every = max(1, int(args.fsync_every_n_batches))
    empty_maps = torch.zeros((args.max_slots, args.grid_side * args.grid_side), dtype=torch.float32)
    empty_mask = torch.zeros(args.max_slots, dtype=torch.bool)

    def _register_error_record(record: Dict[str, Any]) -> List[Dict[str, Any]]:
        record["phrase_valid_mask"] = [False] * int(args.max_slots)
        record["head_valid_mask"] = [False] * int(args.max_slots)
        record["n_cached_chunks"] = int(min(len(record.get("noun_chunks", [])), args.max_slots))
        return shard_writer.append(
            record=record,
            image_id=record["image_id"],
            phrase_maps=empty_maps.clone(),
            head_maps=empty_maps.clone(),
            phrase_valid_mask=empty_mask.clone(),
            head_valid_mask=empty_mask.clone(),
        )

    metadata_open_mode = "a" if os.path.isfile(metadata_path) else "w"
    with open(metadata_path, metadata_open_mode, encoding="utf-8") as metadata_file, tqdm(
        total=len(entries),
        desc="StageA SteerViT Prior",
        unit="img",
        dynamic_ncols=True,
    ) as pbar:
        for batch_idx, batch in enumerate(loader, start=1):
            batch_items: List[Dict[str, Any]] = []
            for entry in batch:
                ann = dict(entry["entry"])
                record: Dict[str, Any] = {
                    "image": os.path.abspath(str(ann.get("image") or entry.get("image_path") or "")),
                    "image_id": str(ann.get("image_id") or entry["image_id"]),
                    "caption": ann.get("caption", ""),
                    "token_ids": ann.get("token_ids", []),
                    "noun_chunks": ann.get("noun_chunks", []),
                    "object_texts": ann.get("object_texts", []),
                    "clean_num_objects": int(ann.get("clean_num_objects", 0)),
                    "raw_num_objects": int(ann.get("raw_num_objects", 0)),
                    "dropped_for_token_budget": int(ann.get("dropped_for_token_budget", 0)),
                    "object_cleaning": ann.get("object_cleaning", {}),
                    "selection_strategy": "object_refexp",
                    "prior_source": "steervit_attention" if args.steervit_map_type == "attention" else "steervit_heatmap",
                    "grid_side": int(args.grid_side),
                    "max_slots": int(args.max_slots),
                    "num_shards": int(args.num_shards),
                    "shard_index": int(args.shard_index),
                }
                if not entry.get("ok", False):
                    total_errors += 1
                    record["error"] = entry.get("error", "load_failed")
                    finalized = _register_error_record(record)
                    finalize_records(finalized, metadata_file, annotations)
                    pbar.update(1)
                    continue
                batch_items.append(
                    {
                        "image_path": entry["image_path"],
                        "record": record,
                        "steervit_pixel": entry["steervit_pixel"],
                    }
                )

            if not batch_items:
                continue

            for item in batch_items:
                record = item["record"]
                record["n_cached_chunks"] = int(min(len(record.get("noun_chunks", [])), args.max_slots))
                if record["object_texts"]:
                    images_with_any_object += 1
                    total_cached_objects += min(len(record["object_texts"]), int(args.max_slots))
                item["head_maps"] = torch.zeros((args.max_slots, args.grid_side * args.grid_side), dtype=torch.float32)
                item["phrase_maps"] = torch.zeros((args.max_slots, args.grid_side * args.grid_side), dtype=torch.float32)
                item["head_valid_mask"] = torch.zeros(args.max_slots, dtype=torch.bool)
                item["phrase_valid_mask"] = torch.zeros(args.max_slots, dtype=torch.bool)

            steervit_pixels_gpu = torch.stack([item["steervit_pixel"] for item in batch_items], dim=0).to(
                extractor.device, non_blocking=True
            )

            flat_texts: List[str] = []
            flat_row_idx: List[int] = []
            flat_slot_idx: List[int] = []
            for row_idx, item in enumerate(batch_items):
                for slot_idx, text in enumerate(item["record"]["object_texts"][: args.max_slots]):
                    flat_texts.append(text)
                    flat_row_idx.append(row_idx)
                    flat_slot_idx.append(slot_idx)

            if flat_texts:
                steervit_bs = max(1, int(args.steervit_batch_size))
                for chunk_start in range(0, len(flat_texts), steervit_bs):
                    chunk_end = min(chunk_start + steervit_bs, len(flat_texts))
                    chunk_texts = flat_texts[chunk_start:chunk_end]
                    chunk_rows = flat_row_idx[chunk_start:chunk_end]
                    chunk_slots = flat_slot_idx[chunk_start:chunk_end]
                    idx_tensor = torch.tensor(chunk_rows, dtype=torch.long, device=extractor.device)
                    pixel_chunk = steervit_pixels_gpu.index_select(0, idx_tensor)
                    try:
                        chunk_heatmaps = extractor.forward_batch(pixel_values=pixel_chunk, texts=chunk_texts)
                        chunk_vectors = heatmaps_to_patch_vectors(chunk_heatmaps, grid_side=args.grid_side).detach().cpu()
                        nonempty = (chunk_vectors.sum(dim=-1) > 0.0).tolist()
                        for t_idx, (row_idx, slot_idx) in enumerate(zip(chunk_rows, chunk_slots)):
                            if not nonempty[t_idx]:
                                continue
                            v = chunk_vectors[t_idx]
                            item = batch_items[row_idx]
                            item["head_maps"][slot_idx] = v
                            item["phrase_maps"][slot_idx] = v
                            item["head_valid_mask"][slot_idx] = True
                            item["phrase_valid_mask"][slot_idx] = True
                    except Exception as exc:
                        total_errors += 1
                        err = f"{type(exc).__name__}: {exc}"
                        for row_idx in set(chunk_rows):
                            batch_items[row_idx]["record"]["error"] = err

            for item in batch_items:
                record = item["record"]
                head_valid_mask = item["head_valid_mask"]
                phrase_valid_mask = item["phrase_valid_mask"]
                record["phrase_valid_mask"] = [bool(x) for x in phrase_valid_mask.tolist()]
                record["head_valid_mask"] = [bool(x) for x in head_valid_mask.tolist()]
                any_head_prior = bool(head_valid_mask.any().item())
                record["any_head_prior"] = any_head_prior
                if any_head_prior:
                    images_with_any_head_prior += 1
                    total_valid_head_maps += int(head_valid_mask.sum().item())

                if args.save_debug_images and debug_saved < args.debug_limit and record["object_texts"]:
                    sample_dir = os.path.join(args.output_dir, "samples", record["image_id"])
                    ensure_output_dir(sample_dir)
                    input_path = os.path.join(sample_dir, "input_processed.png")
                    head_map_path = os.path.join(sample_dir, "slot0_head_map.png")
                    head_overlay_path = os.path.join(sample_dir, "slot0_head_overlay.png")
                    try:
                        pil_debug = Image.open(item["image_path"]).convert("RGB").resize((224, 224))
                        pil_debug.save(input_path)
                        make_attention_heatmap(item["head_maps"][0]).save(head_map_path)
                        source_tensor = torch.from_numpy(np.array(pil_debug).astype(np.float32) / 255.0).permute(2, 0, 1)
                        tensor_to_pil(make_attention_overlay(source_tensor, item["head_maps"][0])).save(head_overlay_path)
                        record["debug_paths"] = {
                            "input_processed": os.path.abspath(input_path),
                            "slot0_head_map": os.path.abspath(head_map_path),
                            "slot0_head_overlay": os.path.abspath(head_overlay_path),
                        }
                        save_json(os.path.join(sample_dir, "record.json"), record)
                    except Exception:
                        pass
                    debug_saved += 1

                finalized = shard_writer.append(
                    record=record,
                    image_id=record["image_id"],
                    phrase_maps=item["phrase_maps"],
                    head_maps=item["head_maps"],
                    phrase_valid_mask=phrase_valid_mask,
                    head_valid_mask=head_valid_mask,
                )
                finalize_records(finalized, metadata_file, annotations)
                pbar.update(1)

            if batch_idx % fsync_every == 0:
                metadata_file.flush()
                os.fsync(metadata_file.fileno())
            pbar.set_postfix(
                batch=batch_idx,
                obj_imgs=images_with_any_object,
                head_valid=images_with_any_head_prior,
                errors=total_errors,
                steervit_bs=args.steervit_batch_size,
            )

        finalized = shard_writer.flush()
        finalize_records(finalized, metadata_file, annotations)
        metadata_file.flush()
        os.fsync(metadata_file.fileno())

    ordered_annotations = {
        image_id: annotations[image_id]
        for image_id in sorted(annotations.keys(), key=lambda key: int(key) if str(key).isdigit() else str(key))
    }
    save_json(annotations_path, ordered_annotations)

    summary = {
        "caption_input_dir": os.path.abspath(args.caption_input_dir),
        "output_dir": os.path.abspath(args.output_dir),
        "steervit_src": os.path.abspath(args.steervit_src),
        "steervit_checkpoint": args.steervit_checkpoint,
        "steervit_map_type": args.steervit_map_type,
        "steervit_runtime": steervit_runtime,
        "grid_side": int(args.grid_side),
        "grid_tokens": int(args.grid_side * args.grid_side),
        "requested_dtype": str(dtype),
        "dtype": str(dtype),
        "steervit_batch_size": int(args.steervit_batch_size),
        "max_slots": int(args.max_slots),
        "tf32": bool(args.tf32),
        "seed": int(args.seed),
        "num_shards": int(args.num_shards),
        "shard_index": int(args.shard_index),
        "requested_samples": int(len(entries_all_for_shard)),
        "newly_processed_samples": int(len(entries)),
        "resumed_sample_count": int(resumed_sample_count),
        "global_requested_samples": int(global_requested_samples),
        "annotation_entry_count": int(len(ordered_annotations)),
        "images_with_any_object": int(images_with_any_object + hist_counts_merged["images_with_any_object"]),
        "images_with_any_head_prior": int(images_with_any_head_prior + hist_counts_merged["images_with_any_head_prior"]),
        "total_cached_objects": int(total_cached_objects + hist_counts_merged["total_cached_objects"]),
        "total_valid_head_maps": int(total_valid_head_maps + hist_counts_merged["total_valid_head_maps"]),
        "error_count": int(total_errors + hist_counts_merged["total_errors"]),
        "metadata_jsonl": os.path.abspath(metadata_path),
        "annotations_json": os.path.abspath(annotations_path),
        "map_shards": list(shard_writer.shard_paths),
        "notes": [
            "This cache is built from precomputed postprocessed object captions; no caption regeneration was performed.",
            "captionslot_annotations.json is compatible with the current CaptionSlotDataset.",
            "head_maps are SteerViT priors flattened onto the current 16x16 = 256 patch grid.",
            "phrase_maps are stored as aliases of head_maps for compatibility/debugging.",
        ],
    }
    save_json(summary_path, summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
