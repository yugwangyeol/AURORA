#!/usr/bin/env python
"""
Precompute DINO patch features for AURORA source images.

This script deduplicates source images by ``base_image_id`` from a manifest and writes
one feature file per source image. By default it saves to:

    <image_root>/<base_image_id>/dino_256x768_fp16.pt

Each saved file is a dict with:
    {
        "features": Tensor[256, 768],
        "image_id": str,
        "source_path": str,
        "model_name": str,
        "target_grid": int,
    }

Example:
    PYTHONNOUSERSITE=1 /home/jovyan/.conda/envs/scale_rae/bin/python \
        /home/jovyan/Scale-RAE/scripts/precompute_aurora_dino.py \
        --manifest-path /home/jovyan/processed_coco/training_data_v4_patch/training_manifest_patch_v4_0_to_None.json \
        --image-root /home/jovyan/processed_coco/training_data_v4_patch \
        --device cuda:1 \
        --batch-size 64
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

import torch
import torch.nn.functional as F
from PIL import Image, PngImagePlugin
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoImageProcessor, Dinov2Model


@dataclass(frozen=True)
class SourceItem:
    image_id: str
    source_path: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Precompute AURORA DINO features.")
    parser.add_argument("--manifest-path", type=str, required=True)
    parser.add_argument("--image-root", type=str, required=True)
    parser.add_argument("--output-root", type=str, default=None)
    parser.add_argument("--output-name", type=str, default="dino_256x768_fp16.pt")
    parser.add_argument("--model-name", type=str, default="facebook/dinov2-base")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--target-grid", type=int, default=16)
    parser.add_argument("--save-dtype", type=str, default="fp16", choices=["fp16", "bf16", "fp32"])
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--end-index", type=int, default=None)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--png-max-text-chunk-mb", type=int, default=64)
    parser.add_argument("--png-max-text-memory-mb", type=int, default=256)
    parser.add_argument("--fail-on-image-error", action="store_true")
    parser.add_argument("--error-log-path", type=str, default=None)
    return parser.parse_args()


def load_manifest(path: Path) -> Sequence[dict]:
    with path.open("r", encoding="utf-8") as f:
        while True:
            ch = f.read(1)
            if not ch:
                return []
            if not ch.isspace():
                f.seek(0)
                if ch == "[":
                    return json.load(f)
                return [json.loads(line) for line in f if line.strip()]


def collect_unique_sources(entries: Sequence[dict]) -> List[SourceItem]:
    seen: Dict[str, SourceItem] = {}
    for item in entries:
        image_id = str(item.get("base_image_id") or item.get("image_id") or "")
        source_path = item.get("source_path") or item.get("image")
        if not image_id or not source_path:
            continue
        if image_id not in seen:
            seen[image_id] = SourceItem(image_id=image_id, source_path=source_path)
    return sorted(seen.values(), key=lambda x: x.image_id)


def select_shard(items: Sequence[SourceItem], num_shards: int, shard_index: int) -> List[SourceItem]:
    if num_shards <= 1:
        return list(items)
    return [item for idx, item in enumerate(items) if idx % num_shards == shard_index]


def get_save_dtype(name: str) -> torch.dtype:
    if name == "fp16":
        return torch.float16
    if name == "bf16":
        return torch.bfloat16
    return torch.float32


def configure_png_text_limits(max_text_chunk_mb: int, max_text_memory_mb: int) -> None:
    chunk_bytes = max_text_chunk_mb * 1024 * 1024
    memory_bytes = max_text_memory_mb * 1024 * 1024
    PngImagePlugin.MAX_TEXT_CHUNK = max(PngImagePlugin.MAX_TEXT_CHUNK, chunk_bytes)
    PngImagePlugin.MAX_TEXT_MEMORY = max(PngImagePlugin.MAX_TEXT_MEMORY, memory_bytes)


def interpolate_patch_tokens(tokens: torch.Tensor, target_grid: int) -> torch.Tensor:
    target_tokens = target_grid * target_grid
    if tokens.shape[1] == target_tokens:
        return tokens
    side = int(tokens.shape[1] ** 0.5)
    tokens = tokens.view(tokens.shape[0], side, side, tokens.shape[-1]).permute(0, 3, 1, 2)
    tokens = F.interpolate(tokens.float(), size=(target_grid, target_grid), mode="bilinear", align_corners=False)
    return tokens.permute(0, 2, 3, 1).reshape(tokens.shape[0], target_tokens, -1)


class SourceDataset(Dataset):
    def __init__(
        self,
        items: Sequence[SourceItem],
        image_root: Path,
        *,
        png_max_text_chunk_mb: int,
        png_max_text_memory_mb: int,
        fail_on_image_error: bool,
    ):
        self.items = list(items)
        self.image_root = image_root
        self.png_max_text_chunk_mb = png_max_text_chunk_mb
        self.png_max_text_memory_mb = png_max_text_memory_mb
        self.fail_on_image_error = fail_on_image_error

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        item = self.items[idx]
        path = self.image_root / item.source_path
        configure_png_text_limits(self.png_max_text_chunk_mb, self.png_max_text_memory_mb)
        try:
            with Image.open(path) as image:
                image = image.convert("RGB")
        except Exception as exc:
            if self.fail_on_image_error:
                raise
            return {
                "image_id": item.image_id,
                "source_path": item.source_path,
                "image": None,
                "load_error": {
                    "image_id": item.image_id,
                    "source_path": item.source_path,
                    "path": str(path),
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                },
            }
        return {
            "image_id": item.image_id,
            "source_path": item.source_path,
            "image": image,
            "load_error": None,
        }


def build_collate_fn(processor):
    def collate(instances: Sequence[dict]) -> dict:
        valid_instances = [inst for inst in instances if inst["image"] is not None]
        load_errors = [inst["load_error"] for inst in instances if inst["load_error"] is not None]
        pixel_values = None
        if valid_instances:
            images = [inst["image"] for inst in valid_instances]
            pixel_values = processor(images=images, return_tensors="pt")["pixel_values"]
        return {
            "image_ids": [inst["image_id"] for inst in valid_instances],
            "source_paths": [inst["source_path"] for inst in valid_instances],
            "pixel_values": pixel_values,
            "load_errors": load_errors,
        }

    return collate


def resolve_output_path(image_id: str, image_root: Path, output_root: Path | None, output_name: str) -> Path:
    root = output_root if output_root is not None else image_root
    return root / image_id / output_name


def resolve_error_log_path(image_root: Path, output_root: Path | None, explicit_path: str | None) -> Path:
    if explicit_path:
        return Path(explicit_path)
    root = output_root if output_root is not None else image_root
    return root / "_aurora_dino_image_errors.jsonl"


def filter_already_processed(
    items: Sequence[SourceItem],
    image_root: Path,
    output_root: Path | None,
    output_name: str,
    overwrite: bool,
) -> tuple[List[SourceItem], int]:
    if overwrite:
        return list(items), 0

    remaining: List[SourceItem] = []
    skipped = 0
    for item in items:
        output_path = resolve_output_path(item.image_id, image_root, output_root, output_name)
        if output_path.exists():
            skipped += 1
        else:
            remaining.append(item)
    return remaining, skipped


def append_error_records(path: Path, records: Sequence[dict]) -> None:
    if not records:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    manifest_path = Path(args.manifest_path)
    image_root = Path(args.image_root)
    output_root = Path(args.output_root) if args.output_root else None
    error_log_path = resolve_error_log_path(image_root, output_root, args.error_log_path)

    if args.shard_index < 0 or args.shard_index >= args.num_shards:
        raise ValueError("shard-index must satisfy 0 <= shard-index < num-shards")

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device requested but unavailable: {args.device}")

    entries = load_manifest(manifest_path)
    items = collect_unique_sources(entries)
    items = select_shard(items, args.num_shards, args.shard_index)

    if args.start_index:
        items = items[args.start_index :]
    if args.end_index is not None:
        items = items[: max(args.end_index - args.start_index, 0)]
    if args.limit is not None:
        items = items[: args.limit]

    configure_png_text_limits(args.png_max_text_chunk_mb, args.png_max_text_memory_mb)
    total_items = len(items)
    items, skipped_existing = filter_already_processed(
        items,
        image_root=image_root,
        output_root=output_root,
        output_name=args.output_name,
        overwrite=args.overwrite,
    )

    print(
        json.dumps(
            {
                "manifest_path": str(manifest_path),
                "total_items": total_items,
                "remaining_items": len(items),
                "skipped_existing": skipped_existing,
                "overwrite": args.overwrite,
            },
            ensure_ascii=False,
        )
    )

    if not items:
        print(
            json.dumps(
                {
                    "manifest_path": str(manifest_path),
                    "image_root": str(image_root),
                    "output_root": str(output_root) if output_root is not None else None,
                    "output_name": args.output_name,
                    "model_name": args.model_name,
                    "processed": 0,
                    "skipped": skipped_existing,
                    "failed": 0,
                    "target_grid": args.target_grid,
                    "save_dtype": args.save_dtype,
                    "num_shards": args.num_shards,
                    "shard_index": args.shard_index,
                    "error_log_path": str(error_log_path),
                },
                ensure_ascii=False,
            )
        )
        return

    processor = AutoImageProcessor.from_pretrained(args.model_name)
    model = Dinov2Model.from_pretrained(args.model_name)
    model.eval().to(device)

    dataset = SourceDataset(
        items,
        image_root=image_root,
        png_max_text_chunk_mb=args.png_max_text_chunk_mb,
        png_max_text_memory_mb=args.png_max_text_memory_mb,
        fail_on_image_error=args.fail_on_image_error,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=build_collate_fn(processor),
    )

    save_dtype = get_save_dtype(args.save_dtype)
    processed = 0
    skipped = skipped_existing
    failed = 0

    progress = tqdm(loader, total=len(loader), desc="Precomputing DINO")
    for batch in progress:
        load_errors: List[dict] = batch["load_errors"]
        if load_errors:
            failed += len(load_errors)
            append_error_records(error_log_path, load_errors)
            for error in load_errors:
                tqdm.write(
                    f"[image-error] {error['image_id']} {error['path']} :: "
                    f"{error['error_type']}: {error['error']}"
                )

        image_ids: List[str] = batch["image_ids"]
        source_paths: List[str] = batch["source_paths"]
        pixel_values = batch["pixel_values"]

        if pixel_values is None:
            progress.set_postfix(processed=processed, skipped=skipped, failed=failed)
            continue

        pixel_values = pixel_values.to(device)

        with torch.inference_mode():
            outputs = model(pixel_values=pixel_values)
            patch_tokens = outputs.last_hidden_state[:, 1:, :]
            patch_tokens = interpolate_patch_tokens(patch_tokens, target_grid=args.target_grid)
            patch_tokens = patch_tokens.to(dtype=save_dtype).cpu()

        for image_id, source_path, features in zip(image_ids, source_paths, patch_tokens):
            output_path = resolve_output_path(image_id, image_root, output_root, args.output_name)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "features": features.contiguous(),
                    "image_id": image_id,
                    "source_path": source_path,
                    "model_name": args.model_name,
                    "target_grid": args.target_grid,
                },
                output_path,
            )
            processed += 1

        progress.set_postfix(processed=processed, skipped=skipped, failed=failed)

    print(
        json.dumps(
            {
                "manifest_path": str(manifest_path),
                "image_root": str(image_root),
                "output_root": str(output_root) if output_root is not None else None,
                "output_name": args.output_name,
                "model_name": args.model_name,
                "processed": processed,
                "skipped": skipped,
                "failed": failed,
                "target_grid": args.target_grid,
                "save_dtype": args.save_dtype,
                "num_shards": args.num_shards,
                "shard_index": args.shard_index,
                "error_log_path": str(error_log_path),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
