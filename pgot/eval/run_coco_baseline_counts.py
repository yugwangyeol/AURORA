"""Extract per-image COCO object counts for COCO-trained slot baselines.

This is intentionally count-only: no segmentation upsampling or segmentation
metric is computed. AdaSlot and MetaSlot use their official native selection
decisions. CODA has a fixed slot capacity, which is read from its released
checkpoint config without loading the image model.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


class COCOCountDataset(Dataset):
    """Raw COCO images plus visible-instance counts from the frozen CODA cache."""

    def __init__(self, root: str, instance_cache: str, max_samples: int = 0):
        self.root = Path(root)
        self.cache = Path(instance_cache)
        meta = json.loads((self.cache / "meta.json").read_text())
        if meta.get("mode") != "coda" or meta.get("gt_granularity", "instance") != "instance":
            raise ValueError("Expected the CODA-style COCO instance cache")
        self.masks = np.load(self.cache / "masks.npy", mmap_mode="r")
        self.overlaps = np.load(self.cache / "overlap_masks.npy", mmap_mode="r")
        ids = np.load(self.cache / "image_ids.npy")
        if self.masks.shape != self.overlaps.shape or len(ids) != len(self.masks):
            raise ValueError("Invalid COCO instance cache")
        if len(set(ids.tolist())) != len(ids):
            raise ValueError("Duplicate COCO image IDs")
        limit = int(max_samples) if max_samples else len(ids)
        self.ids = ids[:limit].astype(np.int64)

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, index: int) -> dict:
        image_id = int(self.ids[index])
        image_path = self.root / "val2017" / f"{image_id:012d}.jpg"
        with Image.open(image_path) as image:
            rgb = np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
        labels = np.asarray(self.masks[index])
        valid = ~np.asarray(self.overlaps[index], dtype=bool)
        visible_ids = np.unique(labels[valid])
        gt_objects = int(np.count_nonzero(visible_ids))
        return {"id": image_id, "image": rgb, "gt_objects": gt_objects}


def collate(items: list[dict]) -> dict:
    return {
        "id": [item["id"] for item in items],
        "image": [item["image"] for item in items],
        "gt_objects": [item["gt_objects"] for item in items],
    }


def distributions(values: list[int]) -> dict:
    counts = Counter(values)
    total = len(values)
    return {
        str(key): {"n": counts[key], "fraction": counts[key] / total}
        for key in sorted(counts)
    }


def count_bucket(value: int) -> str:
    if value == 0:
        return "0"
    if value <= 3:
        return "1-3"
    if value <= 6:
        return "4-6"
    if value <= 10:
        return "7-10"
    return "11+"


def grouped_summary(records: list[dict]) -> dict:
    grouped: dict[str, list[dict]] = {}
    for row in records:
        grouped.setdefault(count_bucket(row["gt_objects"]), []).append(row)
    result = {}
    for key in ("0", "1-3", "4-6", "7-10", "11+"):
        rows = grouped.get(key)
        if not rows:
            continue
        result[key] = {
            "n": len(rows),
            "gt_count_mean": float(np.mean([row["gt_objects"] for row in rows])),
            "active_count_mean": float(np.mean([row["active_objects"] for row in rows])),
            "count_mae": float(np.mean([
                abs(row["active_objects"] - row["gt_objects"]) for row in rows
            ])),
        }
    return result


def load_baseline(args: argparse.Namespace, device: str):
    if args.model == "adaslot":
        from pgot.eval.adaslot_baseline import AdaSlotBaseline

        return AdaSlotBaseline(args, device)
    if args.model == "metaslot":
        from pgot.eval.eval_baselines_voc_zeroshot import MetaSlotBaseline

        return MetaSlotBaseline(args, device)
    raise ValueError(f"No image baseline is needed for {args.model}")


def coda_capacity(args: argparse.Namespace) -> tuple[int, dict]:
    from omegaconf import OmegaConf

    config_path = Path(args.coda_ckpt) / "config.yaml"
    config = OmegaConf.load(config_path)
    target = str(config.encoder.get("_target_", ""))
    if not target.endswith("RegisterSlotDiffusion"):
        raise ValueError(f"Unexpected CODA encoder target: {target}")
    capacity = int(config.encoder.slot_n_slots)
    if capacity < 1:
        raise ValueError("Invalid CODA slot capacity")
    return capacity, {
        "checkpoint": str(Path(args.coda_ckpt)),
        "config": str(config_path),
        "encoder": target,
        "num_slots": capacity,
        "selection": "fixed slot capacity; no per-image active-slot prediction",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=("adaslot", "metaslot", "coda"))
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--coco_root", default="/home/jovyan/data/coco")
    parser.add_argument(
        "--coco_instance_cache",
        default="/home/jovyan/PGOT/data/coco_inst_mask_cache_coda512",
    )
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--adaslot_root", default="/home/jovyan/baselines/AdaSlot")
    parser.add_argument("--adaslot_ckpt", default="/home/jovyan/baselines/ckpt/adaslot-coco.ckpt")
    parser.add_argument("--metaslot_root", default="/home/jovyan/baselines/MetaSlot")
    parser.add_argument(
        "--metaslot_cfg",
        default="/home/jovyan/baselines/MetaSlot/Config/config-metaslot/dinosaur_r-coco.py",
    )
    parser.add_argument("--metaslot_ckpt", default="/home/jovyan/baselines/ckpt/metaslot-dinosaur-coco.pth")
    parser.add_argument("--coda_ckpt", default="/home/jovyan/coda/pretrained/coda/coco")
    args = parser.parse_args()

    if min(args.batch_size, args.num_workers + 1) < 1 or args.max_samples < 0 or args.seed < 0:
        parser.error("Invalid batch size, workers, sample limit, or seed")

    threads = int(os.environ.get("OMP_NUM_THREADS", "3"))
    torch.set_num_threads(threads)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    dataset = COCOCountDataset(args.coco_root, args.coco_instance_cache, args.max_samples)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate,
    )

    started = time.time()
    records: list[dict] = []
    if args.model == "coda":
        capacity, model_info = coda_capacity(args)
        for batch in tqdm(loader, desc="CODA fixed count"):
            for image_id, gt_objects in zip(batch["id"], batch["gt_objects"]):
                records.append({
                    "id": image_id,
                    "gt_objects": gt_objects,
                    "active_objects": capacity,
                    "slot_capacity": capacity,
                })
        device = "cpu"
        peak_gpu_memory = 0.0
    else:
        if not torch.cuda.is_available():
            raise RuntimeError(f"{args.model} count extraction requires CUDA")
        device = "cuda"
        torch.cuda.reset_peak_memory_stats()
        baseline = load_baseline(args, device)
        model_info = baseline.info
        for batch in tqdm(loader, desc=f"{baseline.name} native count"):
            masks = baseline(batch["image"])
            diagnostics = baseline.last_diagnostics
            if len(diagnostics) != len(batch["id"]):
                raise RuntimeError("Native count diagnostics do not match the image batch")
            for image_id, gt_objects, diagnostic in zip(
                batch["id"], batch["gt_objects"], diagnostics
            ):
                kept = int(diagnostic["kept_slots"])
                row = {
                    "id": image_id,
                    "gt_objects": gt_objects,
                    "active_objects": kept,
                    **diagnostic,
                }
                records.append(row)
            del masks
        torch.cuda.synchronize()
        peak_gpu_memory = torch.cuda.max_memory_allocated() / 2**30

    if len(records) != len(dataset) or len({row["id"] for row in records}) != len(records):
        raise RuntimeError("Incomplete or duplicate per-image count records")

    active = [int(row["active_objects"]) for row in records]
    gt = [int(row["gt_objects"]) for row in records]
    summary = {
        "task": "coco_object_count_extraction",
        "model": {"adaslot": "AdaSlot", "metaslot": "MetaSlot", "coda": "CODA"}[args.model],
        "train_dataset": "COCO",
        "dataset": "coco",
        "split": "val2017",
        "num_samples": len(records),
        "seed": args.seed,
        "batch_size": args.batch_size,
        "device": device,
        "protocol": (
            "COCO val2017 cache order; visible GT instance IDs after CODA 512 center crop "
            "and overlap exclusion; count-only inference with no segmentation metric"
        ),
        "active_count_source": {
            "adaslot": "official hard_keep_decision.sum",
            "metaslot": "official prototype duplicate-removal keep mask sum",
            "coda": "released checkpoint fixed slot_n_slots",
        }[args.model],
        "active_count_mean": float(np.mean(active)),
        "active_count_min": min(active),
        "active_count_max": max(active),
        "gt_count_mean": float(np.mean(gt)),
        "object_count_mae": float(np.mean(np.abs(np.asarray(active) - np.asarray(gt)))),
        "active_count_distribution": distributions(active),
        "gt_count_distribution": distributions(gt),
        "by_gt_object_count": grouped_summary(records),
        "model_info": model_info,
        "elapsed_sec": time.time() - started,
        "peak_gpu_memory_GiB": peak_gpu_memory,
    }

    temporary = output / "per_image.tmp.jsonl"
    with temporary.open("w") as stream:
        for row in records:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(output / "per_image.jsonl")
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    )
    print(json.dumps({
        "model": summary["model"],
        "n": summary["num_samples"],
        "active_mean": summary["active_count_mean"],
        "gt_mean": summary["gt_count_mean"],
        "count_mae": summary["object_count_mae"],
        "output": str(output),
    }, indent=2))


if __name__ == "__main__":
    main()
