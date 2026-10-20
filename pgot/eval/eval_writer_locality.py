"""Measure whether one-shot visual memories stay local to their semantic owner."""

from __future__ import annotations

import argparse
import json
import logging
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from pgot.eval.eval_recon_oracles import build_loader, load_model_and_tokenizer
from pgot.eval.pgot_inference import pgot_forward_eval


LOG = logging.getLogger("pgot.eval_writer_locality")
SIZE_ORDER = ("small", "medium", "large")


def coco_style_size_bin(area_fraction: float, resolution: int = 512) -> str:
    """COCO-style 32^2/96^2 thresholds on the evaluator's square grid."""
    equivalent_area = float(area_fraction) * resolution * resolution
    if equivalent_area < 32**2:
        return "small"
    if equivalent_area < 96**2:
        return "medium"
    return "large"


def summarize(
    values: list[float], seed: int = 0,
    image_clusters: list[tuple[float, int]] | None = None,
) -> dict:
    if not values:
        return {"mean": None, "std": None, "ci95": [None, None], "count": 0}
    array = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    if image_clusters:
        sums = np.asarray([item[0] for item in image_clusters], dtype=np.float64)
        counts = np.asarray([item[1] for item in image_clusters], dtype=np.float64)
        means = np.empty(2000, dtype=np.float64)
        for index in range(means.size):
            sampled = rng.integers(0, len(image_clusters), size=len(image_clusters))
            means[index] = sums[sampled].sum() / counts[sampled].sum()
        ci = np.quantile(means, [0.025, 0.975]).tolist()
    elif array.size == 1:
        ci = [float(array[0]), float(array[0])]
    else:
        means = np.empty(2000, dtype=np.float64)
        for index in range(means.size):
            means[index] = rng.choice(array, size=array.size, replace=True).mean()
        ci = np.quantile(means, [0.025, 0.975]).tolist()
    return {
        "mean": float(array.mean()),
        "std": float(array.std()),
        "ci95": [float(ci[0]), float(ci[1])],
        "count": int(array.size),
    }


def resize_object_masks(batch: dict, k_objects: int, patches: int, device) -> tuple[torch.Tensor, torch.Tensor]:
    n = 1
    masks = batch["gt_masks_per_ovt"][:, : k_objects * n].to(device=device, dtype=torch.float32)
    valid = batch["ovt_valid_mask"][:, : k_objects * n].to(device=device, dtype=torch.bool)
    batch_size = masks.shape[0]
    source_side = int(round(math.sqrt(masks.shape[-1])))
    target_side = int(round(math.sqrt(patches)))
    if source_side * source_side != masks.shape[-1] or target_side * target_side != patches:
        raise ValueError("Writer locality expects square mask and patch grids")
    masks = F.interpolate(
        masks.reshape(batch_size * k_objects * n, 1, source_side, source_side),
        size=(target_side, target_side), mode="area",
    ).reshape(batch_size, k_objects, n, patches).amax(dim=2).clamp(0, 1)
    valid = valid.reshape(batch_size, k_objects, n).any(dim=2)
    valid &= masks.sum(dim=-1) > 0
    return masks.masked_fill(~valid.unsqueeze(-1), 0), valid


@torch.no_grad()
def run(args: argparse.Namespace) -> dict:
    model, tokenizer, device, _ = load_model_and_tokenizer(args)
    if not bool(getattr(model.config, "pgot_one_shot_reader_enable", False)):
        raise ValueError("Writer locality requires a one-shot visual-memory checkpoint")
    loader = build_loader(
        args, tokenizer, model, args.val_jsonl, shuffle=False, max_samples=args.max_samples
    )
    rows: dict[str, list[float]] = defaultdict(list)
    size_rows: dict[str, dict[str, list[float]]] = {
        name: defaultdict(list) for name in SIZE_ORDER
    }
    image_clusters: dict[str, list[tuple[float, int]]] = defaultdict(list)
    size_image_clusters: dict[str, dict[str, list[tuple[float, int]]]] = {
        name: defaultdict(list) for name in SIZE_ORDER
    }
    size_object_counts = {name: 0 for name in SIZE_ORDER}
    samples = 0
    for batch in tqdm(loader, desc="Writer locality"):
        out = pgot_forward_eval(
            model,
            images=batch["images"], target_images=batch["target_images"],
            caption_input_ids=batch["caption_input_ids"],
            caption_attention_mask=batch["caption_attention_mask"],
            ovt_positions_in_caption=batch["ovt_positions_in_caption"],
            ovt_valid_mask=batch["ovt_valid_mask"],
        )
        record = out["e8_write_records"][-1]
        weights = record.get("write_weights")
        memory_valid = record.get("memory_valid")
        if weights is None or memory_valid is None or weights.ndim != 4:
            raise ValueError("Checkpoint did not expose one-shot write_weights [B,S,J,P]")
        weights = weights.float()
        memory_valid = memory_valid.bool()
        object_valid = out["ovt_object_valid"].bool()
        batch_size, k_objects = object_valid.shape
        masks, gt_valid = resize_object_masks(batch, k_objects, weights.shape[-1], device)
        object_valid &= gt_valid
        union = masks.amax(dim=1)
        area_fraction = masks.mean(dim=-1)
        for b in range(batch_size):
            image_values: dict[str, list[float]] = defaultdict(list)
            size_image_values: dict[str, dict[str, list[float]]] = {
                name: defaultdict(list) for name in SIZE_ORDER
            }
            for k in torch.nonzero(object_valid[b], as_tuple=False).flatten().tolist():
                size_bin = coco_style_size_bin(float(area_fraction[b, k]))
                size_object_counts[size_bin] += 1
                other = (
                    masks[b, torch.arange(k_objects, device=device) != k].amax(dim=0)
                    if k_objects > 1 else torch.zeros_like(masks[b, k])
                )
                for j in torch.nonzero(memory_valid[b, k], as_tuple=False).flatten().tolist():
                    distribution = weights[b, k, j]
                    distribution = distribution / distribution.sum().clamp_min(1e-8)
                    inside = (distribution * masks[b, k]).sum()
                    object_values = {
                        "object_inside_mask_mass": float(inside),
                        "object_other_object_mass": float((distribution * other).sum()),
                        "object_background_mass": float((distribution * (1-union[b])).sum()),
                        "object_area_normalized_enrichment": float(
                            inside / area_fraction[b, k].clamp_min(1e-8)
                        ),
                    }
                    for key, value in object_values.items():
                        rows[key].append(value)
                        size_rows[size_bin][key].append(value)
                        image_values[key].append(value)
                        size_image_values[size_bin][key].append(value)
            for slot in range(k_objects, weights.shape[1]):
                for j in torch.nonzero(memory_valid[b, slot], as_tuple=False).flatten().tolist():
                    distribution = weights[b, slot, j]
                    distribution = distribution / distribution.sum().clamp_min(1e-8)
                    register_values = {
                        "register_annotated_object_mass": float((distribution * union[b]).sum()),
                        "register_background_mass": float((distribution * (1-union[b])).sum()),
                    }
                    for key, value in register_values.items():
                        rows[key].append(value)
                        image_values[key].append(value)
            for key, values in image_values.items():
                image_clusters[key].append((float(sum(values)), len(values)))
            for size_bin in SIZE_ORDER:
                for key, values in size_image_values[size_bin].items():
                    size_image_clusters[size_bin][key].append((float(sum(values)), len(values)))
        samples += batch_size

    summary = {
        "model_path": args.model_path,
        "num_samples": samples,
        "protocol": {
            "caption": "teacher-forced training-format descriptions",
            "object_metric_unit": "valid visual memory token",
            "register_metric_unit": "valid register memory token",
            "register_term": "annotated-object mass (neutral; COCO does not annotate all visual content)",
            "enrichment": "inside-mask mass divided by mask area fraction; 1 means spatially uniform",
            "object_size_bins": (
                "COCO-style thresholds on the 512x512 evaluation grid: "
                "small < 32^2 pixels, medium < 96^2 pixels, large otherwise"
            ),
            "confidence_interval": (
                "95% percentile interval from image-cluster bootstrap; token-weighted mean is "
                "preserved within each resampled image cluster"
            ),
        },
        "metrics": {
            key: summarize(value, args.seed, image_clusters[key])
            for key, value in sorted(rows.items())
        },
        "object_counts_by_size": size_object_counts,
        "object_metrics_by_size": {
            size_bin: {
                key: summarize(value, args.seed, size_image_clusters[size_bin][key])
                for key, value in sorted(size_rows[size_bin].items())
            }
            for size_bin in SIZE_ORDER
        },
        "config": {
            "reader_supervision_mode": getattr(model.config, "pgot_e8_reader_supervision_mode", None),
            "owner_weight": getattr(model.config, "pgot_e8_owner_weight", None),
            "object_memories": getattr(model.config, "pgot_e11_object_memories_per_owner", None),
            "register_memories": getattr(model.config, "pgot_e11_register_memories_per_owner", None),
        },
    }
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--val_jsonl", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--max_samples", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--grid_size", type=int, default=32)
    parser.add_argument("--max_caption_tokens", type=int, default=1024)
    parser.add_argument("--n_ovt_per_object", type=int, default=1)
    parser.add_argument("--max_objects", type=int, default=50)
    parser.add_argument("--image_preprocess_mode", default="coda_center_crop")
    parser.add_argument("--coda_crop_size", type=int, default=512)
    parser.add_argument("--dtype", choices=["fp32", "bf16"], default="fp32")
    parser.add_argument("--diffusion_inference_steps", type=int, default=2)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s :: %(message)s")
    run(args)


if __name__ == "__main__":
    main()
