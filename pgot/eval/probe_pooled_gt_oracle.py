"""Pooled-GT latent quantization stress test for E11 Dual-M4.

This is an inference-only diagnostic.  GT target SigLIP/RAE latent patches are
partitioned independently inside each GT object and inside the aggregate
background.  Every patch is then replaced by its partition's mean latent and
the resulting piecewise-constant map is decoded by the frozen RAE decoder.

The result answers a narrow question: how much distortion is introduced by a
given number of region-level vectors under an oracle assignment?  It is *not*
an end-to-end architecture ceiling: source-image encoding, predicted
ownership, Writer/Reader learning, DiT conditioning, and on-manifold latent
generation are all bypassed.  In particular, the Scale-RAE decoder is known to
be brittle to averaged/off-manifold SigLIP features, so exact-GT is always
reported as a required decoder control.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

import torch
import torch.nn.functional as F
from tqdm import tqdm

from pgot.eval.eval_recon_oracles import (
    build_loader,
    encode_gt_siglip,
    load_model_and_tokenizer,
)
from pgot.eval.pgot_metrics import FIDAccumulator, compute_recon_metrics
from pgot.eval.run_eval import decode_to_image, denormalize_images, load_rae_decoder


logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s :: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("pgot.pooled_gt_oracle")


@dataclass(frozen=True)
class Setting:
    mode: str
    object_codes: int
    background_codes: int

    @property
    def name(self) -> str:
        return f"{self.mode}_obj{self.object_codes}_bg{self.background_codes}"


def _parse_positive_ints(spec: str, name: str) -> List[int]:
    values = [int(x.strip()) for x in str(spec).split(",") if x.strip()]
    if not values or any(x <= 0 for x in values):
        raise ValueError(f"{name} must contain positive comma-separated integers")
    return list(dict.fromkeys(values))


def _parse_modes(spec: str) -> List[str]:
    modes = [x.strip().lower() for x in str(spec).split(",") if x.strip()]
    allowed = {"coordinate", "kmeans"}
    bad = [x for x in modes if x not in allowed]
    if not modes or bad:
        raise ValueError(f"cluster_modes must use {sorted(allowed)}; bad={bad}")
    return list(dict.fromkeys(modes))


def _coordinate_groups(
    indices: torch.Tensor,
    coords: torch.Tensor,
    n_groups: int,
) -> List[torch.Tensor]:
    """Deterministic recursive median split using only patch coordinates."""
    if indices.numel() == 0:
        return []
    if n_groups <= 1 or indices.numel() <= 1:
        return [indices]

    xy = coords[indices]
    spread = xy.max(dim=0).values - xy.min(dim=0).values
    axis = int(torch.argmax(spread).item())
    order = torch.argsort(xy[:, axis])
    ordered = indices[order]
    half = max(1, ordered.numel() // 2)
    left, right = ordered[:half], ordered[half:]
    n_left = max(1, n_groups // 2)
    n_right = max(1, n_groups - n_left)

    groups: List[torch.Tensor] = []
    if left.numel() > 0:
        groups.extend(_coordinate_groups(left, coords, n_left))
    if right.numel() > 0:
        groups.extend(_coordinate_groups(right, coords, n_right))
    return groups


def _squared_distances(x: torch.Tensor, centers: torch.Tensor) -> torch.Tensor:
    return (
        x.float().pow(2).sum(dim=-1, keepdim=True)
        + centers.float().pow(2).sum(dim=-1).unsqueeze(0)
        - 2.0 * x.float() @ centers.float().transpose(0, 1)
    ).clamp_min_(0.0)


def _kmeans_groups(
    indices: torch.Tensor,
    features: torch.Tensor,
    n_groups: int,
    iterations: int,
) -> List[torch.Tensor]:
    """Deterministic farthest-first k-means in GT latent space.

    This intentionally uses target features and is therefore an optimistic
    quantization bound, not an inference-time routing algorithm.
    """
    if indices.numel() == 0:
        return []
    n_clusters = min(max(int(n_groups), 1), int(indices.numel()))
    if n_clusters == 1:
        return [indices]

    x = features[indices].float()
    mean = x.mean(dim=0, keepdim=True)
    first = int((x - mean).pow(2).sum(dim=-1).argmax().item())
    chosen = [first]
    min_dist = _squared_distances(x, x[first : first + 1]).squeeze(1)
    for _ in range(1, n_clusters):
        next_idx = int(min_dist.argmax().item())
        chosen.append(next_idx)
        dist = _squared_distances(x, x[next_idx : next_idx + 1]).squeeze(1)
        min_dist = torch.minimum(min_dist, dist)
    centers = x[torch.tensor(chosen, device=x.device, dtype=torch.long)].clone()

    assignment = torch.zeros(x.shape[0], device=x.device, dtype=torch.long)
    for _ in range(max(int(iterations), 1)):
        assignment = _squared_distances(x, centers).argmin(dim=1)
        new_centers = centers.clone()
        min_to_center = _squared_distances(x, centers).min(dim=1).values
        for cluster_idx in range(n_clusters):
            members = assignment == cluster_idx
            if bool(members.any()):
                new_centers[cluster_idx] = x[members].mean(dim=0)
            else:
                # Deterministic empty-cluster recovery.
                replacement = int(min_to_center.argmax().item())
                new_centers[cluster_idx] = x[replacement]
                min_to_center[replacement] = -1.0
        if torch.equal(
            _squared_distances(x, centers).argmin(dim=1),
            _squared_distances(x, new_centers).argmin(dim=1),
        ):
            centers = new_centers
            break
        centers = new_centers
    assignment = _squared_distances(x, centers).argmin(dim=1)

    groups = [
        indices[(assignment == cluster_idx).nonzero(as_tuple=False).flatten()]
        for cluster_idx in range(n_clusters)
    ]
    return [group for group in groups if group.numel() > 0]


def _partition(
    indices: torch.Tensor,
    *,
    features: torch.Tensor,
    coords: torch.Tensor,
    n_groups: int,
    mode: str,
    kmeans_iterations: int,
) -> List[torch.Tensor]:
    if mode == "coordinate":
        return _coordinate_groups(indices, coords, n_groups)
    if mode == "kmeans":
        return _kmeans_groups(indices, features, n_groups, kmeans_iterations)
    raise ValueError(f"Unknown clustering mode: {mode}")


@torch.no_grad()
def pooled_gt_map(
    gt_latent: torch.Tensor,
    object_masks: torch.Tensor,
    coords: torch.Tensor,
    setting: Setting,
    *,
    fg_threshold: float,
    kmeans_iterations: int,
) -> tuple[torch.Tensor, Dict[str, float], torch.Tensor]:
    """Quantize one [P,C] GT map with per-object and aggregate-BG budgets."""
    num_patches = gt_latent.shape[0]
    if object_masks.numel() > 0:
        best, owner = object_masks.max(dim=0)
        owner = torch.where(
            best > float(fg_threshold),
            owner,
            torch.full_like(owner, -1),
        )
    else:
        owner = torch.full(
            (num_patches,), -1, device=gt_latent.device, dtype=torch.long
        )

    output = torch.empty_like(gt_latent)
    assigned = torch.zeros(num_patches, device=gt_latent.device, dtype=torch.bool)
    object_code_count = 0
    object_region_count = 0

    for object_idx in range(object_masks.shape[0]):
        indices = (owner == object_idx).nonzero(as_tuple=False).flatten()
        if indices.numel() == 0:
            continue
        object_region_count += 1
        groups = _partition(
            indices,
            features=gt_latent,
            coords=coords,
            n_groups=setting.object_codes,
            mode=setting.mode,
            kmeans_iterations=kmeans_iterations,
        )
        for group in groups:
            output[group] = gt_latent[group].mean(dim=0, keepdim=True)
            assigned[group] = True
        object_code_count += len(groups)

    background_indices = (owner == -1).nonzero(as_tuple=False).flatten()
    background_groups = _partition(
        background_indices,
        features=gt_latent,
        coords=coords,
        n_groups=setting.background_codes,
        mode=setting.mode,
        kmeans_iterations=kmeans_iterations,
    )
    for group in background_groups:
        output[group] = gt_latent[group].mean(dim=0, keepdim=True)
        assigned[group] = True

    # This should only be reachable for malformed masks; preserve exact GT as
    # a finite fallback rather than decoding uninitialized values.
    output[~assigned] = gt_latent[~assigned]
    foreground = owner >= 0
    stats = {
        "object_regions": float(object_region_count),
        "object_codes": float(object_code_count),
        "background_codes": float(len(background_groups)),
        "total_codes": float(object_code_count + len(background_groups)),
        "foreground_fraction": float(foreground.float().mean().item()),
    }
    return output, stats, foreground


def _mean(values: Sequence[float]) -> float:
    return float(sum(values) / max(len(values), 1))


@torch.no_grad()
def run(args, model, loader, decoder, device) -> List[Dict[str, float]]:
    object_codes = _parse_positive_ints(args.object_codes, "object_codes")
    background_codes = _parse_positive_ints(args.background_codes, "background_codes")
    modes = _parse_modes(args.cluster_modes)
    settings = [
        Setting(mode=mode, object_codes=obj, background_codes=bg)
        for mode in modes
        for obj in object_codes
        for bg in background_codes
    ]
    names = ["exact_gt"] + [setting.name for setting in settings]

    fid = {name: FIDAccumulator(device=device, feature=2048) for name in names}
    metrics = {
        name: {key: [] for key in ("psnr", "ssim", "mse", "mae")}
        for name in names
    }
    latent_mse_fg = {name: [] for name in names}
    latent_mse_bg = {name: [] for name in names}
    code_stats = {
        name: {
            key: []
            for key in (
                "object_regions",
                "object_codes",
                "background_codes",
                "total_codes",
                "foreground_fraction",
            )
        }
        for name in names
    }

    vision_towers = model.get_vision_tower_aux_list()
    target_processor = (
        vision_towers[1].image_processor
        if len(vision_towers) > 1
        else vision_towers[0].image_processor
    )
    target_mean = torch.tensor(target_processor.image_mean)
    target_std = torch.tensor(target_processor.image_std)
    n_ovt = max(int(args.n_ovt_per_object), 1)
    coords_cache: Dict[int, torch.Tensor] = {}
    num_samples = 0

    for batch in tqdm(loader, desc="Pooled-GT oracle"):
        gt_latent = encode_gt_siglip(model, batch, device)
        batch_size, num_patches, _ = gt_latent.shape
        side = int(round(math.sqrt(num_patches)))
        if side * side != num_patches:
            raise ValueError(f"Target latent must use a square grid, got P={num_patches}")
        if side not in coords_cache:
            yy, xx = torch.meshgrid(
                torch.linspace(-1.0, 1.0, side, device=device),
                torch.linspace(-1.0, 1.0, side, device=device),
                indexing="ij",
            )
            coords_cache[side] = torch.stack(
                [xx.flatten(), yy.flatten()], dim=-1
            )
        coords = coords_cache[side]

        masks = batch["gt_rae_masks_per_ovt"].to(device=device, dtype=torch.float32)
        valid = batch["ovt_valid_mask"].to(device=device, dtype=torch.bool)
        max_objects = masks.shape[1] // n_ovt
        masks = masks[:, : max_objects * n_ovt]
        masks = masks.reshape(batch_size, max_objects, n_ovt, -1).amax(dim=2)
        valid_objects = (
            valid[:, : max_objects * n_ovt]
            .reshape(batch_size, max_objects, n_ovt)
            .any(dim=2)
        )
        if masks.shape[-1] != num_patches:
            source_side = int(round(math.sqrt(masks.shape[-1])))
            if source_side * source_side != masks.shape[-1]:
                raise ValueError(
                    f"Object masks must use a square grid, got P={masks.shape[-1]}"
                )
            masks = F.interpolate(
                masks.reshape(batch_size * max_objects, 1, source_side, source_side),
                size=(side, side),
                mode="area",
            ).reshape(batch_size, max_objects, num_patches)
        masks = masks.clamp(0.0, 1.0) * valid_objects.unsqueeze(-1).float()

        real = denormalize_images(
            batch["target_images"].to(device).float(), target_mean, target_std
        )
        generated: Dict[str, torch.Tensor] = {"exact_gt": gt_latent}
        foreground_masks: Dict[str, torch.Tensor] = {}
        for setting in settings:
            quantized = torch.empty_like(gt_latent)
            fg_batch = torch.zeros(
                batch_size, num_patches, device=device, dtype=torch.bool
            )
            for batch_idx in range(batch_size):
                active_masks = masks[batch_idx][valid_objects[batch_idx]]
                quantized[batch_idx], stats, foreground = pooled_gt_map(
                    gt_latent[batch_idx],
                    active_masks,
                    coords,
                    setting,
                    fg_threshold=float(args.fg_threshold),
                    kmeans_iterations=int(args.kmeans_iterations),
                )
                fg_batch[batch_idx] = foreground
                for key, value in stats.items():
                    code_stats[setting.name][key].append(float(value))
            generated[setting.name] = quantized
            foreground_masks[setting.name] = fg_batch

        for name, latent in generated.items():
            fake = decode_to_image(decoder, latent, device)
            real_for_metric = real
            if real_for_metric.shape[-2:] != fake.shape[-2:]:
                real_for_metric = F.interpolate(
                    real_for_metric,
                    size=fake.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
            fid[name].add(real_for_metric, fake)
            batch_metrics = compute_recon_metrics(real_for_metric, fake)
            for key in metrics[name]:
                metrics[name][key].extend(
                    [float(x) for x in batch_metrics[key].detach().cpu()]
                )

            if name != "exact_gt":
                patch_mse = (latent.float() - gt_latent.float()).pow(2).mean(dim=-1)
                fg = foreground_masks[name]
                for batch_idx in range(batch_size):
                    if bool(fg[batch_idx].any()):
                        latent_mse_fg[name].append(
                            float(patch_mse[batch_idx][fg[batch_idx]].mean().item())
                        )
                    if bool((~fg[batch_idx]).any()):
                        latent_mse_bg[name].append(
                            float(patch_mse[batch_idx][~fg[batch_idx]].mean().item())
                        )
        num_samples += batch_size

    rows: List[Dict[str, float]] = []
    for name in names:
        if name == "exact_gt":
            mode = "exact"
            obj_count = num_patches
            bg_count = 0
        else:
            setting = next(item for item in settings if item.name == name)
            mode = setting.mode
            obj_count = setting.object_codes
            bg_count = setting.background_codes
        row = {
            "setting": name,
            "cluster_mode": mode,
            "requested_object_codes": int(obj_count),
            "requested_background_codes": int(bg_count),
            "num_samples": int(num_samples),
            "rFID": float(fid[name].compute()),
            "recon_psnr": _mean(metrics[name]["psnr"]),
            "recon_ssim": _mean(metrics[name]["ssim"]),
            "recon_mse": _mean(metrics[name]["mse"]),
            "recon_mae": _mean(metrics[name]["mae"]),
            "latent_mse_fg": _mean(latent_mse_fg[name]),
            "latent_mse_bg": _mean(latent_mse_bg[name]),
        }
        for key, values in code_stats[name].items():
            row[f"mean_{key}"] = _mean(values)
        if name == "exact_gt":
            row["mean_total_codes"] = float(num_patches)
        rows.append(row)
        log.info("[%s] %s", name, json.dumps(row, sort_keys=True))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--val_jsonl", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--object_codes", default="2,4,8")
    parser.add_argument("--background_codes", default="4,8,16")
    parser.add_argument("--cluster_modes", default="coordinate,kmeans")
    parser.add_argument("--fg_threshold", type=float, default=0.5)
    parser.add_argument("--kmeans_iterations", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--grid_size", type=int, default=32)
    parser.add_argument("--max_caption_tokens", type=int, default=1024)
    parser.add_argument("--n_ovt_per_object", type=int, default=1)
    parser.add_argument("--max_objects", type=int, default=50)
    parser.add_argument(
        "--image_preprocess_mode",
        choices=["default", "coda_center_crop"],
        default="coda_center_crop",
    )
    parser.add_argument("--coda_crop_size", type=int, default=512)
    parser.add_argument("--dtype", choices=["fp32", "bf16"], default="fp32")
    # Kept for compatibility with the shared checkpoint loader. DiT is not run.
    parser.add_argument("--diffusion_inference_steps", type=int, default=10)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model, tokenizer, device, _ = load_model_and_tokenizer(args)
    loader = build_loader(
        args,
        tokenizer,
        model,
        args.val_jsonl,
        shuffle=False,
        max_samples=args.max_samples,
    )
    decoder = load_rae_decoder(model, device=device, dtype=torch.float32)
    log.warning(
        "This probe measures GT-latent quantization through a brittle frozen "
        "decoder; it is not an end-to-end E11 rFID guarantee."
    )

    rows = run(args, model, loader, decoder, device)
    payload = {
        "probe": "pooled_gt_latent_quantization",
        "interpretation": (
            "Inference-only stress test. Low rFID is necessary evidence that the "
            "requested code budget can survive direct decoding, but does not "
            "guarantee Writer/Reader/DiT learnability. High rFID may also reflect "
            "the known off-manifold brittleness of the frozen RAE decoder."
        ),
        "model_path": str(args.model_path),
        "val_jsonl": str(args.val_jsonl),
        "fg_threshold": float(args.fg_threshold),
        "settings": rows,
    }
    with open(output_dir / "summary.json", "w") as handle:
        json.dump(payload, handle, indent=2)

    fields = list(rows[0].keys()) if rows else []
    with open(output_dir / "summary.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    print("\nPOOLED-GT LATENT QUANTIZATION")
    print(f"{'setting':<32} {'codes/img':>10} {'rFID':>9} {'PSNR':>8} {'L-MSE-FG':>11} {'L-MSE-BG':>11}")
    for row in rows:
        print(
            f"{row['setting']:<32} "
            f"{row.get('mean_total_codes', 0.0):>10.1f} "
            f"{row['rFID']:>9.2f} "
            f"{row['recon_psnr']:>8.2f} "
            f"{row['latent_mse_fg']:>11.5f} "
            f"{row['latent_mse_bg']:>11.5f}"
        )
    log.info("Wrote %s and %s", output_dir / "summary.json", output_dir / "summary.csv")


if __name__ == "__main__":
    main()
