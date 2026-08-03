"""E2 root-cause diagnostics D1--D4.

D1  SigLIP object-removal leakage
    Remove one object before SigLIP and measure how much *background* patch
    features change at every vision-transformer layer, grouped by distance.

D2  Register leakage / decoder-prior decomposition
    Compare fixed-noise reconstructions from:
      - unrestricted E2,
      - post-SigLIP oracle register blocking,
      - post-SigLIP register-only,
      - pre-SigLIP pixel-masked register-only,
      - RAE self-only,
      - original OVT + pre-SigLIP-masked register transplant.

D3  Same-category appearance swap
    Swap OVTs between objects with the same category but visibly different
    appearance (brown/white bear, black/brown-white dog).

D4  256-query binding map
    Reshape the per-RAE-query change caused by the D3 swap to 16x16 and compare
    it with the target object's GT mask, both at the final state and by layer.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, "/home/jovyan/PGOT")

from pgot.eval.eval_recon_oracles import load_model_and_tokenizer
from pgot.eval.pgot_inference import (
    generate_siglip_latent,
    ovt_swap_all_layers_inference,
    pgot_forward_eval,
)
from pgot.eval.run_eval import (
    decode_to_image,
    denormalize_images,
    load_rae_decoder,
)
from pgot.model.pgot_utils import gather_ovt_hidden_states
from pgot.train.pgot_dataset import PGOTDataCollator, Pix2CapPGOTDataset


logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s :: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("pgot.e2_d1_d4")


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(value, f, indent=2)


def _font(size: int = 14):
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except Exception:
        return None


def _tensor_to_pil(x: torch.Tensor) -> Image.Image:
    arr = (
        x.detach()
        .float()
        .cpu()
        .clamp(0, 1)
        .permute(1, 2, 0)
        .numpy()
        * 255
    ).round().astype(np.uint8)
    return Image.fromarray(arr)


def _labeled_tile(x: torch.Tensor | Image.Image, label: str, height: int = 32) -> Image.Image:
    image = x if isinstance(x, Image.Image) else _tensor_to_pil(x)
    canvas = Image.new("RGB", (image.width, image.height + height), "white")
    canvas.paste(image, (0, height))
    ImageDraw.Draw(canvas).text((6, 7), label, fill="black", font=_font(13))
    return canvas


def _horizontal_grid(tiles: Sequence[Image.Image]) -> Image.Image:
    out = Image.new(
        "RGB",
        (sum(t.width for t in tiles), max(t.height for t in tiles)),
        "white",
    )
    left = 0
    for tile in tiles:
        out.paste(tile, (left, 0))
        left += tile.width
    return out


def _parse_ints(spec: str) -> List[int]:
    return [int(x.strip()) for x in str(spec).split(",") if x.strip()]


def _resolve_index(dataset: Pix2CapPGOTDataset, image_id: int) -> int:
    for idx, sample in enumerate(dataset.samples):
        if int(sample["image_id"]) == int(image_id):
            return idx
    raise KeyError(f"image_id={image_id} is absent from {dataset.jsonl_path}")


def _build_dataset(args, model, tokenizer) -> Pix2CapPGOTDataset:
    towers = model.get_vision_tower_aux_list()
    image_processor = towers[0].image_processor
    target_processor = towers[1].image_processor if len(towers) > 1 else image_processor
    return Pix2CapPGOTDataset(
        jsonl_path=args.val_jsonl,
        tokenizer=tokenizer,
        image_processor=image_processor,
        target_image_processor=target_processor,
        grid_size=args.grid_size,
        rae_grid_size=16,
        max_caption_tokens=args.max_caption_tokens,
        n_ovt_per_object=args.n_ovt_per_object,
        max_objects=args.max_objects,
        panoptic_categories_json=args.panoptic_categories_json,
        image_preprocess_mode=args.image_preprocess_mode,
        coda_crop_size=args.coda_crop_size,
    )


def _batch(dataset, collator, idx: int) -> Dict:
    return collator([dataset[idx]])


def _forward(model, batch: Dict, **kwargs) -> Dict:
    device = model._pgot_model_device()
    return pgot_forward_eval(
        model,
        images=batch["images"].to(device),
        target_images=batch["target_images"].to(device),
        caption_input_ids=batch["caption_input_ids"].to(device),
        caption_attention_mask=batch["caption_attention_mask"].to(device),
        ovt_positions_in_caption=batch["ovt_positions_in_caption"].to(device),
        ovt_valid_mask=batch["ovt_valid_mask"].to(device),
        **kwargs,
    )


def _object_union(batch: Dict, threshold: float = 0.0) -> torch.Tensor:
    masks = batch["gt_masks_per_ovt"].float()
    valid = batch["ovt_valid_mask"].bool()
    union = (masks * valid.unsqueeze(-1).float()).amax(dim=1)
    return union > float(threshold)


def _object_mask(batch: Dict, obj_idx: int, n_ovt: int) -> torch.Tensor:
    start = int(obj_idx) * int(n_ovt)
    end = start + int(n_ovt)
    return batch["gt_masks_per_ovt"][:, start:end].amax(dim=1)


def _pixel_mask_from_patch(mask: torch.Tensor, size: Tuple[int, int]) -> torch.Tensor:
    side = int(round(math.sqrt(mask.shape[-1])))
    return F.interpolate(
        mask.float().reshape(mask.shape[0], 1, side, side),
        size=size,
        mode="nearest",
    )


def _remove_pixels(images: torch.Tensor, patch_mask: torch.Tensor) -> torch.Tensor:
    """Replace foreground pixels with that image's mean background colour.

    Inputs are already SigLIP-normalized.  Filling in normalized space keeps
    every outside pixel bit-identical and avoids introducing a saturated colour.
    """
    mask = _pixel_mask_from_patch(patch_mask, images.shape[-2:]).to(
        device=images.device, dtype=images.dtype
    )
    bg = 1.0 - mask
    fill = (images * bg).sum(dim=(-2, -1), keepdim=True) / bg.sum(
        dim=(-2, -1), keepdim=True
    ).clamp_min(1.0)
    return images * bg + fill * mask


def _distance_bins(mask_2d: torch.Tensor) -> Dict[str, torch.Tensor]:
    """Euclidean distance from the selected object on the 32x32 patch grid."""
    mask = mask_2d.bool()
    h, w = mask.shape
    yy, xx = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    coords = torch.stack([yy.flatten(), xx.flatten()], dim=1).float()
    fg = coords[mask.flatten()]
    if fg.numel() == 0:
        raise ValueError("empty foreground mask")
    dist = torch.cdist(coords, fg).amin(dim=1).reshape(h, w)
    return {
        "inside": mask,
        "ring_1": (~mask) & (dist <= 1.5),
        "near_2_4": (dist > 1.5) & (dist <= 4.0),
        "mid_5_8": (dist > 4.0) & (dist <= 8.0),
        "far_gt8": dist > 8.0,
    }


def _feature_delta(a: torch.Tensor, b: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    a = a.float()
    b = b.float()
    cos = 1.0 - F.cosine_similarity(a, b, dim=-1)
    rel = (a - b).norm(dim=-1) / a.norm(dim=-1).clamp_min(1e-8)
    return cos, rel


@torch.no_grad()
def run_d1(args, model, dataset, output_dir: Path) -> Dict:
    out_dir = output_dir / "D1_siglip_leakage"
    out_dir.mkdir(parents=True, exist_ok=True)
    tower = model.get_vision_tower_aux_list()[0]
    vision = tower.vision_tower
    param = next(vision.parameters())
    device, dtype = param.device, param.dtype

    selected = []
    for idx in range(len(dataset)):
        item = dataset[idx]
        valid = item["ovt_valid_mask"]
        masks = item["gt_masks_per_ovt"][valid]
        if masks.numel() == 0:
            continue
        areas = masks.mean(dim=-1)
        eligible = (areas >= args.d1_min_area) & (areas <= args.d1_max_area)
        if not bool(eligible.any()):
            continue
        candidates = torch.where(eligible)[0]
        # Prefer a well-sized object instead of a tiny or image-filling one.
        local = candidates[(areas[candidates] - 0.12).abs().argmin()]
        selected.append((idx, int(local.item())))
        if len(selected) >= args.d1_samples:
            break
    if not selected:
        raise RuntimeError("D1 found no eligible object masks.")
    log.info("[D1] selected %d object interventions.", len(selected))

    records: List[Dict] = []
    example_payloads = []
    batch_size = max(int(args.d1_batch_size), 1)
    for start in range(0, len(selected), batch_size):
        chunk = selected[start : start + batch_size]
        items = [dataset[i] for i, _ in chunk]
        images = torch.stack([x["image"] for x in items]).to(device=device, dtype=dtype)
        masks = torch.stack(
            [x["gt_masks_per_ovt"][obj] for x, (_, obj) in zip(items, chunk)]
        ).to(device)
        masked_images = _remove_pixels(images, masks)

        original_out = vision(images, output_hidden_states=True, return_dict=True)
        removed_out = vision(masked_images, output_hidden_states=True, return_dict=True)
        original_states = list(original_out.hidden_states)
        removed_states = list(removed_out.hidden_states)
        if len(original_states) != len(removed_states):
            raise RuntimeError("SigLIP hidden-state trajectories differ in length.")

        for b, ((idx, obj_idx), item) in enumerate(zip(chunk, items)):
            mask_2d = masks[b].reshape(args.grid_size, args.grid_size) > 0.0
            bins = _distance_bins(mask_2d.cpu())
            image_id = int(item["image_id"])
            for layer_idx, (ha, hb) in enumerate(zip(original_states, removed_states)):
                # The PGOT tower applies feature-wise LayerNorm after selection.
                # Apply the same normalization at every layer for comparability.
                xa = F.layer_norm(ha[b].float(), (ha.shape[-1],))
                xb = F.layer_norm(hb[b].float(), (hb.shape[-1],))
                cos, rel = _feature_delta(xa, xb)
                side = int(round(math.sqrt(cos.numel())))
                if side != args.grid_size:
                    raise RuntimeError(
                        f"D1 expected {args.grid_size}x{args.grid_size}, got {side}x{side}"
                    )
                cos = cos.reshape(side, side).cpu()
                rel = rel.reshape(side, side).cpu()
                for bin_name, which in bins.items():
                    if not bool(which.any()):
                        continue
                    records.append(
                        {
                            "image_id": image_id,
                            "dataset_idx": idx,
                            "object_idx": obj_idx,
                            "layer": layer_idx,
                            "distance_bin": bin_name,
                            "patch_count": int(which.sum()),
                            "cosine_distance": float(cos[which].mean()),
                            "relative_l2": float(rel[which].mean()),
                        }
                    )

            if len(example_payloads) < args.d1_visual_examples:
                final_a = F.layer_norm(
                    original_states[-1][b].float(), (original_states[-1].shape[-1],)
                )
                final_b = F.layer_norm(
                    removed_states[-1][b].float(), (removed_states[-1].shape[-1],)
                )
                mid = len(original_states) // 2
                mid_a = F.layer_norm(
                    original_states[mid][b].float(), (original_states[mid].shape[-1],)
                )
                mid_b = F.layer_norm(
                    removed_states[mid][b].float(), (removed_states[mid].shape[-1],)
                )
                _, final_rel = _feature_delta(final_a, final_b)
                _, mid_rel = _feature_delta(mid_a, mid_b)
                _, emb_rel = _feature_delta(
                    original_states[0][b].float(), removed_states[0][b].float()
                )
                example_payloads.append(
                    {
                        "image_id": image_id,
                        "image": images[b].detach().cpu(),
                        "masked": masked_images[b].detach().cpu(),
                        "mask": masks[b].detach().cpu(),
                        "embedding": emb_rel.reshape(args.grid_size, args.grid_size).cpu(),
                        "mid": mid_rel.reshape(args.grid_size, args.grid_size).cpu(),
                        "final": final_rel.reshape(args.grid_size, args.grid_size).cpu(),
                        "mid_layer": mid,
                    }
                )

    aggregate: Dict[int, Dict[str, Dict[str, float]]] = defaultdict(dict)
    for layer in sorted({r["layer"] for r in records}):
        for name in ["inside", "ring_1", "near_2_4", "mid_5_8", "far_gt8"]:
            rows = [
                r
                for r in records
                if r["layer"] == layer and r["distance_bin"] == name
            ]
            if not rows:
                continue
            aggregate[layer][name] = {
                "n_images": len(rows),
                "cosine_distance_mean": float(
                    np.mean([r["cosine_distance"] for r in rows])
                ),
                "cosine_distance_std": float(
                    np.std([r["cosine_distance"] for r in rows])
                ),
                "relative_l2_mean": float(np.mean([r["relative_l2"] for r in rows])),
                "relative_l2_std": float(np.std([r["relative_l2"] for r in rows])),
            }

    with (out_dir / "per_image_layer_distance.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)

    # Layerwise summary plot.
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), constrained_layout=True)
    colours = {
        "inside": "#d62728",
        "ring_1": "#ff7f0e",
        "near_2_4": "#bcbd22",
        "mid_5_8": "#2ca02c",
        "far_gt8": "#1f77b4",
    }
    for metric, ax in zip(["relative_l2_mean", "cosine_distance_mean"], axes):
        for name, colour in colours.items():
            xs, ys = [], []
            for layer, values in aggregate.items():
                if name in values:
                    xs.append(layer)
                    ys.append(values[name][metric])
            ax.plot(xs, ys, marker="o", ms=3, lw=1.6, label=name, color=colour)
        ax.set_xlabel("SigLIP hidden-state index (0 = patch embedding)")
        ax.set_ylabel(metric.replace("_mean", ""))
        ax.grid(alpha=0.25)
    axes[0].legend(fontsize=8)
    fig.suptitle("D1: object-removal influence by distance and SigLIP depth")
    fig.savefig(out_dir / "layerwise_leakage.png", dpi=180)
    plt.close(fig)

    # Qualitative delta maps.
    image_mean = torch.tensor(dataset.image_processor.image_mean)
    image_std = torch.tensor(dataset.image_processor.image_std)
    for payload in example_payloads:
        src = denormalize_images(
            payload["image"].unsqueeze(0), image_mean, image_std
        )[0]
        masked = denormalize_images(
            payload["masked"].unsqueeze(0), image_mean, image_std
        )[0]
        maps = [payload["embedding"], payload["mid"], payload["final"]]
        vmax = max(float(x.max()) for x in maps)
        fig, axes = plt.subplots(1, 5, figsize=(18, 3.8), constrained_layout=True)
        axes[0].imshow(_tensor_to_pil(src))
        axes[0].set_title("original")
        axes[1].imshow(_tensor_to_pil(masked))
        axes[1].set_title("object removed")
        titles = ["patch embedding", f"layer {payload['mid_layer']}", "final layer"]
        for ax, delta, title in zip(axes[2:], maps, titles):
            im = ax.imshow(delta.numpy(), cmap="magma", vmin=0, vmax=max(vmax, 1e-8))
            ax.contour(payload["mask"].reshape(args.grid_size, args.grid_size), [0.01], colors="cyan")
            ax.set_title(title)
        for ax in axes:
            ax.axis("off")
        fig.colorbar(im, ax=axes[2:], fraction=0.015, label="relative feature change")
        fig.suptitle(f"D1 image_id={payload['image_id']}")
        fig.savefig(out_dir / f"image_{payload['image_id']}_delta_maps.png", dpi=170)
        plt.close(fig)

    first_layer = min(aggregate)
    final_layer = max(aggregate)
    summary = {
        "definition": (
            "Foreground pixels are replaced before SigLIP with the image's mean "
            "background colour; every outside pixel remains bit-identical."
        ),
        "num_images": len(selected),
        "num_hidden_states": len(aggregate),
        "first_layer": first_layer,
        "final_layer": final_layer,
        "aggregate_by_layer": {str(k): v for k, v in aggregate.items()},
        "headline": {
            "patch_embedding_far_relative_l2": aggregate[first_layer]
            .get("far_gt8", {})
            .get("relative_l2_mean"),
            "final_far_relative_l2": aggregate[final_layer]
            .get("far_gt8", {})
            .get("relative_l2_mean"),
            "final_inside_relative_l2": aggregate[final_layer]
            .get("inside", {})
            .get("relative_l2_mean"),
            "final_far_cosine_distance": aggregate[final_layer]
            .get("far_gt8", {})
            .get("cosine_distance_mean"),
        },
    }
    _write_json(out_dir / "summary.json", summary)
    log.info("[D1] wrote %s", out_dir)
    return summary


def _run_layer(
    layer,
    hidden: torch.Tensor,
    attn_bias: torch.Tensor,
    position_ids: torch.Tensor,
) -> torch.Tensor:
    try:
        out = layer(
            hidden,
            attention_mask=attn_bias,
            position_ids=position_ids,
            use_cache=False,
        )
    except TypeError:
        out = layer(hidden, attention_mask=attn_bias)
    return out[0] if isinstance(out, tuple) else out


@torch.no_grad()
def _register_transplant_all_layers(model, out_a: Dict, out_donor: Dict) -> torch.Tensor:
    """Use A's OVT/image/self path but donor registers before every LLM layer."""
    hs_a = out_a["hidden_states"]
    hs_d = out_donor["hidden_states"]
    layers = model.model.layers
    if hs_a is None or hs_d is None or len(hs_a) < len(layers) + 1:
        raise ValueError("register transplant requires complete hidden trajectories")
    hidden = hs_a[0].clone()
    pos = out_a["positions"]
    reg = slice(pos["reg_s"], pos["reg_e"])
    position_ids = torch.arange(hidden.shape[1], device=hidden.device).unsqueeze(0)
    for layer_idx, layer in enumerate(layers):
        hidden[:, reg, :] = hs_d[layer_idx][:, reg, :]
        hidden = _run_layer(layer, hidden, out_a["attn_bias"], position_ids)
    if hasattr(model.model, "norm"):
        hidden = model.model.norm(hidden)
    return hidden[:, pos["rae_s"] : pos["rae_e"]]


def _masked_errors(
    pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
) -> Dict[str, float]:
    if target.shape[-2:] != pred.shape[-2:]:
        target = F.interpolate(target, size=pred.shape[-2:], mode="bilinear", align_corners=False)
    mask = F.interpolate(mask.float(), size=pred.shape[-2:], mode="nearest")
    err = (pred.float() - target.float()).pow(2).mean(dim=1, keepdim=True)
    inside = (err * mask).sum() / mask.sum().clamp_min(1.0)
    outside = (err * (1 - mask)).sum() / (1 - mask).sum().clamp_min(1.0)
    return {
        "foreground_mse": float(inside),
        "background_mse": float(outside),
        "foreground_psnr": float(-10.0 * torch.log10(inside.clamp_min(1e-12))),
        "background_psnr": float(-10.0 * torch.log10(outside.clamp_min(1e-12))),
    }


@torch.no_grad()
def _generate_fixed(model, decoder, rae_hidden: torch.Tensor, seed: int, guidance: float):
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    latent = generate_siglip_latent(model, rae_hidden, guidance_level=guidance)
    return decode_to_image(decoder, latent, rae_hidden.device)


@torch.no_grad()
def run_d2(args, model, dataset, collator, decoder, output_dir: Path) -> Dict:
    out_dir = output_dir / "D2_register_leakage"
    out_dir.mkdir(parents=True, exist_ok=True)
    device = model._pgot_model_device()
    image_mean = torch.tensor(dataset.image_processor.image_mean)
    image_std = torch.tensor(dataset.image_processor.image_std)
    target_mean = torch.tensor(dataset.target_image_processor.image_mean)
    target_std = torch.tensor(dataset.target_image_processor.image_std)

    all_rows = []
    for image_id in _parse_ints(args.d2_image_ids):
        idx = _resolve_index(dataset, image_id)
        batch = _batch(dataset, collator, idx)
        union = _object_union(batch).to(device)
        masked_images = _remove_pixels(
            batch["images"].to(device), union.float()
        )
        masked_batch = dict(batch)
        masked_batch["images"] = masked_images.cpu()

        # R0 and R1 use the original post-SigLIP feature stream.
        out_r0 = _forward(model, batch)
        out_r1 = _forward(
            model,
            batch,
            register_image_block_mask=union,
            return_hidden_states=True,
        )
        out_post_reg = _forward(
            model,
            batch,
            register_image_block_mask=union,
            rae_access_mode="register_only",
        )
        # Removing the pixels *before* SigLIP prevents untouched background
        # tokens from contextually encoding the original foreground pixels.
        out_pre_reg = _forward(
            model,
            masked_batch,
            rae_access_mode="register_only",
        )
        out_self = _forward(model, batch, rae_access_mode="self_only")
        out_pre_full = _forward(
            model,
            masked_batch,
            return_hidden_states=True,
        )
        hybrid_rae = _register_transplant_all_layers(model, out_r1, out_pre_full)

        variants = {
            "R0_unrestricted": out_r0["rae_hidden"],
            "R1_post_siglip_all": out_r1["rae_hidden"],
            "R1_post_siglip_register_only": out_post_reg["rae_hidden"],
            "pre_siglip_masked_register_only": out_pre_reg["rae_hidden"],
            "RAE_self_only": out_self["rae_hidden"],
            "hybrid_original_OVT_masked_register": hybrid_rae,
        }
        generated = {
            name: _generate_fixed(
                model, decoder, rae, args.seed, args.guidance_scale
            )
            for name, rae in variants.items()
        }

        source = denormalize_images(
            batch["target_images"].to(device).float(), target_mean, target_std
        )
        masked_source = denormalize_images(
            masked_images.float(), image_mean, image_std
        )
        target_hw = next(iter(generated.values())).shape[-2:]
        source = F.interpolate(source, size=target_hw, mode="bilinear", align_corners=False)
        masked_source = F.interpolate(
            masked_source, size=target_hw, mode="bilinear", align_corners=False
        )
        mask_px = _pixel_mask_from_patch(union.float(), target_hw)

        rows = {}
        for name, pred in generated.items():
            metrics = _masked_errors(pred, source, mask_px)
            rows[name] = metrics
            all_rows.append({"image_id": image_id, "variant": name, **metrics})
        # Direct intervention effect: how much changing the register source
        # changes the reconstruction, split by foreground/background.
        reference = generated["R1_post_siglip_register_only"]
        for name in [
            "pre_siglip_masked_register_only",
            "RAE_self_only",
            "hybrid_original_OVT_masked_register",
        ]:
            delta = (generated[name] - reference).abs().mean(dim=1, keepdim=True)
            rows[name]["mean_abs_delta_vs_post_register_fg"] = float(
                (delta * mask_px).sum() / mask_px.sum().clamp_min(1.0)
            )
            rows[name]["mean_abs_delta_vs_post_register_bg"] = float(
                (delta * (1 - mask_px)).sum() / (1 - mask_px).sum().clamp_min(1.0)
            )

        tiles = [
            _labeled_tile(source[0], "source"),
            _labeled_tile(masked_source[0], "foreground removed before SigLIP"),
        ] + [
            _labeled_tile(generated[name][0], name)
            for name in variants
        ]
        grid = _horizontal_grid(tiles)
        grid.save(out_dir / f"image_{image_id}_fixed_noise_grid.png")
        _write_json(
            out_dir / f"image_{image_id}.json",
            {
                "image_id": image_id,
                "dataset_idx": idx,
                "n_objects": int(batch["n_objects_list"][0]),
                "foreground_patch_fraction": float(union.float().mean()),
                "caption": batch["caption_texts"][0],
                "seed": args.seed,
                "metrics": rows,
            },
        )
        log.info("[D2] image_id=%d complete.", image_id)

    with (out_dir / "metrics.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_rows)
    aggregate = {}
    for name in sorted({r["variant"] for r in all_rows}):
        rows = [r for r in all_rows if r["variant"] == name]
        aggregate[name] = {
            k: float(np.mean([r[k] for r in rows]))
            for k in [
                "foreground_mse",
                "background_mse",
                "foreground_psnr",
                "background_psnr",
            ]
        }
    summary = {
        "num_images": len(_parse_ints(args.d2_image_ids)),
        "image_ids": _parse_ints(args.d2_image_ids),
        "fixed_noise_seed": args.seed,
        "aggregate": aggregate,
        "interpretation_key": {
            "post_vs_pre_siglip": (
                "A large change means post-SigLIP background tokens carried "
                "foreground-dependent contextual information."
            ),
            "pre_siglip_vs_self": (
                "Similarity means the remaining reconstruction is dominated "
                "by RAE self state / DiT scene prior rather than register evidence."
            ),
            "hybrid": (
                "Original OVTs are retained while every-layer register states "
                "come from the pre-SigLIP foreground-removed image."
            ),
        },
    }
    _write_json(out_dir / "summary.json", summary)
    return summary


def _binary_auc(scores: torch.Tensor, target: torch.Tensor) -> float:
    scores = scores.flatten().float()
    target = target.flatten().bool()
    pos, neg = scores[target], scores[~target]
    if pos.numel() == 0 or neg.numel() == 0:
        return float("nan")
    comparisons = (pos[:, None] > neg[None, :]).float()
    ties = (pos[:, None] == neg[None, :]).float()
    return float((comparisons + 0.5 * ties).mean())


def _binding_metrics(score: torch.Tensor, mask: torch.Tensor) -> Dict[str, float]:
    score = score.float().reshape(-1)
    soft_mask = mask.float().reshape(-1).clamp(0, 1)
    target = soft_mask > 0.0
    inside = score[target]
    outside = score[~target]
    k = max(int(target.sum()), 1)
    top = torch.topk(score, k=min(k, score.numel())).indices
    top_precision = float(target[top].float().mean())
    side = int(round(math.sqrt(score.numel())))
    yy, xx = torch.meshgrid(
        torch.arange(side), torch.arange(side), indexing="ij"
    )
    coords = torch.stack([yy.flatten(), xx.flatten()], dim=-1).float()
    score_w = score.clamp_min(0)
    score_com = (coords * score_w[:, None]).sum(0) / score_w.sum().clamp_min(1e-8)
    mask_com = (coords * soft_mask[:, None]).sum(0) / soft_mask.sum().clamp_min(1e-8)
    max_dist = math.sqrt(2.0) * max(side - 1, 1)
    return {
        "inside_mean": float(inside.mean()) if inside.numel() else float("nan"),
        "outside_mean": float(outside.mean()) if outside.numel() else float("nan"),
        "inside_outside_ratio": float(
            inside.mean() / outside.mean().clamp_min(1e-8)
        )
        if inside.numel() and outside.numel()
        else float("nan"),
        "auroc": _binary_auc(score, target),
        "top_gt_area_precision": top_precision,
        "center_of_mass_distance_normalized": float(
            (score_com - mask_com).norm() / max_dist
        ),
        "foreground_query_fraction": float(target.float().mean()),
    }


@torch.no_grad()
def _ovt_swap_trace(
    model,
    out_a: Dict,
    out_b: Dict,
    obj_a: int,
    obj_b: int,
    n_ovt: int,
) -> Tuple[torch.Tensor, List[torch.Tensor]]:
    """All-layer OVT swap plus one relative-delta map per LLM layer."""
    hs_a = out_a["hidden_states"]
    hs_b = out_b["hidden_states"]
    layers = model.model.layers
    hidden = hs_a[0].clone()
    abs_a, abs_b = out_a["ovt_abs_positions"], out_b["ovt_abs_positions"]
    pos = out_a["positions"]
    position_ids = torch.arange(hidden.shape[1], device=hidden.device).unsqueeze(0)
    maps = []
    for layer_idx, layer in enumerate(layers):
        for offset in range(int(n_ovt)):
            pa = int(abs_a[0, obj_a * n_ovt + offset])
            pb = int(abs_b[0, obj_b * n_ovt + offset])
            hidden[:, pa, :] = hs_b[layer_idx][:, pb, :]
        hidden = _run_layer(layer, hidden, out_a["attn_bias"], position_ids)
        if (
            bool(getattr(model.config, "pgot_fvw_enable", False))
            and getattr(model, "pgot_fvw_block", None) is not None
            and int(layer_idx) in set(int(x) for x in model.pgot_fvw_layers)
        ):
            current_ovts = gather_ovt_hidden_states(
                hidden,
                abs_a,
                out_a["ovt_valid_mask"],
            )
            image_states = hidden[:, pos["img_s"] : pos["img_e"], :]
            visual_ovts, _ = model.pgot_fvw_block(
                ovt_states=current_ovts,
                image_states=image_states,
                ovt_valid_mask=out_a["ovt_valid_mask"],
            )
            hidden = model._pgot_fvw_scatter_overwrite(
                hidden_states=hidden,
                ovt_abs_positions=abs_a,
                ovt_valid_mask=out_a["ovt_valid_mask"],
                visual_ovts=visual_ovts,
            )
        compare = hidden
        if layer_idx == len(layers) - 1 and hasattr(model.model, "norm"):
            compare = model.model.norm(compare)
        base = hs_a[layer_idx + 1]
        rae_m = compare[:, pos["rae_s"] : pos["rae_e"]].float()
        rae_a = base[:, pos["rae_s"] : pos["rae_e"]].float()
        delta = (rae_m - rae_a).norm(dim=-1) / rae_a.norm(dim=-1).clamp_min(1e-8)
        maps.append(delta[0].detach().cpu())
    if hasattr(model.model, "norm"):
        hidden = model.model.norm(hidden)
    return hidden[:, pos["rae_s"] : pos["rae_e"]], maps


def _sample_description(dataset, idx: int, obj: int) -> Dict[str, str]:
    segment = dataset.samples[idx]["segments"][obj]
    return {
        "category": str(segment.get("category", "")),
        "description": str(segment.get("description", "")),
    }


@torch.no_grad()
def run_d3_d4(args, model, dataset, collator, decoder, output_dir: Path) -> Tuple[Dict, Dict]:
    d3_dir = output_dir / "D3_same_category_swap"
    d4_dir = output_dir / "D4_rae_binding"
    d3_dir.mkdir(parents=True, exist_ok=True)
    d4_dir.mkdir(parents=True, exist_ok=True)
    device = model._pgot_model_device()
    target_mean = torch.tensor(dataset.target_image_processor.image_mean)
    target_std = torch.tensor(dataset.target_image_processor.image_std)

    pair_specs = []
    for chunk in args.d3_pairs.split(","):
        values = _parse_ints(chunk.replace(":", ","))
        if len(values) != 4:
            raise ValueError(
                f"Bad D3 pair '{chunk}'; expected imageA:imageB:objA:objB"
            )
        pair_specs.append(tuple(values))

    d3_rows, d4_rows = [], []
    for pair_idx, (id_a, id_b, obj_a, obj_b) in enumerate(pair_specs):
        idx_a, idx_b = _resolve_index(dataset, id_a), _resolve_index(dataset, id_b)
        batch_a = _batch(dataset, collator, idx_a)
        batch_b = _batch(dataset, collator, idx_b)
        desc_a = _sample_description(dataset, idx_a, obj_a)
        desc_b = _sample_description(dataset, idx_b, obj_b)
        if desc_a["category"].lower() != desc_b["category"].lower():
            raise ValueError(
                f"D3 is same-category only, got {desc_a['category']} vs {desc_b['category']}"
            )

        out_a = _forward(model, batch_a, return_hidden_states=True)
        out_b = _forward(model, batch_b, return_hidden_states=True)
        own_ovt_indices = tuple(
            range(
                obj_a * args.n_ovt_per_object,
                (obj_a + 1) * args.n_ovt_per_object,
            )
        )
        out_a_without_own_ovt = _forward(
            model,
            batch_a,
            rae_block_ovt_indices=own_ovt_indices,
        )
        swap_rae, layer_maps = _ovt_swap_trace(
            model,
            out_a,
            out_b,
            obj_a,
            obj_b,
            args.n_ovt_per_object,
        )
        # Cross-check the diagnostic implementation against the shared helper.
        shared_rae, _ = ovt_swap_all_layers_inference(
            model,
            out_A=out_a,
            out_B=out_b,
            swap_pairs=[(obj_a, obj_b)],
            n_ovt_per_object=args.n_ovt_per_object,
        )
        max_impl_diff = float((swap_rae - shared_rae).abs().max())
        if max_impl_diff > 1e-4:
            raise RuntimeError(f"swap trace/helper mismatch: {max_impl_diff}")

        self_img = _generate_fixed(
            model, decoder, out_a["rae_hidden"], args.seed, args.guidance_scale
        )
        swap_img = _generate_fixed(
            model, decoder, swap_rae, args.seed, args.guidance_scale
        )
        source_a = denormalize_images(
            batch_a["target_images"].to(device).float(), target_mean, target_std
        )
        source_b = denormalize_images(
            batch_b["target_images"].to(device).float(), target_mean, target_std
        )
        target_hw = self_img.shape[-2:]
        source_a = F.interpolate(source_a, size=target_hw, mode="bilinear", align_corners=False)
        source_b = F.interpolate(source_b, size=target_hw, mode="bilinear", align_corners=False)

        mask_a = _object_mask(batch_a, obj_a, args.n_ovt_per_object)
        mask_px = _pixel_mask_from_patch(mask_a, target_hw).to(device)
        delta_img = (swap_img - self_img).abs().mean(dim=1, keepdim=True)
        inside_delta = float(
            (delta_img * mask_px).sum() / mask_px.sum().clamp_min(1.0)
        )
        outside_delta = float(
            (delta_img * (1 - mask_px)).sum() / (1 - mask_px).sum().clamp_min(1.0)
        )
        d3_record = {
            "pair": pair_idx,
            "image_id_A": id_a,
            "image_id_B": id_b,
            "object_A": desc_a,
            "object_B": desc_b,
            "same_category": True,
            "swap_mode": "all_layers",
            "fixed_noise_seed": args.seed,
            "mean_abs_recon_delta_inside_A": inside_delta,
            "mean_abs_recon_delta_outside_A": outside_delta,
            "recon_delta_inside_outside_ratio": inside_delta / max(outside_delta, 1e-8),
            "implementation_max_abs_difference": max_impl_diff,
        }
        d3_rows.append(d3_record)
        grid = _horizontal_grid(
            [
                _labeled_tile(source_a[0], f"A: {desc_a['category']}"),
                _labeled_tile(self_img[0], "A self reconstruction"),
                _labeled_tile(
                    swap_img[0],
                    f"A {desc_a['category']} OVT <- B {desc_b['category']} OVT",
                ),
                _labeled_tile(source_b[0], f"B donor: {desc_b['category']}"),
            ]
        )
        grid.save(d3_dir / f"pair_{pair_idx:02d}_A{id_a}_B{id_b}.png")
        _write_json(d3_dir / f"pair_{pair_idx:02d}.json", d3_record)

        # D4: target object's 16x16 RAE ownership and per-query intervention.
        gt_rae = batch_a["gt_rae_masks_per_ovt"][
            0, obj_a * args.n_ovt_per_object
        ].float()
        final_map = layer_maps[-1]
        final_metrics = _binding_metrics(final_map, gt_rae)
        own_block_map = (
            out_a_without_own_ovt["rae_hidden"].float() - out_a["rae_hidden"].float()
        ).norm(dim=-1) / out_a["rae_hidden"].float().norm(dim=-1).clamp_min(1e-8)
        own_block_map = own_block_map[0].detach().cpu()
        own_block_metrics = _binding_metrics(own_block_map, gt_rae)
        layer_metrics = []
        for layer_idx, delta_map in enumerate(layer_maps):
            metrics = _binding_metrics(delta_map, gt_rae)
            row = {
                "pair": pair_idx,
                "image_id_A": id_a,
                "image_id_B": id_b,
                "layer": layer_idx,
                **metrics,
            }
            layer_metrics.append(row)
            d4_rows.append(row)
        np.save(
            d4_dir / f"pair_{pair_idx:02d}_rae_delta_16x16.npy",
            final_map.reshape(16, 16).numpy(),
        )
        np.save(
            d4_dir / f"pair_{pair_idx:02d}_own_ovt_block_delta_16x16.npy",
            own_block_map.reshape(16, 16).numpy(),
        )

        fig, axes = plt.subplots(1, 5, figsize=(18, 3.8), constrained_layout=True)
        axes[0].imshow(_tensor_to_pil(source_a[0]))
        axes[0].set_title("source A")
        axes[1].imshow(gt_rae.reshape(16, 16), cmap="gray", vmin=0, vmax=1)
        axes[1].set_title("GT object on RAE grid")
        im = axes[2].imshow(final_map.reshape(16, 16), cmap="magma")
        axes[2].contour(gt_rae.reshape(16, 16), [0.01], colors="cyan")
        axes[2].set_title("donor OVT swap: RAE-query change")
        im_block = axes[3].imshow(own_block_map.reshape(16, 16), cmap="magma")
        axes[3].contour(gt_rae.reshape(16, 16), [0.01], colors="cyan")
        axes[3].set_title("block A's own OVT: RAE-query change")
        xs = [r["layer"] for r in layer_metrics]
        axes[4].plot(xs, [r["auroc"] for r in layer_metrics], label="swap AUROC")
        axes[4].plot(
            xs,
            [min(r["inside_outside_ratio"], 10.0) for r in layer_metrics],
            label="swap inside/outside (clipped 10)",
        )
        axes[4].axhline(0.5, ls="--", lw=1, color="gray")
        axes[4].set_xlabel("LLM layer")
        axes[4].grid(alpha=0.25)
        axes[4].legend(fontsize=8)
        for ax in axes[:4]:
            ax.axis("off")
        fig.colorbar(im, ax=axes[2], fraction=0.046)
        fig.colorbar(im_block, ax=axes[3], fraction=0.046)
        fig.suptitle(
            f"D4 A={id_a} <- B={id_b} | final AUROC={final_metrics['auroc']:.3f}, "
            f"own-block AUROC={own_block_metrics['auroc']:.3f}"
        )
        fig.savefig(d4_dir / f"pair_{pair_idx:02d}_binding.png", dpi=180)
        plt.close(fig)
        _write_json(
            d4_dir / f"pair_{pair_idx:02d}.json",
            {
                "pair": pair_idx,
                "image_id_A": id_a,
                "image_id_B": id_b,
                "object_A": desc_a,
                "object_B": desc_b,
                "final": final_metrics,
                "own_ovt_block_final": own_block_metrics,
                "by_layer": layer_metrics,
            },
        )
        log.info("[D3/D4] pair %d A=%d B=%d complete.", pair_idx, id_a, id_b)

    d3_summary = {
        "num_pairs": len(d3_rows),
        "pairs": d3_rows,
        "aggregate": {
            "mean_recon_delta_inside": float(
                np.mean([r["mean_abs_recon_delta_inside_A"] for r in d3_rows])
            ),
            "mean_recon_delta_outside": float(
                np.mean([r["mean_abs_recon_delta_outside_A"] for r in d3_rows])
            ),
            "mean_locality_ratio": float(
                np.mean([r["recon_delta_inside_outside_ratio"] for r in d3_rows])
            ),
        },
    }
    final_d4 = [r for r in d4_rows if r["layer"] == max(x["layer"] for x in d4_rows)]
    d4_summary = {
        "num_pairs": len(pair_specs),
        "definition": (
            "For each of the 256 final RAE queries: "
            "||q_swap-q_self||_2 / ||q_self||_2, reshaped row-major to 16x16."
        ),
        "final_aggregate": {
            k: float(np.mean([r[k] for r in final_d4]))
            for k in [
                "inside_mean",
                "outside_mean",
                "inside_outside_ratio",
                "auroc",
                "top_gt_area_precision",
                "center_of_mass_distance_normalized",
            ]
        },
        "own_ovt_block_final_aggregate": {
            k: float(
                np.mean(
                    [
                        json.load(
                            (d4_dir / f"pair_{idx:02d}.json").open()
                        )["own_ovt_block_final"][k]
                        for idx in range(len(pair_specs))
                    ]
                )
            )
            for k in [
                "inside_mean",
                "outside_mean",
                "inside_outside_ratio",
                "auroc",
                "top_gt_area_precision",
                "center_of_mass_distance_normalized",
            ]
        },
    }
    _write_json(d3_dir / "summary.json", d3_summary)
    _write_json(d4_dir / "summary.json", d4_summary)
    with (d4_dir / "layerwise_metrics.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(d4_rows[0].keys()))
        writer.writeheader()
        writer.writerows(d4_rows)
    return d3_summary, d4_summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path",
        default="/home/jovyan/PGOT/checkpoints/pgot_lviscap_hardreg_e2/checkpoint-10000",
    )
    parser.add_argument(
        "--val_jsonl",
        default="/home/jovyan/PGOT/data/pgot_pix2cap_generated_val5k.jsonl",
    )
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--dtype", choices=["fp32", "bf16"], default="fp32")
    parser.add_argument("--grid_size", type=int, default=32)
    parser.add_argument("--max_caption_tokens", type=int, default=1024)
    parser.add_argument("--n_ovt_per_object", type=int, default=1)
    parser.add_argument("--max_objects", type=int, default=50)
    parser.add_argument(
        "--panoptic_categories_json",
        default="/home/jovyan/data/coco/annotations/panoptic_val2017.json",
    )
    parser.add_argument(
        "--image_preprocess_mode",
        choices=["default", "coda_center_crop"],
        default="coda_center_crop",
    )
    parser.add_argument("--coda_crop_size", type=int, default=512)
    parser.add_argument("--diffusion_inference_steps", type=int, default=25)
    parser.add_argument("--guidance_scale", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument(
        "--only",
        choices=["all", "d1", "d2", "d3d4"],
        default="all",
        help="Run a subset while still using the same checkpoint/data loader.",
    )
    parser.add_argument("--d1_samples", type=int, default=64)
    parser.add_argument("--d1_batch_size", type=int, default=4)
    parser.add_argument("--d1_min_area", type=float, default=0.01)
    parser.add_argument("--d1_max_area", type=float, default=0.35)
    parser.add_argument("--d1_visual_examples", type=int, default=4)
    parser.add_argument(
        "--d2_image_ids",
        default="285,92839,22192",
        help="Simple foreground examples used for fixed-noise leakage decomposition.",
    )
    parser.add_argument(
        "--d3_pairs",
        default="285:92839:0:0,22192:131273:0:0",
        help="imageA:imageB:objA:objB pairs; A and B must share a category.",
    )
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model, tokenizer, device, dtype = load_model_and_tokenizer(args)
    dataset = _build_dataset(args, model, tokenizer)
    collator = PGOTDataCollator(pad_token_id=tokenizer.pad_token_id)
    log.info("Dataset ready: %d samples.", len(dataset))

    d1 = None
    d2 = None
    d3 = None
    d4 = None
    if args.only in {"all", "d1"}:
        d1 = run_d1(args, model, dataset, output_dir)
    decoder = None
    if args.only in {"all", "d2", "d3d4"}:
        decoder = load_rae_decoder(model, device=device, dtype=dtype)
    if args.only in {"all", "d2"}:
        d2 = run_d2(args, model, dataset, collator, decoder, output_dir)
    if args.only in {"all", "d3d4"}:
        d3, d4 = run_d3_d4(
            args, model, dataset, collator, decoder, output_dir
        )
    summary = {
        "model_path": args.model_path,
        "val_jsonl": args.val_jsonl,
        "only": args.only,
    }
    if d1 is not None:
        summary["D1"] = d1["headline"]
    if d2 is not None:
        summary["D2"] = d2["aggregate"]
    if d3 is not None:
        summary["D3"] = d3["aggregate"]
    if d4 is not None:
        summary["D4"] = {
            "swap": d4["final_aggregate"],
            "own_ovt_block": d4["own_ovt_block_final_aggregate"],
        }
    _write_json(output_dir / "summary.json", summary)
    log.info("D1--D4 complete: %s", output_dir / "summary.json")


if __name__ == "__main__":
    main()
