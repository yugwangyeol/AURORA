"""Matched object-removal leakage probe for SigLIP2 and DINOv2.

The image and COCO instance mask first receive the same CODA 512 crop. One
well-sized instance is replaced with the mean colour of its background. The
probe measures how much patch features change as a function of transformer
depth and distance from the intervention. Both encoders are evaluated on an
exact 32x32 patch grid (SigLIP: 512/16, DINOv2: 448/14).
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoImageProcessor, AutoModel

from pgot.eval.run_eval import CocoInstanceMaskCache
from pgot.train.pgot_dataset import _coda_center_crop_image


def distance_bins(mask: torch.Tensor) -> dict[str, torch.Tensor]:
    mask = mask.bool()
    h, w = mask.shape
    yy, xx = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    coords = torch.stack([yy.flatten(), xx.flatten()], dim=1).float()
    fg = coords[mask.flatten()]
    dist = torch.cdist(coords, fg).amin(dim=1).reshape(h, w)
    return {
        "inside": mask,
        "ring_1": (~mask) & (dist <= 1.5),
        "near_2_4": (dist > 1.5) & (dist <= 4.0),
        "mid_5_8": (dist > 4.0) & (dist <= 8.0),
        "far_gt8": dist > 8.0,
    }


def feature_delta(a: torch.Tensor, b: torch.Tensor):
    a = F.layer_norm(a.float(), (a.shape[-1],))
    b = F.layer_norm(b.float(), (b.shape[-1],))
    cosine = 1.0 - F.cosine_similarity(a, b, dim=-1)
    relative_l2 = (a - b).norm(dim=-1) / a.norm(dim=-1).clamp_min(1e-8)
    return cosine, relative_l2


def select_interventions(samples, cache, count, min_area, max_area):
    selected = []
    for sample in samples:
        gt = cache.get(int(sample["image_id"]))
        if gt is None:
            continue
        labels = [int(x) for x in torch.unique(gt) if int(x) > 0]
        candidates = []
        for label in labels:
            mask = gt == label
            area = float(mask.float().mean())
            if min_area <= area <= max_area:
                candidates.append((abs(area - 0.12), label, mask, area))
        if not candidates:
            continue
        _, label, mask, area = min(candidates, key=lambda x: x[0])
        selected.append((sample, label, mask, area))
        if len(selected) >= count:
            break
    return selected


def build_images(selected):
    original, removed, masks32, meta = [], [], [], []
    for sample, label, mask512, area in selected:
        image = Image.open(sample["image_path"]).convert("RGB")
        image = _coda_center_crop_image(image, 512)
        array = np.asarray(image, dtype=np.float32) / 255.0
        mask = mask512.numpy().astype(bool)
        background = array[~mask]
        fill = background.mean(axis=0) if background.size else array.mean(axis=(0, 1))
        changed = array.copy()
        changed[mask] = fill
        original.append(Image.fromarray((array * 255).round().astype(np.uint8)))
        removed.append(Image.fromarray((changed * 255).round().astype(np.uint8)))
        mask32 = F.interpolate(
            mask512.float()[None, None], size=(32, 32), mode="nearest"
        )[0, 0] > 0
        masks32.append(mask32)
        meta.append(
            {
                "image_id": int(sample["image_id"]),
                "instance_label": int(label),
                "area_fraction": float(area),
            }
        )
    return original, removed, masks32, meta


def dino_inputs(images, device):
    values = []
    mean = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
    std = torch.tensor([0.229, 0.224, 0.225])[:, None, None]
    for image in images:
        resized = image.resize((448, 448), Image.Resampling.BILINEAR)
        array = np.asarray(resized, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(array).permute(2, 0, 1)
        values.append((tensor - mean) / std)
    return torch.stack(values).to(device)


def dino_states(model, pixels):
    x = model.prepare_tokens_with_masks(pixels)
    states = [model.norm(x)[:, 1:, :]]
    for block in model.blocks:
        x = block(x)
        states.append(model.norm(x)[:, 1:, :])
    for state in states:
        if state.shape[1] != 1024:
            raise RuntimeError(f"DINO expected 1024 patch tokens, got {state.shape}")
    return states


def siglip_states(model, processor, images, device):
    pixels = processor(images=images, return_tensors="pt")["pixel_values"].to(device)
    vision = getattr(model, "vision_model", model)
    output = vision(pixel_values=pixels, output_hidden_states=True, return_dict=True)
    states = list(output.hidden_states)
    for state in states:
        if state.shape[1] != 1024:
            raise RuntimeError(f"SigLIP expected 1024 patch tokens, got {state.shape}")
    return states


def collect_records(name, original_states, removed_states, masks32, meta, records):
    for b, (mask, info) in enumerate(zip(masks32, meta)):
        bins = distance_bins(mask)
        for layer, (a, b_state) in enumerate(zip(original_states, removed_states)):
            cosine, relative_l2 = feature_delta(a[b], b_state[b])
            cosine = cosine.reshape(32, 32).cpu()
            relative_l2 = relative_l2.reshape(32, 32).cpu()
            for bin_name, which in bins.items():
                if not bool(which.any()):
                    continue
                records.append(
                    {
                        "encoder": name,
                        "image_id": info["image_id"],
                        "layer": layer,
                        "distance_bin": bin_name,
                        "cosine_distance": float(cosine[which].mean()),
                        "relative_l2": float(relative_l2[which].mean()),
                    }
                )


def aggregate_records(records):
    aggregate = defaultdict(lambda: defaultdict(dict))
    encoders = sorted({row["encoder"] for row in records})
    for encoder in encoders:
        layers = sorted({row["layer"] for row in records if row["encoder"] == encoder})
        for layer in layers:
            for bin_name in ["inside", "ring_1", "near_2_4", "mid_5_8", "far_gt8"]:
                rows = [
                    row
                    for row in records
                    if row["encoder"] == encoder
                    and row["layer"] == layer
                    and row["distance_bin"] == bin_name
                ]
                if rows:
                    aggregate[encoder][layer][bin_name] = {
                        "relative_l2_mean": float(np.mean([r["relative_l2"] for r in rows])),
                        "relative_l2_std": float(np.std([r["relative_l2"] for r in rows])),
                        "cosine_distance_mean": float(
                            np.mean([r["cosine_distance"] for r in rows])
                        ),
                    }
    return aggregate


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--val_jsonl",
        default="/home/jovyan/PGOT/data/pgot_pix2cap_generated_val5k.jsonl",
    )
    parser.add_argument(
        "--coco_mask_cache",
        default="/home/jovyan/PGOT/data/coco_inst_mask_cache_coda512",
    )
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--num_samples", type=int, default=32)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--min_area", type=float, default=0.03)
    parser.add_argument("--max_area", type=float, default=0.30)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    samples = [json.loads(line) for line in Path(args.val_jsonl).open()]
    cache = CocoInstanceMaskCache(args.coco_mask_cache)
    selected = select_interventions(
        samples, cache, args.num_samples, args.min_area, args.max_area
    )
    if len(selected) < args.num_samples:
        raise RuntimeError(f"Only found {len(selected)} eligible interventions")
    original, removed, masks32, meta = build_images(selected)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    records = []

    siglip_id = "google/siglip2-so400m-patch16-512"
    siglip_processor = AutoImageProcessor.from_pretrained(siglip_id, local_files_only=True)
    siglip = AutoModel.from_pretrained(
        siglip_id, torch_dtype=torch.float32, local_files_only=True
    ).to(device).eval()
    for start in range(0, len(original), args.batch_size):
        end = min(start + args.batch_size, len(original))
        original_states = siglip_states(
            siglip, siglip_processor, original[start:end], device
        )
        removed_states = siglip_states(
            siglip, siglip_processor, removed[start:end], device
        )
        collect_records(
            "siglip2_so400m", original_states, removed_states,
            masks32[start:end], meta[start:end], records,
        )
    del siglip
    torch.cuda.empty_cache()

    dino = torch.hub.load(
        "/home/jovyan/.cache/torch/hub/facebookresearch_dinov2_main",
        "dinov2_vitb14",
        source="local",
        pretrained=True,
    ).to(device).eval()
    for start in range(0, len(original), args.batch_size):
        end = min(start + args.batch_size, len(original))
        original_states = dino_states(dino, dino_inputs(original[start:end], device))
        removed_states = dino_states(dino, dino_inputs(removed[start:end], device))
        collect_records(
            "dinov2_vitb14", original_states, removed_states,
            masks32[start:end], meta[start:end], records,
        )

    aggregate = aggregate_records(records)
    serializable = {
        encoder: {str(layer): values for layer, values in layers.items()}
        for encoder, layers in aggregate.items()
    }
    headline = {}
    for encoder, layers in aggregate.items():
        first, last = min(layers), max(layers)
        headline[encoder] = {
            "num_layers_including_embedding": len(layers),
            "embedding_far_relative_l2": layers[first]["far_gt8"]["relative_l2_mean"],
            "final_far_relative_l2": layers[last]["far_gt8"]["relative_l2_mean"],
            "final_far_cosine_distance": layers[last]["far_gt8"]["cosine_distance_mean"],
            "final_inside_relative_l2": layers[last]["inside"]["relative_l2_mean"],
            "final_far_to_inside_ratio": (
                layers[last]["far_gt8"]["relative_l2_mean"]
                / max(layers[last]["inside"]["relative_l2_mean"], 1e-8)
            ),
        }

    summary = {
        "definition": (
            "Matched CODA-cropped object removal; exact 32x32 patch grids. "
            "A lower far/inside ratio means less global contamination relative "
            "to the encoder's response at the edited object."
        ),
        "num_samples": len(selected),
        "samples": meta,
        "headline": headline,
        "aggregate_by_encoder_layer": serializable,
        "feature_oracle_reference": {
            "siglip2_raw_fARI": 0.3281,
            "siglip2_raw_mBO": 0.3383,
            "dinov2_raw_fARI": 0.5796,
            "dinov2_raw_mBO": 0.5173,
            "source": "/home/jovyan/PGOT/outputs/probe_encoder_compare/encoder_compare.json",
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    colours = {"siglip2_so400m": "#d62728", "dinov2_vitb14": "#1f77b4"}
    for encoder, layers in aggregate.items():
        xs = sorted(layers)
        for ax, bin_name, title in [
            (axes[0], "inside", "Inside-object response"),
            (axes[1], "far_gt8", "Far-background leakage"),
        ]:
            ax.plot(
                [x / max(xs) for x in xs],
                [layers[x][bin_name]["relative_l2_mean"] for x in xs],
                marker="o", ms=3, label=encoder, color=colours[encoder],
            )
            ax.set_title(title)
            ax.set_xlabel("normalized encoder depth")
            ax.set_ylabel("relative L2 change")
            ax.grid(alpha=0.25)
    for ax in axes:
        ax.legend()
    fig.savefig(output_dir / "layerwise_leakage_compare.png", dpi=180)
    plt.close(fig)
    print(json.dumps(headline, indent=2))


if __name__ == "__main__":
    main()
