"""Same-category appearance transfer for the one-shot owner-masked Reader.

The one-shot architecture has no detachable ``m_k``.  A faithful appearance
intervention must therefore move the donor semantic owner together with the
donor raw SigLIP features belonging to that owner.  Donor features are warped
from their predicted owner bounding box into the target owner's bounding box;
the target ownership map remains fixed so this tests appearance transfer at a
fixed target location.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from tqdm import tqdm

from pgot.eval.diagnose_e11_followup import (
    _grid,
    _label,
    _object_categories,
    _object_masks,
    _pool_features,
    _source_image,
)
from pgot.eval.diagnose_one_shot_reader import _condition
from pgot.eval.eval_recon_oracles import build_loader, load_model_and_tokenizer
from pgot.eval.pgot_inference import generate_siglip_latent, pgot_forward_eval
from pgot.eval.run_eval import decode_to_image, load_rae_decoder


log = logging.getLogger("pgot.diagnose_one_shot_swap")


def _to_pil(image: torch.Tensor) -> Image.Image:
    array = (
        image.detach().float().cpu().clamp(0, 1).permute(1, 2, 0).numpy()
        * 255.0
    ).round().astype(np.uint8)
    return Image.fromarray(array, mode="RGB")


def _outline_owner(image: Image.Image, mask: torch.Tensor) -> Image.Image:
    """Draw the selected 32x32 owner support as a compact red bounding box."""
    image = image.convert("RGB").copy()
    side = int(round(math.sqrt(mask.numel())))
    indices = torch.nonzero(mask.detach().cpu().reshape(side, side), as_tuple=False)
    if indices.numel() == 0:
        return image
    y0, x0 = indices.min(dim=0).values.tolist()
    y1, x1 = indices.max(dim=0).values.tolist()
    left = int(x0 * image.width / side)
    top = int(y0 * image.height / side)
    right = max(left + 1, int((x1 + 1) * image.width / side) - 1)
    bottom = max(top + 1, int((y1 + 1) * image.height / side) - 1)
    ImageDraw.Draw(image).rectangle(
        (left, top, right, bottom), outline=(230, 20, 20), width=4
    )
    return image


def _difference_image(swapped: torch.Tensor, baseline: torch.Tensor) -> Image.Image:
    difference = (swapped.float() - baseline.float()).abs().mean(dim=0)
    difference = difference / difference.quantile(0.99).clamp_min(1e-6)
    heat = difference.clamp(0, 1)
    rgb = torch.stack((heat, 0.25 * heat, 1.0 - heat), dim=0)
    return _to_pil(rgb)


def _summary(values: list[float]) -> dict:
    if not values:
        return {"mean": None, "median": None, "std": None, "count": 0}
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "std": float(array.std()),
        "count": int(array.size),
    }


def _extract_owner_crop(raw: torch.Tensor, mask: torch.Tensor) -> torch.Tensor | None:
    side = int(round(math.sqrt(raw.shape[0])))
    grid_mask = mask.reshape(side, side)
    indices = torch.nonzero(grid_mask, as_tuple=False)
    if indices.numel() == 0:
        return None
    y0, x0 = indices.min(dim=0).values.tolist()
    y1, x1 = (indices.max(dim=0).values + 1).tolist()
    grid = raw.reshape(side, side, -1).permute(2, 0, 1)
    return grid[:, y0:y1, x0:x1].detach().to(device="cpu", dtype=torch.float16)


def _paste_warped_crop(
    raw: torch.Tensor, target_mask: torch.Tensor, donor_crop: torch.Tensor
) -> torch.Tensor:
    side = int(round(math.sqrt(raw.shape[0])))
    mask = target_mask.reshape(side, side)
    indices = torch.nonzero(mask, as_tuple=False)
    if indices.numel() == 0:
        return raw
    y0, x0 = indices.min(dim=0).values.tolist()
    y1, x1 = (indices.max(dim=0).values + 1).tolist()
    warped = F.interpolate(
        donor_crop.to(device=raw.device, dtype=raw.dtype).unsqueeze(0),
        size=(y1 - y0, x1 - x0),
        mode="bilinear",
        align_corners=False,
    )[0].permute(1, 2, 0)
    grid = raw.reshape(side, side, -1).clone()
    local_mask = mask[y0:y1, x0:x1]
    grid[y0:y1, x0:x1][local_mask] = warped[local_mask]
    return grid.reshape_as(raw)


def _record_transfer(
    rows: dict[str, list[float]],
    prefix: str,
    full: torch.Tensor,
    swapped: torch.Tensor,
    target: torch.Tensor,
    donor: torch.Tensor,
    mask: torch.Tensor,
) -> None:
    full_feature = _pool_features(full, mask)
    swap_feature = _pool_features(swapped, mask)
    target_feature = _pool_features(target, mask)
    full_n = F.normalize(full_feature, dim=-1)
    swap_n = F.normalize(swap_feature, dim=-1)
    target_n = F.normalize(target_feature, dim=-1)
    donor_n = F.normalize(donor, dim=-1)
    full_donor = (full_n * donor_n).sum(dim=-1)
    swap_donor = (swap_n * donor_n).sum(dim=-1)
    full_target = (full_n * target_n).sum(dim=-1)
    swap_target = (swap_n * target_n).sum(dim=-1)
    direction = F.cosine_similarity(
        swap_feature - full_feature, donor - target_feature, dim=-1
    )
    delta = (swapped - full).float().square().mean(dim=-1)
    inside = (delta * mask).sum(dim=-1) / mask.sum(dim=-1).clamp_min(1e-6)
    outside_mask = 1.0 - mask.clamp(0, 1)
    outside = (delta * outside_mask).sum(dim=-1) / outside_mask.sum(dim=-1).clamp_min(1e-6)
    total_inside_fraction = (delta * mask).sum(dim=-1) / delta.sum(dim=-1).clamp_min(1e-8)
    values = {
        "full_to_donor_cosine": full_donor,
        "swap_to_donor_cosine": swap_donor,
        "donor_similarity_gain": swap_donor - full_donor,
        "full_to_target_cosine": full_target,
        "swap_to_target_cosine": swap_target,
        "target_similarity_change": swap_target - full_target,
        "donor_direction_alignment": direction,
        "donor_closer": (swap_donor > full_donor).float(),
        "change_inside_outside_mse_ratio": inside / outside.clamp_min(1e-8),
        "change_energy_inside_fraction": total_inside_fraction,
    }
    for name, tensor in values.items():
        rows[f"{prefix}_{name}"].extend(tensor.detach().cpu().tolist())


@torch.no_grad()
def run(args: argparse.Namespace) -> dict:
    random.seed(int(args.seed))
    model, tokenizer, device, _ = load_model_and_tokenizer(args)
    if str(getattr(model.config, "pgot_one_shot_readout_mode", "")) != "owner_masked":
        raise ValueError("Swap diagnostic is defined for owner_masked one-shot Reader")
    loader = build_loader(
        args, tokenizer, model, args.val_jsonl, shuffle=False, max_samples=args.max_samples
    )
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    visual_dir = output / "swap_visuals"
    target_processor = model.get_vision_tower_aux_list()[-1].image_processor
    decoder = None
    if int(args.max_visualizations) > 0:
        decoder = load_rae_decoder(model, device, dtype=torch.float32)
        visual_dir.mkdir(parents=True, exist_ok=True)
    donor_bank: dict[int, list[dict]] = defaultdict(list)
    rows: dict[str, list[float]] = defaultdict(list)
    pairs = []
    visual_paths: list[str] = []

    for batch_idx, batch in enumerate(tqdm(loader, desc="one-shot swap")):
        out = pgot_forward_eval(
            model,
            images=batch["images"],
            target_images=batch["target_images"],
            caption_input_ids=batch["caption_input_ids"],
            caption_attention_mask=batch["caption_attention_mask"],
            ovt_positions_in_caption=batch["ovt_positions_in_caption"],
            ovt_valid_mask=batch["ovt_valid_mask"],
        )
        raw = out["raw_img_features"].float()
        valid = out["ovt_object_valid"].bool()
        B, K = valid.shape
        categories = _object_categories(batch, K, 1, device)
        owner = torch.cat([out["ovt_object_probs"], out["ovt_void_probs"]], dim=1).argmax(dim=1)
        masks = _object_masks(batch, K, out["gt_siglip"].shape[1], device)
        candidates = []
        if len(pairs) < args.max_pairs:
            for b in range(B):
                available = [
                    int(k)
                    for k in torch.nonzero(valid[b], as_tuple=False).flatten().tolist()
                    if donor_bank.get(int(categories[b, int(k)]))
                ]
                if available:
                    k = random.choice(available)
                    cat = int(categories[b, k])
                    candidates.append((b, k, random.choice(donor_bank[cat])))
            candidates = candidates[: args.max_pairs - len(pairs)]

        if candidates:
            swap_raw = raw.clone()
            bundle_semantic = out["semantic_slots"].float().clone()
            selected = []
            selected_masks = []
            donor_features = []
            candidate_visuals = []
            for b, k, donor in candidates:
                target_owner_mask = owner[b] == k
                swap_raw[b] = _paste_warped_crop(
                    raw[b], target_owner_mask, donor["raw_crop"]
                )
                bundle_semantic[b, k] = donor["semantic"].to(
                    device=device, dtype=bundle_semantic.dtype
                )
                selected.append(b)
                selected_masks.append(masks[b, k])
                donor_features.append(donor["target_feature"])
                pair = {
                    "category_id": int(categories[b, k]),
                    "target_sample_index": int(batch_idx * args.batch_size + b),
                    "donor_sample_index": int(donor["sample_index"]),
                }
                pairs.append(pair)
                candidate_visuals.append(
                    {
                        "pair": pair,
                        "target_image": batch["target_images"][b].detach().cpu(),
                        "target_owner_mask": target_owner_mask.detach().cpu(),
                        "donor_image": donor["source_image"],
                        "donor_owner_mask": donor["owner_mask"],
                    }
                )
            index = torch.tensor(selected, device=device, dtype=torch.long)
            mask = torch.stack(selected_masks)
            donor_feature = torch.stack(donor_features).to(device=device, dtype=torch.float32)
            appearance_condition, _ = _condition(model, out, raw_patches=swap_raw)
            bundle_condition, _ = _condition(
                model, out, raw_patches=swap_raw, semantic_slots=bundle_semantic
            )
            full_condition = out["rae_hidden"].float()[index]
            appearance_condition = appearance_condition.float()[index]
            bundle_condition = bundle_condition.float()[index]
            generation_seed = int(args.seed) + 100000 + batch_idx
            torch.manual_seed(generation_seed)
            full_generated = generate_siglip_latent(model, full_condition, args.guidance_scale).float()
            torch.manual_seed(generation_seed)
            appearance_generated = generate_siglip_latent(
                model, appearance_condition, args.guidance_scale
            ).float()
            torch.manual_seed(generation_seed)
            bundle_generated = generate_siglip_latent(
                model, bundle_condition, args.guidance_scale
            ).float()
            target_features = out["gt_siglip"].float()[index]
            _record_transfer(
                rows, "appearance", full_generated, appearance_generated,
                target_features, donor_feature, mask
            )
            _record_transfer(
                rows, "bundle", full_generated, bundle_generated,
                target_features, donor_feature, mask
            )
            if decoder is not None and len(visual_paths) < int(args.max_visualizations):
                full_images = decode_to_image(decoder, full_generated, device)
                appearance_images = decode_to_image(decoder, appearance_generated, device)
                bundle_images = decode_to_image(decoder, bundle_generated, device)
                for row_index, visual in enumerate(candidate_visuals):
                    if len(visual_paths) >= int(args.max_visualizations):
                        break
                    pair = visual["pair"]
                    target_source = _source_image(
                        visual["target_image"], target_processor
                    )
                    donor_source = _source_image(
                        visual["donor_image"], target_processor
                    )
                    tiles = [
                        _label(
                            _outline_owner(target_source, visual["target_owner_mask"]),
                            "target (red: swapped owner)",
                        ),
                        _label(
                            _outline_owner(donor_source, visual["donor_owner_mask"]),
                            "same-category donor",
                        ),
                        _label(_to_pil(full_images[row_index]), "baseline reconstruction"),
                        _label(
                            _to_pil(appearance_images[row_index]),
                            "swap raw appearance; keep target s_k",
                        ),
                        _label(
                            _to_pil(bundle_images[row_index]),
                            "swap raw appearance + s_k",
                        ),
                        _label(
                            _difference_image(
                                appearance_images[row_index], full_images[row_index]
                            ),
                            "|appearance swap - baseline|",
                        ),
                    ]
                    filename = (
                        f"pair_{len(visual_paths):02d}_cat{pair['category_id']}_"
                        f"target{pair['target_sample_index']}_"
                        f"donor{pair['donor_sample_index']}.png"
                    )
                    path = visual_dir / filename
                    _grid(tiles, columns=3).save(path)
                    pair["visual_path"] = str(path)
                    visual_paths.append(str(path))

        # Populate after sampling so no image can donate to itself.
        for b in range(B):
            for k in torch.nonzero(valid[b], as_tuple=False).flatten().tolist():
                cat = int(categories[b, int(k)])
                if cat < 0 or len(donor_bank[cat]) >= args.max_donors_per_category:
                    continue
                crop = _extract_owner_crop(raw[b], owner[b] == int(k))
                if crop is None:
                    continue
                donor_bank[cat].append(
                    {
                        "raw_crop": crop,
                        "semantic": out["semantic_slots"][b, int(k)].detach().cpu(),
                        "target_feature": _pool_features(
                            out["gt_siglip"].float()[b : b + 1],
                            masks[b : b + 1, int(k)],
                        )[0].detach().cpu(),
                        "source_image": batch["target_images"][b].detach().cpu(),
                        "owner_mask": (owner[b] == int(k)).detach().cpu(),
                        "sample_index": int(batch_idx * args.batch_size + b),
                    }
                )
        if len(pairs) >= args.max_pairs:
            break

    summary = {
        "model_path": args.model_path,
        "protocol": (
            "same-category donor raw-SigLIP owner crop warped into the fixed target "
            "owner region; appearance-only keeps target s_k, bundle also swaps s_k"
        ),
        "num_pairs": len(pairs),
        "diffusion_inference_steps": args.diffusion_inference_steps,
        "guidance_scale": args.guidance_scale,
        "visual_paths": visual_paths,
        "pairs": pairs,
        "metrics": {name: _summary(values) for name, values in sorted(rows.items())},
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2))
    log.info("Wrote %s", output / "summary.json")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--val_jsonl", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--max_samples", type=int, default=512)
    parser.add_argument("--max_pairs", type=int, default=64)
    parser.add_argument("--max_visualizations", type=int, default=6)
    parser.add_argument("--max_donors_per_category", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--grid_size", type=int, default=32)
    parser.add_argument("--max_caption_tokens", type=int, default=1024)
    parser.add_argument("--n_ovt_per_object", type=int, default=1)
    parser.add_argument("--max_objects", type=int, default=50)
    parser.add_argument("--image_preprocess_mode", default="coda_center_crop")
    parser.add_argument("--coda_crop_size", type=int, default=512)
    parser.add_argument("--dtype", choices=["fp32", "bf16"], default="fp32")
    parser.add_argument("--diffusion_inference_steps", type=int, default=10)
    parser.add_argument("--guidance_scale", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s :: %(message)s")
    print(json.dumps(run(args), indent=2))


if __name__ == "__main__":
    main()
