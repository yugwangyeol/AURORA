"""Controlled same-category object-memory transfer with matched diffusion noise."""

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
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

from pgot.eval.eval_recon_oracles import build_loader, load_model_and_tokenizer
from pgot.eval.pgot_inference import generate_siglip_latent, pgot_forward_eval
from pgot.eval.run_eval import decode_to_image, load_rae_decoder


LOG = logging.getLogger("pgot.eval_object_memory_transfer")


def summary(values: list[float], seed: int) -> dict:
    if not values:
        return {"mean": None, "median": None, "std": None, "ci95": [None, None], "count": 0}
    array = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    boot = np.empty(2000, dtype=np.float64)
    for index in range(boot.size):
        boot[index] = rng.choice(array, size=array.size, replace=True).mean()
    lo, hi = np.quantile(boot, [0.025, 0.975])
    return {
        "mean": float(array.mean()), "median": float(np.median(array)),
        "std": float(array.std()), "ci95": [float(lo), float(hi)],
        "count": int(array.size),
    }


def resize_masks(batch: dict, k_objects: int, patches: int, device) -> tuple[torch.Tensor, torch.Tensor]:
    masks = batch["gt_masks_per_ovt"][:, :k_objects].to(device=device, dtype=torch.float32)
    valid = batch["ovt_valid_mask"][:, :k_objects].to(device=device, dtype=torch.bool)
    src = int(round(math.sqrt(masks.shape[-1])))
    dst = int(round(math.sqrt(patches)))
    if src * src != masks.shape[-1] or dst * dst != patches:
        raise ValueError("Transfer evaluation expects square spatial grids")
    masks = F.interpolate(
        masks.reshape(-1, 1, src, src), size=(dst, dst), mode="area"
    ).reshape(masks.shape[0], k_objects, patches).clamp(0, 1)
    valid &= masks.sum(dim=-1) > 0
    return masks.masked_fill(~valid.unsqueeze(-1), 0), valid


def pooled(features: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return torch.einsum("bp,bpd->bd", mask, features) / mask.sum(dim=-1, keepdim=True).clamp_min(1e-6)


def masked_change(delta: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    per_patch = delta.float().square().mean(dim=-1)
    inside = (per_patch * mask).sum(dim=-1) / mask.sum(dim=-1).clamp_min(1e-6)
    outside_mask = (1 - mask).clamp(0, 1)
    outside = (per_patch * outside_mask).sum(dim=-1) / outside_mask.sum(dim=-1).clamp_min(1e-6)
    return inside, outside


@torch.no_grad()
def run(args: argparse.Namespace) -> dict:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    model, tokenizer, device, _ = load_model_and_tokenizer(args)
    if not bool(getattr(model.config, "pgot_one_shot_reader_enable", False)):
        raise ValueError("Memory transfer requires a one-shot visual-memory checkpoint")
    loader = build_loader(
        args, tokenizer, model, args.val_jsonl, shuffle=False, max_samples=args.max_samples
    )
    decoder = load_rae_decoder(model, device=device, dtype=torch.float32)
    lpips = LearnedPerceptualImagePatchSimilarity(
        net_type="alex", normalize=True
    ).to(device).eval()
    donor_bank: dict[int, dict] = {}
    values: dict[str, list[float]] = defaultdict(list)
    pair_rows = []
    sample_index = 0

    for batch in tqdm(loader, desc="Controlled object transfer"):
        out = pgot_forward_eval(
            model,
            images=batch["images"], target_images=batch["target_images"],
            caption_input_ids=batch["caption_input_ids"],
            caption_attention_mask=batch["caption_attention_mask"],
            ovt_positions_in_caption=batch["ovt_positions_in_caption"],
            ovt_valid_mask=batch["ovt_valid_mask"],
        )
        memory = out["visual_memory"].float()
        if memory.ndim != 4:
            raise ValueError(f"Expected [B,S,J,D] visual memory, got {tuple(memory.shape)}")
        record = out["e8_write_records"][-1]
        memory_valid = record["memory_valid"].bool()
        object_valid = out["ovt_object_valid"].bool()
        batch_size, k_objects = object_valid.shape
        masks, gt_valid = resize_masks(batch, k_objects, out["gt_siglip"].shape[1], device)
        object_valid &= gt_valid
        categories = batch["ovt_category_ids"][:, :k_objects].to(device=device, dtype=torch.long)

        for b in range(batch_size):
            target_k = None
            donor = None
            for k in torch.nonzero(object_valid[b], as_tuple=False).flatten().tolist():
                category = int(categories[b, k])
                if category >= 0 and category in donor_bank:
                    target_k, donor = int(k), donor_bank[category]
                    break
            if target_k is not None and len(pair_rows) < args.max_pairs:
                category = int(categories[b, target_k])
                swap_memory = memory[b:b+1].clone()
                swap_memory[0, target_k] = donor["memory"].to(device=device, dtype=memory.dtype)
                reader = model.pgot_e8_reader(
                    rae_queries=out["raw_rae_hidden"][b:b+1],
                    semantic_slots=out["semantic_slots"][b:b+1],
                    visual_memory=swap_memory,
                    slot_valid=memory_valid[b:b+1].any(dim=-1),
                    memory_valid=memory_valid[b:b+1],
                )
                generation_seed = args.seed + 100000 + len(pair_rows)
                torch.manual_seed(generation_seed)
                unedited = generate_siglip_latent(
                    model, out["rae_hidden"][b:b+1].float(),
                    guidance_level=args.guidance_scale,
                ).float()
                torch.manual_seed(generation_seed)
                edited = generate_siglip_latent(
                    model, reader["condition_hidden"].float(),
                    guidance_level=args.guidance_scale,
                ).float()
                target_mask = masks[b:b+1, target_k]
                before_feature = pooled(unedited, target_mask)
                after_feature = pooled(edited, target_mask)
                target_feature = pooled(out["gt_siglip"][b:b+1].float(), target_mask)
                donor_feature = donor["feature"].to(device=device).unsqueeze(0)
                before_n = F.normalize(before_feature, dim=-1)
                after_n = F.normalize(after_feature, dim=-1)
                target_n = F.normalize(target_feature, dim=-1)
                donor_n = F.normalize(donor_feature, dim=-1)
                before_donor = (before_n * donor_n).sum(dim=-1)
                after_donor = (after_n * donor_n).sum(dim=-1)
                donor_gain = after_donor - before_donor
                target_change = 1 - (before_n * after_n).sum(dim=-1)
                inside_mse, outside_mse = masked_change(edited - unedited, target_mask)
                unedited_image = decode_to_image(decoder, unedited, device)
                edited_image = decode_to_image(decoder, edited, device)
                mask_side = int(round(math.sqrt(target_mask.shape[-1])))
                image_mask = F.interpolate(
                    target_mask.reshape(1, 1, mask_side, mask_side),
                    size=unedited_image.shape[-2:], mode="nearest",
                )
                # Neutralize the target itself before perceptual comparison;
                # remaining LPIPS measures changes outside the edited region.
                outside_composite = (
                    edited_image * (1 - image_mask) + unedited_image * image_mask
                )
                outside_lpips = lpips(outside_composite, unedited_image)
                outside_l1 = (
                    (edited_image - unedited_image).abs() * (1 - image_mask)
                ).sum() / ((1 - image_mask).sum() * edited_image.shape[1]).clamp_min(1)
                row = {
                    "pair_index": len(pair_rows), "category_id": category,
                    "target_sample_index": sample_index + b, "target_object_index": target_k,
                    "donor_sample_index": donor["sample_index"],
                    "donor_object_index": donor["object_index"],
                    "diffusion_seed": generation_seed,
                    "donor_similarity_before": float(before_donor),
                    "donor_similarity_after": float(after_donor),
                    "donor_gain": float(donor_gain),
                    "donor_closer": float(donor_gain > 0),
                    "target_similarity_before": float((before_n * target_n).sum(dim=-1)),
                    "target_similarity_after": float((after_n * target_n).sum(dim=-1)),
                    "target_change": float(target_change),
                    "inside_target_latent_mse": float(inside_mse),
                    "outside_target_latent_mse": float(outside_mse),
                    "outside_target_lpips": float(outside_lpips),
                    "outside_target_pixel_l1": float(outside_l1),
                }
                pair_rows.append(row)
                for key in (
                    "donor_similarity_before", "donor_similarity_after", "donor_gain",
                    "donor_closer", "target_change", "inside_target_latent_mse",
                    "outside_target_latent_mse", "outside_target_lpips",
                    "outside_target_pixel_l1",
                ):
                    values[key].append(row[key])

            # Populate after selection: a target never donates to itself.
            for k in torch.nonzero(object_valid[b], as_tuple=False).flatten().tolist():
                category = int(categories[b, k])
                if category < 0 or category in donor_bank:
                    continue
                donor_bank[category] = {
                    "memory": memory[b, k].detach().cpu(),
                    "feature": pooled(
                        out["gt_siglip"][b:b+1].float(), masks[b:b+1, k]
                    )[0].detach().cpu(),
                    "sample_index": sample_index + b,
                    "object_index": int(k),
                }
        sample_index += batch_size
        if len(pair_rows) >= args.max_pairs:
            break

    if len(pair_rows) != args.max_pairs:
        LOG.warning("Requested %d pairs but produced %d", args.max_pairs, len(pair_rows))
    result = {
        "model_path": args.model_path,
        "num_pairs": len(pair_rows),
        "protocol": {
            "selection": "deterministic same-category previous-image donor; oracle object IDs",
            "intervention": "replace all visual memories of target owner; keep target semantic token and registers",
            "reference": "same-model unedited reconstruction with identical diffusion noise",
            "primary": "donor_gain up together with outside_target_lpips down",
            "preservation_metric": "AlexNet LPIPS after replacing target pixels with the unedited reference; same-seed model reconstruction",
        },
        "metrics": {key: summary(item, args.seed) for key, item in sorted(values.items())},
    }
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    with (output / "pairs.jsonl").open("w") as stream:
        for row in pair_rows:
            stream.write(json.dumps(row) + "\n")
    print(json.dumps(result, indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--val_jsonl", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--max_pairs", type=int, default=100)
    parser.add_argument("--max_samples", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=2)
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
    run(args)


if __name__ == "__main__":
    main()
