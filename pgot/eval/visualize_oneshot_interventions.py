"""Qualitative object interventions for the one-shot owner-masked Reader.

The one-shot Reader has no stored visual memory: appearance reaches the DiT
condition only as raw SigLIP patch values, gated by a hard owner mask
(``selected_owner == patch_owner``).  Its ``visual_memory`` tensor is a
diagnostic byproduct that ``owner_masked`` reconstruction never consumes, so the
E8/E11 memory interventions have no effect here.

This script intervenes where the information actually flows: the raw patch
values, partitioned by the model's own predicted patch ownership.  Zeroing an
owner's patches removes exactly that owner's appearance, and swapping them in
from a donor image transfers a different instance onto the target layout.
"""

import argparse
import colorsys
import json
import math
import textwrap
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from pgot.eval.analyze_ar_caption_records import _parse_objects
from pgot.eval.diagnose_e8_visual_memory import _fixed_recon_loss
from pgot.eval.pgot_inference import generate_siglip_latent, pgot_forward_eval
from pgot.eval.pgot_metrics import fari_metric, mbo_metric, miou_metric
from pgot.eval.run_eval import (
    CocoInstanceMaskCache,
    decode_to_image,
    denormalize_images,
    load_rae_decoder,
)
from pgot.model.pgot_utils import build_pred_mask_ovt_owner_eval
from pgot.eval.visualize_e8_memory_interventions import (
    _grid,
    _label,
    _mask_overlay,
    _parse_pair,
    _to_pil,
)
from pgot.eval.visualize_ovt_overlays import _load_model
from pgot.train.pgot_dataset import PGOTDataCollator, Pix2CapPGOTDataset


def _patch_owner(owner_probs: torch.Tensor, slot_valid: torch.Tensor) -> torch.Tensor:
    """Hard patch->owner assignment, matching the Reader's own routing argmax."""
    routing = owner_probs.float() * slot_valid.unsqueeze(-1).float()
    return routing.argmax(dim=1)  # [B, P]


def _font(size: int = 15) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except Exception:
        return ImageFont.load_default()


def _instance_overlay(source: Image.Image, mask: torch.Tensor) -> Image.Image:
    """Overlay integer instance ids and label sufficiently large regions."""
    source = source.convert("RGB").resize((512, 512), Image.Resampling.BILINEAR)
    ids = mask.detach().cpu().numpy().astype(np.int64)
    if ids.shape != (512, 512):
        ids_image = Image.fromarray(ids.astype(np.int32), mode="I")
        ids = np.asarray(ids_image.resize((512, 512), Image.Resampling.NEAREST))
    rgb = np.asarray(source, dtype=np.float32).copy()
    draw_labels = []
    for instance_id in sorted(int(value) for value in np.unique(ids) if value > 0):
        hue = (instance_id * 0.61803398875) % 1.0
        color = np.asarray(colorsys.hsv_to_rgb(hue, 0.72, 1.0)) * 255.0
        region = ids == instance_id
        rgb[region] = 0.48 * rgb[region] + 0.52 * color
        yy, xx = np.nonzero(region)
        if len(xx) >= 80:
            draw_labels.append((instance_id, int(np.median(xx)), int(np.median(yy))))
    image = Image.fromarray(rgb.clip(0, 255).astype(np.uint8), mode="RGB")
    draw = ImageDraw.Draw(image)
    for instance_id, x, y in draw_labels:
        text = str(instance_id)
        box = draw.textbbox((x, y), text, font=_font(16), anchor="mm")
        draw.rectangle(box, fill=(255, 255, 255))
        draw.text((x, y), text, fill=(0, 0, 0), font=_font(16), anchor="mm")
    return image


def _text_panel(lines: list[str], size: tuple[int, int] = (512, 512)) -> Image.Image:
    panel = Image.new("RGB", size, "white")
    draw = ImageDraw.Draw(panel)
    y = 8
    font = _font(13)
    for line in lines:
        wrapped = textwrap.wrap(str(line), width=64) or [""]
        for segment in wrapped:
            if y > size[1] - 18:
                draw.text((8, y), "...", fill="black", font=font)
                return panel
            draw.text((8, y), segment, fill="black", font=font)
            y += 17
        y += 3
    return panel


def _caption_tensors(
    text: str, tokenizer, *, max_caption_tokens: int, max_objects: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    ids = tokenizer.encode(text, add_special_tokens=False)[: int(max_caption_tokens)]
    token_ids = torch.tensor(ids, dtype=torch.long).unsqueeze(0)
    attention_mask = torch.ones_like(token_ids, dtype=torch.bool)
    ovt_id = int(tokenizer.convert_tokens_to_ids("<ovt>"))
    positions = torch.nonzero(token_ids[0] == ovt_id, as_tuple=False).flatten()
    positions = positions[: int(max_objects)]
    ovt_positions = torch.zeros((1, int(max_objects)), dtype=torch.long)
    ovt_valid = torch.zeros((1, int(max_objects)), dtype=torch.bool)
    if positions.numel():
        ovt_positions[0, : positions.numel()] = positions
        ovt_valid[0, : positions.numel()] = True
    return token_ids, attention_mask, ovt_positions, ovt_valid


def _grid_coords(patch_index: torch.Tensor, side: int) -> torch.Tensor:
    rows = torch.div(patch_index, side, rounding_mode="floor").float()
    cols = (patch_index % side).float()
    return torch.stack([rows, cols], dim=-1)


def _normalized_within_box(coords: torch.Tensor) -> torch.Tensor:
    """Map coordinates to [0,1]^2 inside their own bounding box."""
    lo = coords.min(dim=0).values
    hi = coords.max(dim=0).values
    span = (hi - lo).clamp_min(1.0)
    return (coords - lo) / span


def _swap_patches(
    target_raw: torch.Tensor,
    target_owner: torch.Tensor,
    target_object: int,
    donor_raw: torch.Tensor,
    donor_owner: torch.Tensor,
    donor_object: int,
    side: int,
) -> torch.Tensor:
    """Paste the donor object's patches onto the target object's patch set.

    Correspondence is by relative position inside each object's bounding box,
    so a donor of a different size and shape still lands on the target layout.
    """
    swapped = target_raw.clone()
    target_slots = (target_owner[0] == target_object).nonzero(as_tuple=True)[0]
    donor_slots = (donor_owner[0] == donor_object).nonzero(as_tuple=True)[0]
    if target_slots.numel() == 0 or donor_slots.numel() == 0:
        return swapped
    target_norm = _normalized_within_box(_grid_coords(target_slots, side))
    donor_norm = _normalized_within_box(_grid_coords(donor_slots, side))
    distance = torch.cdist(target_norm, donor_norm)
    nearest = distance.argmin(dim=1)
    swapped[0, target_slots] = donor_raw[0, donor_slots[nearest]]
    return swapped


@torch.no_grad()
def run(args: argparse.Namespace) -> dict:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32 if args.dtype == "fp32" else torch.bfloat16
    model, tokenizer = _load_model(args.model_path, dtype=dtype, device=device)
    if not bool(getattr(model.config, "pgot_one_shot_reader_enable", False)):
        raise ValueError(
            "This script requires a one-shot Reader checkpoint "
            "(pgot_one_shot_reader_enable=True)"
        )
    readout_mode = str(getattr(model.config, "pgot_one_shot_readout_mode", "pooled"))

    vision_towers = model.get_vision_tower_aux_list()
    image_processor = vision_towers[0].image_processor
    target_processor = (
        vision_towers[1].image_processor if len(vision_towers) > 1 else image_processor
    )
    dataset = Pix2CapPGOTDataset(
        jsonl_path=args.val_jsonl,
        tokenizer=tokenizer,
        image_processor=image_processor,
        target_image_processor=target_processor,
        grid_size=args.grid_size,
        max_caption_tokens=args.max_caption_tokens,
        n_ovt_per_object=1,
        max_objects=50,
        panoptic_categories_json="/home/jovyan/data/coco/annotations/panoptic_val2017.json",
        image_preprocess_mode=args.image_preprocess_mode,
        coda_crop_size=args.coda_crop_size,
    )
    collator = PGOTDataCollator(pad_token_id=tokenizer.pad_token_id)
    decoder = load_rae_decoder(model, device, dtype)

    if args.diffusion_inference_steps != 50:
        from scale_rae.model.diffusion_loss.diffusion import create_diffusion

        inference = model.diff_head.inference_flow
        model.diff_head.inference_flow = create_diffusion(
            str(args.diffusion_inference_steps),
            noise_schedule="linear",
            use_kl=False,
            sigma_small=False,
            predict_xstart=False,
            learn_sigma=False,
            rescale_learned_sigmas=False,
            diffusion_steps=int(getattr(inference, "diffusion_steps", 1000)),
            input_base_dimension_ratio=float(getattr(inference, "size_ratio", 1.0)),
            diffusion_type="rf",
            use_loss_weighting=False,
        )

    with open(args.val_jsonl) as handle:
        raw_samples = [json.loads(line) for line in handle]

    mean = torch.tensor(target_processor.image_mean)
    std = torch.tensor(target_processor.image_std)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    ar_visual_records = []

    def forward_features(batch: dict) -> dict:
        return model._pgot_one_shot_forward_features(
            images=batch["images"],
            target_images=batch["target_images"],
            caption_input_ids=batch["caption_input_ids"],
            caption_attention_mask=batch["caption_attention_mask"],
            ovt_positions_in_caption=batch["ovt_positions_in_caption"],
            ovt_valid_mask=batch["ovt_valid_mask"],
        )

    if args.ar_records and args.ar_sample:
        with open(args.ar_records) as handle:
            ar_records = [json.loads(line) for line in handle]
        if len(ar_records) != len(raw_samples):
            raise ValueError(
                "AR record JSONL and validation JSONL must be index-aligned"
            )
        coco_cache = CocoInstanceMaskCache(args.coco_mask_cache)
        for sample_index in args.ar_sample:
            batch = collator([dataset[int(sample_index)]])
            ar_record = ar_records[int(sample_index)]
            caption_ids, caption_mask, ovt_positions, ovt_valid = _caption_tensors(
                ar_record["text"],
                tokenizer,
                max_caption_tokens=args.max_caption_tokens,
                max_objects=50,
            )
            out = pgot_forward_eval(
                model,
                images=batch["images"],
                target_images=batch["target_images"],
                caption_input_ids=caption_ids,
                caption_attention_mask=caption_mask,
                ovt_positions_in_caption=ovt_positions,
                ovt_valid_mask=ovt_valid,
            )
            pred_mask = build_pred_mask_ovt_owner_eval(
                ovt_object_probs=out["ovt_object_probs"],
                ovt_void_probs=out["ovt_void_probs"],
                ovt_valid_mask=out["ovt_valid_mask"],
                ovt_is_thing=out["ovt_valid_mask"],
                target_size=512,
                n_ovt_per_object=1,
                patch_grid=32,
                map_stuff_to_bg=True,
            )[0]
            sample = raw_samples[int(sample_index)]
            gt_mask = coco_cache.get(int(sample["image_id"]))
            if gt_mask is None:
                raise KeyError(f"No COCO cache mask for image {sample['image_id']}")
            overlap = coco_cache.get_overlap(int(sample["image_id"]))
            gt_batch = gt_mask.unsqueeze(0).to(pred_mask.device)
            pred_batch = pred_mask.unsqueeze(0)
            overlap_batch = (
                overlap.unsqueeze(0).to(pred_mask.device) if overlap is not None else None
            )
            metrics = {
                "fARI": fari_metric(gt_batch, pred_batch, overlap_batch),
                "mBO": mbo_metric(gt_batch, pred_batch, overlap_batch),
                "mIoU": miou_metric(gt_batch, pred_batch, overlap_batch),
            }
            source = denormalize_images(
                batch["target_images"].float(), mean, std
            )[0]
            source_pil = _to_pil(source).resize((512, 512), Image.Resampling.BILINEAR)
            parsed = _parse_objects(ar_record["text"])
            gt_categories = [str(segment["category"]) for segment in sample["segments"]]
            metric_gt_count = int((torch.unique(gt_mask) > 0).sum().item())
            active_pred_count = int((torch.unique(pred_mask) > 0).sum().item())
            caption_lines = [
                f"sample {sample_index} | image {sample['image_id']}",
                f"caption GT={len(gt_categories)} | COCO metric GT={metric_gt_count}",
                f"all OVT={ar_record['object_count']} | complete records={len(parsed)} | spatially active={active_pred_count}",
                f"scene_end={ar_record['scene_end_generated']} | fARI={metrics['fARI']:.3f} mBO={metrics['mBO']:.3f} mIoU={metrics['mIoU']:.3f}",
                "",
                "GT: " + ", ".join(gt_categories),
                "",
                "AR object records:",
            ]
            for object_index, (category, description) in enumerate(parsed, start=1):
                caption_lines.append(
                    f"{object_index}. {category}: {description[:95]}"
                )
            tiles = [
                _label(source_pil, "CODA-cropped source"),
                _label(_instance_overlay(source_pil, gt_mask), "COCO GT instances"),
                _label(_instance_overlay(source_pil, pred_mask), "AR predicted semantic-owner instances"),
                _label(_text_panel(caption_lines), "Generated object-caption table"),
            ]
            output_path = output_dir / f"ar_sample{sample_index}_segmentation_caption.png"
            _grid(tiles, output_path, columns=2)
            ar_visual_records.append(
                {
                    "sample_index": int(sample_index),
                    "image_id": int(sample["image_id"]),
                    "caption_gt_count": len(gt_categories),
                    "coco_metric_gt_count": metric_gt_count,
                    "counted_ovt": int(ar_record["object_count"]),
                    "complete_object_records": len(parsed),
                    "spatially_active_predicted_objects": active_pred_count,
                    "scene_end_generated": bool(ar_record["scene_end_generated"]),
                    "metrics": metrics,
                    "grid": str(output_path),
                }
            )

    for pair_index, spec in enumerate(args.pair):
        target_index, target_object, donor_index, donor_object = _parse_pair(spec)
        target_batch = collator([dataset[target_index]])
        donor_batch = collator([dataset[donor_index]])
        target_seq = forward_features(target_batch)
        donor_seq = forward_features(donor_batch)

        object_count = int(target_seq["object_valid"].shape[1])
        donor_object_count = int(donor_seq["object_valid"].shape[1])
        if target_object >= object_count or donor_object >= donor_object_count:
            raise IndexError(f"Invalid object index in pair {spec}")

        target_segment = raw_samples[target_index]["segments"][target_object]
        donor_segment = raw_samples[donor_index]["segments"][donor_object]
        if int(target_segment["category_id"]) != int(donor_segment["category_id"]):
            raise ValueError(
                f"Pair {spec} is not same-category: "
                f"{target_segment['category']} vs {donor_segment['category']}"
            )

        raw = target_seq["raw_img_features"].float()
        owner_probs = target_seq["owner_probs"].float()
        slot_valid = target_seq["slot_valid"]
        patch_owner = _patch_owner(owner_probs, slot_valid)
        donor_patch_owner = _patch_owner(
            donor_seq["owner_probs"].float(), donor_seq["slot_valid"]
        )
        side = int(round(math.sqrt(int(raw.shape[1]))))

        is_object_patch = patch_owner < object_count
        is_selected_patch = patch_owner == target_object

        object_only = raw.clone()
        object_only[~is_object_patch.unsqueeze(-1).expand_as(raw)] = 0.0
        register_only = raw.clone()
        register_only[is_object_patch.unsqueeze(-1).expand_as(raw)] = 0.0
        all_zero = torch.zeros_like(raw)
        selected_zero = raw.clone()
        selected_zero[is_selected_patch.unsqueeze(-1).expand_as(raw)] = 0.0
        selected_swap = _swap_patches(
            raw,
            patch_owner,
            target_object,
            donor_seq["raw_img_features"].float(),
            donor_patch_owner,
            donor_object,
            side,
        )
        bundle_semantics = target_seq["semantic_slots"].float().clone()
        bundle_semantics[:, target_object] = donor_seq["semantic_slots"].float()[
            :, donor_object
        ]

        def condition_for(
            patches: torch.Tensor, semantics: torch.Tensor | None = None
        ) -> torch.Tensor:
            reader_semantics = (
                target_seq["semantic_slots"] if semantics is None else semantics
            )
            return model.pgot_e8_reader(
                rae_queries=target_seq["raw_rae_hidden"],
                semantic_slots=reader_semantics.to(target_seq["semantic_slots"].dtype),
                raw_patches=patches.to(target_seq["raw_img_features"].dtype),
                owner_probs=owner_probs,
                slot_valid=slot_valid,
            )["condition_hidden"].float()

        conditions = {
            "full": target_seq["condition_hidden"].float(),
            "object_only": condition_for(object_only),
            "register_only": condition_for(register_only),
            "all_zero": condition_for(all_zero),
            "selected_zero": condition_for(selected_zero),
            "same_category_swap": condition_for(selected_swap),
            "same_category_bundle_swap": condition_for(selected_swap, bundle_semantics),
        }

        losses = {
            name: _fixed_recon_loss(
                model, condition, target_seq["gt_siglip"], args.seed + pair_index
            )
            for name, condition in conditions.items()
        }
        decoded = {}
        for name, condition in conditions.items():
            torch.manual_seed(args.seed + pair_index)
            generated = generate_siglip_latent(
                model, condition, guidance_level=args.guidance_scale
            )
            decoded[name] = decode_to_image(decoder, generated, condition.device)[0].cpu()

        target_source = denormalize_images(target_batch["target_images"].float(), mean, std)[0]
        donor_source = denormalize_images(donor_batch["target_images"].float(), mean, std)[0]
        target_pil = _mask_overlay(
            _to_pil(target_source),
            target_batch["gt_masks_per_ovt"][0, target_object],
            (255, 30, 30),
        )
        donor_pil = _mask_overlay(
            _to_pil(donor_source),
            donor_batch["gt_masks_per_ovt"][0, donor_object],
            (30, 100, 255),
        )
        difference = (decoded["same_category_swap"] - decoded["full"]).abs().mul(4.0).clamp(0, 1)
        bundle_difference = (
            decoded["same_category_bundle_swap"] - decoded["full"]
        ).abs().mul(4.0).clamp(0, 1)
        semantic_effect = (
            decoded["same_category_bundle_swap"] - decoded["same_category_swap"]
        ).abs().mul(4.0).clamp(0, 1)
        category = str(target_segment["category"])
        selected_patch_count = int(is_selected_patch.sum())
        tiles = [
            _label(target_pil, f"target source | {category} #{target_object} (red)"),
            _label(donor_pil, f"donor source | {category} #{donor_object} (blue)"),
            _label(_to_pil(decoded["full"]), f"full | loss {losses['full']:.4f}"),
            _label(_to_pil(decoded["object_only"]), f"object patches only | {losses['object_only']:.4f}"),
            _label(_to_pil(decoded["register_only"]), f"register patches only | {losses['register_only']:.4f}"),
            _label(_to_pil(decoded["all_zero"]), f"all patches zero | {losses['all_zero']:.4f}"),
            _label(
                _to_pil(decoded["selected_zero"]),
                f"selected {category} patches zero | {losses['selected_zero']:.4f}",
            ),
            _label(
                _to_pil(decoded["same_category_swap"]),
                f"same-category patch swap | {losses['same_category_swap']:.4f}",
            ),
            _label(_to_pil(difference), "|swap - full| x4"),
            _label(
                _to_pil(decoded["same_category_bundle_swap"]),
                f"semantic + patch bundle swap | {losses['same_category_bundle_swap']:.4f}",
            ),
            _label(_to_pil(bundle_difference), "|bundle swap - full| x4"),
            _label(_to_pil(semantic_effect), "|bundle swap - patch swap| x4"),
        ]
        output_path = output_dir / (
            f"sample{target_index}_obj{target_object}_"
            f"{category.replace(' ', '_')}_patch_grid.png"
        )
        _grid(tiles, output_path)
        records.append(
            {
                "target_sample_index": target_index,
                "target_image_id": raw_samples[target_index]["image_id"],
                "target_object_index": target_object,
                "donor_sample_index": donor_index,
                "donor_image_id": raw_samples[donor_index]["image_id"],
                "donor_object_index": donor_object,
                "category": category,
                "selected_object_patch_count": selected_patch_count,
                "object_patch_count": int(is_object_patch.sum()),
                "fixed_noise_seed": args.seed + pair_index,
                "guidance_scale": args.guidance_scale,
                "diffusion_inference_steps": args.diffusion_inference_steps,
                "losses": losses,
                "grid": str(output_path),
            }
        )

    summary = {
        "model_path": args.model_path,
        "one_shot_readout_mode": readout_mode,
        "protocol": (
            "one-shot Reader: same semantic owners/ownership/noise; interventions "
            "zero or swap raw SigLIP patch values grouped by the model's own "
            "predicted patch ownership"
        ),
        "records": records,
        "ar_segmentation_caption_records": ar_visual_records,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--val_jsonl", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument(
        "--pair",
        action="append",
        required=True,
        help="target_sample:target_object:donor_sample:donor_object",
    )
    parser.add_argument("--grid_size", type=int, default=32)
    parser.add_argument("--max_caption_tokens", type=int, default=1024)
    parser.add_argument("--image_preprocess_mode", default="coda_center_crop")
    parser.add_argument("--coda_crop_size", type=int, default=512)
    parser.add_argument("--dtype", choices=["fp32", "bf16"], default="fp32")
    parser.add_argument("--diffusion_inference_steps", type=int, default=10)
    parser.add_argument("--guidance_scale", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--ar_records", default=None)
    parser.add_argument("--ar_sample", action="append", type=int, default=[])
    parser.add_argument(
        "--coco_mask_cache",
        default="/home/jovyan/PGOT/data/coco_inst_mask_cache_coda512",
    )
    args = parser.parse_args()
    summary = run(args)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
