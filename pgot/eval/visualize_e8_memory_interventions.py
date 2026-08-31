"""Generate fixed-noise reconstruction grids for E8 visual-memory interventions.

Each grid keeps the target semantic slots fixed and compares full memory,
object/register ablations, one-object zeroing, and a same-category visual-memory
swap.  This is a qualitative companion to diagnose_e8_visual_memory.py and
diagnose_e8_paired_swap.py; it never changes checkpoint weights.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from pgot.eval.diagnose_e8_visual_memory import _fixed_recon_loss, _reader_condition
from pgot.eval.pgot_inference import generate_siglip_latent, pgot_forward_eval
from pgot.eval.run_eval import decode_to_image, denormalize_images, load_rae_decoder
from pgot.eval.visualize_ovt_overlays import _load_model
from pgot.train.pgot_dataset import PGOTDataCollator, Pix2CapPGOTDataset


def _font(size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except Exception:
        return ImageFont.load_default()


def _to_pil(image: torch.Tensor) -> Image.Image:
    array = (
        image.detach().float().permute(1, 2, 0).cpu().clamp(0, 1).numpy() * 255
    ).round().astype(np.uint8)
    return Image.fromarray(array, mode="RGB")


def _label(tile: Image.Image, text: str, height: int = 42) -> Image.Image:
    tile = tile.convert("RGB")
    canvas = Image.new("RGB", (tile.width, tile.height + height), "white")
    canvas.paste(tile, (0, height))
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 5), text, fill="black", font=_font(14))
    return canvas


def _mask_overlay(image: Image.Image, flat_mask: torch.Tensor, color: tuple[int, int, int]) -> Image.Image:
    side = int(round(math.sqrt(flat_mask.numel())))
    mask = flat_mask.float().reshape(side, side).cpu().numpy()
    mask = Image.fromarray((mask.clip(0, 1) * 255).astype(np.uint8), mode="L")
    mask = mask.resize(image.size, Image.Resampling.NEAREST)
    alpha = mask.point(lambda value: int(value * 0.35))
    overlay = Image.new("RGB", image.size, color)
    return Image.composite(overlay, image.convert("RGB"), alpha)


def _grid(tiles: list[Image.Image], output_path: Path, columns: int = 3) -> None:
    width = max(tile.width for tile in tiles)
    height = max(tile.height for tile in tiles)
    normalized = [
        tile if tile.size == (width, height) else tile.resize((width, height), Image.Resampling.BILINEAR)
        for tile in tiles
    ]
    rows = math.ceil(len(normalized) / columns)
    canvas = Image.new("RGB", (columns * width, rows * height), "white")
    for index, tile in enumerate(normalized):
        canvas.paste(tile, ((index % columns) * width, (index // columns) * height))
    canvas.save(output_path)


def _parse_pair(spec: str) -> tuple[int, int, int, int]:
    values = tuple(int(value.strip()) for value in spec.split(":"))
    if len(values) != 4:
        raise ValueError(
            "Pair must be target_sample:target_object:donor_sample:donor_object, "
            f"got {spec!r}"
        )
    return values


def _forward(model, batch: dict) -> dict:
    return pgot_forward_eval(
        model,
        images=batch["images"],
        target_images=batch["target_images"],
        caption_input_ids=batch["caption_input_ids"],
        caption_attention_mask=batch["caption_attention_mask"],
        ovt_positions_in_caption=batch["ovt_positions_in_caption"],
        ovt_valid_mask=batch["ovt_valid_mask"],
    )


def _reader_condition_with_semantics(
    model,
    out: dict,
    memory: torch.Tensor,
    semantic_slots: torch.Tensor,
) -> torch.Tensor:
    """Read visual values with an explicitly intervened semantic owner state."""
    object_valid = out["ovt_object_valid"].bool()
    n_register = memory.shape[1] - object_valid.shape[1]
    slot_valid = torch.cat(
        [
            object_valid,
            torch.ones(
                object_valid.shape[0],
                n_register,
                device=object_valid.device,
                dtype=torch.bool,
            ),
        ],
        dim=1,
    )
    return model.pgot_e8_reader(
        rae_queries=out["raw_rae_hidden"],
        semantic_slots=semantic_slots,
        visual_memory=memory,
        slot_valid=slot_valid,
        memory_centroids=out.get("memory_centroids"),
        object_count=object_valid.shape[1],
    )["condition_hidden"]


@torch.no_grad()
def run(args: argparse.Namespace) -> dict:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32 if args.dtype == "fp32" else torch.bfloat16
    model, tokenizer = _load_model(args.model_path, dtype, device)
    if not bool(getattr(model.config, "pgot_e8_visual_memory_enable", False)):
        raise ValueError("This visualizer requires an E8 visual-memory checkpoint")

    towers = model.get_vision_tower_aux_list()
    image_processor = towers[0].image_processor
    target_processor = towers[1].image_processor if len(towers) > 1 else image_processor
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

    for pair_index, spec in enumerate(args.pair):
        target_index, target_object, donor_index, donor_object = _parse_pair(spec)
        target_batch = collator([dataset[target_index]])
        donor_batch = collator([dataset[donor_index]])
        target_out = _forward(model, target_batch)
        donor_out = _forward(model, donor_batch)

        memory = target_out["visual_memory"].float()
        donor_memory = donor_out["visual_memory"].float()
        object_count = target_out["ovt_object_valid"].shape[1]
        donor_object_count = donor_out["ovt_object_valid"].shape[1]
        if target_object >= object_count or donor_object >= donor_object_count:
            raise IndexError(f"Invalid object index in pair {spec}")

        target_segment = raw_samples[target_index]["segments"][target_object]
        donor_segment = raw_samples[donor_index]["segments"][donor_object]
        if int(target_segment["category_id"]) != int(donor_segment["category_id"]):
            raise ValueError(
                f"Pair {spec} is not same-category: "
                f"{target_segment['category']} vs {donor_segment['category']}"
            )

        object_only = memory.clone()
        object_only[:, object_count:] = 0
        register_only = memory.clone()
        register_only[:, :object_count] = 0
        all_zero = torch.zeros_like(memory)
        selected_zero = memory.clone()
        selected_zero[:, target_object] = 0
        selected_swap = memory.clone()
        selected_swap[:, target_object] = donor_memory[:, donor_object]
        bundle_semantics = target_out["semantic_slots"].float().clone()
        bundle_semantics[:, target_object] = donor_out["semantic_slots"].float()[
            :, donor_object
        ]

        conditions = {
            "full": target_out["rae_hidden"].float(),
            "object_only": _reader_condition(model, target_out, object_only).float(),
            "register_only": _reader_condition(model, target_out, register_only).float(),
            "all_zero": _reader_condition(model, target_out, all_zero).float(),
            "selected_zero": _reader_condition(model, target_out, selected_zero).float(),
            "same_category_swap": _reader_condition(model, target_out, selected_swap).float(),
            "same_category_bundle_swap": _reader_condition_with_semantics(
                model,
                target_out,
                selected_swap,
                bundle_semantics,
            ).float(),
        }

        losses = {
            name: _fixed_recon_loss(
                model,
                condition,
                target_out["gt_siglip"],
                args.seed + pair_index,
            )
            for name, condition in conditions.items()
        }
        decoded = {}
        for name, condition in conditions.items():
            torch.manual_seed(args.seed + pair_index)
            generated = generate_siglip_latent(
                model,
                condition,
                guidance_level=args.guidance_scale,
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
        tiles = [
            _label(target_pil, f"target source | {category} #{target_object} (red)"),
            _label(donor_pil, f"donor source | {category} #{donor_object} (blue)"),
            _label(_to_pil(decoded["full"]), f"full | loss {losses['full']:.4f}"),
            _label(_to_pil(decoded["object_only"]), f"object memory only | {losses['object_only']:.4f}"),
            _label(_to_pil(decoded["register_only"]), f"register memory only | {losses['register_only']:.4f}"),
            _label(_to_pil(decoded["all_zero"]), f"all memory zero | {losses['all_zero']:.4f}"),
            _label(_to_pil(decoded["selected_zero"]), f"selected {category} memory zero | {losses['selected_zero']:.4f}"),
            _label(_to_pil(decoded["same_category_swap"]), f"same-category memory swap | {losses['same_category_swap']:.4f}"),
            _label(_to_pil(difference), "|swap - full| x4"),
            _label(
                _to_pil(decoded["same_category_bundle_swap"]),
                f"semantic + memory bundle swap | {losses['same_category_bundle_swap']:.4f}",
            ),
            _label(_to_pil(bundle_difference), "|bundle swap - full| x4"),
            _label(_to_pil(semantic_effect), "|bundle swap - memory swap| x4"),
        ]
        output_path = output_dir / (
            f"sample{target_index}_obj{target_object}_{category.replace(' ', '_')}_memory_grid.png"
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
                "fixed_noise_seed": args.seed + pair_index,
                "guidance_scale": args.guidance_scale,
                "diffusion_inference_steps": args.diffusion_inference_steps,
                "losses": losses,
                "grid": str(output_path),
            }
        )

    summary = {
        "model_path": args.model_path,
        "protocol": (
            "same semantic keys/registers/noise; memory-only global ablations and "
            "one-object same-category memory-only and semantic+memory bundle swaps"
        ),
        "records": records,
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
    print(json.dumps(run(parser.parse_args()), indent=2))


if __name__ == "__main__":
    main()
