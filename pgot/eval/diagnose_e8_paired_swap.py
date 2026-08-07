"""Paired zero/same-category-swap causal diagnostic for PGOT E8 memory.

The semantic key and every other slot stay fixed.  Only one object's visual
memory is zeroed or replaced by a memory from the same category in a previous
image.  All three DiT branches share the exact RF timestep and noise.
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
from tqdm import tqdm

from pgot.eval.eval_recon_oracles import build_loader, load_model_and_tokenizer
from pgot.eval.pgot_inference import pgot_forward_eval


log = logging.getLogger("pgot.diagnose_e8_paired_swap")


def _reader_condition(model, out, memory: torch.Tensor) -> torch.Tensor:
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
        semantic_slots=out["semantic_slots"],
        visual_memory=memory,
        slot_valid=slot_valid,
    )["condition_hidden"]


def _object_categories(batch: dict, k_objects: int, n_ovt: int, device) -> torch.Tensor:
    raw = batch["ovt_category_ids"].to(device=device, dtype=torch.long)
    raw = raw[:, : k_objects * n_ovt]
    return raw.reshape(raw.shape[0], k_objects, n_ovt)[:, :, 0]


def _object_masks(batch: dict, k_objects: int, target_count: int, device) -> torch.Tensor:
    masks = batch["gt_masks_per_ovt"].to(device=device, dtype=torch.float32)
    masks = masks[:, :k_objects]
    source_side = int(round(math.sqrt(masks.shape[-1])))
    target_side = int(round(math.sqrt(target_count)))
    if source_side * source_side != masks.shape[-1] or target_side * target_side != target_count:
        raise ValueError("Paired E8 diagnostic requires square source/target grids")
    return F.interpolate(
        masks.reshape(-1, 1, source_side, source_side),
        size=(target_side, target_side),
        mode="area",
    ).reshape(masks.shape[0], k_objects, target_side, target_side)


def _area_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weighted = values * mask[:, None]
    denom = mask.sum(dim=(1, 2)) * values.shape[1]
    return weighted.sum(dim=(1, 2, 3)) / denom.clamp_min(1.0)


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


@torch.no_grad()
def run(args) -> None:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    model, tokenizer, device, _ = load_model_and_tokenizer(args)
    if not bool(getattr(model.config, "pgot_e8_visual_memory_enable", False)):
        raise ValueError("This diagnostic requires an E8 checkpoint")
    loader = build_loader(
        args,
        tokenizer,
        model,
        args.val_jsonl,
        shuffle=False,
        max_samples=args.max_samples,
    )

    # Previous-image memories only: the donor can never come from the source image.
    donor_bank: dict[int, list[torch.Tensor]] = defaultdict(list)
    rows: dict[str, list[float]] = defaultdict(list)
    images_seen = 0
    images_with_swap = 0

    for batch in tqdm(loader, desc="E8 paired same-category swap"):
        out = pgot_forward_eval(
            model,
            images=batch["images"],
            target_images=batch["target_images"],
            caption_input_ids=batch["caption_input_ids"],
            caption_attention_mask=batch["caption_attention_mask"],
            ovt_positions_in_caption=batch["ovt_positions_in_caption"],
            ovt_valid_mask=batch["ovt_valid_mask"],
        )
        memory = out["visual_memory"].float()
        valid = out["ovt_object_valid"].bool()
        batch_size, k_objects = valid.shape
        categories = _object_categories(
            batch, k_objects, int(model.pgot_n_ovt_per_object), device
        )
        masks = _object_masks(batch, k_objects, out["gt_siglip"].shape[1], device)

        selected = torch.zeros(batch_size, dtype=torch.long, device=device)
        selected_valid = torch.zeros(batch_size, dtype=torch.bool, device=device)
        swap_memory = memory.clone()
        zero_memory = memory.clone()
        for b in range(batch_size):
            candidates = [
                int(k)
                for k in torch.nonzero(valid[b], as_tuple=False).flatten().tolist()
                if int(categories[b, int(k)]) >= 0
                and donor_bank.get(int(categories[b, int(k)]))
            ]
            if not candidates:
                continue
            k = candidates[random.randrange(len(candidates))]
            cat = int(categories[b, k])
            donor = donor_bank[cat][random.randrange(len(donor_bank[cat]))]
            selected[b] = k
            selected_valid[b] = True
            swap_memory[b, k] = donor.to(device=device, dtype=memory.dtype)
            zero_memory[b, k].zero_()

        if bool(selected_valid.any()):
            batch_indices = torch.arange(batch_size, device=device)
            selected_masks = masks[batch_indices, selected]
            selected_masks = selected_masks * selected_valid[:, None, None].float()

            full_condition = model._captionslot_prepare_diffusion_condition(
                out["rae_hidden"]
            ).float()
            zero_condition = model._captionslot_prepare_diffusion_condition(
                _reader_condition(model, out, zero_memory)
            ).float()
            swap_condition = model._captionslot_prepare_diffusion_condition(
                _reader_condition(model, out, swap_memory)
            ).float()

            target = out["gt_siglip"].to(device=device, dtype=torch.float32)
            if getattr(model.diff_head, "normalize_data", False):
                mean = model.diff_head.data_mean.to(device)
                std = model.diff_head.data_std.to(device)
                while mean.dim() < target.dim():
                    mean = mean.unsqueeze(0)
                    std = std.unsqueeze(0)
                target = (target - mean) / std
            else:
                target = F.layer_norm(target, (target.shape[-1],))

            side = int(round(math.sqrt(target.shape[1])))
            target_grid = target.reshape(batch_size, side, side, -1).permute(0, 3, 1, 2)
            flow = model.diff_head.train_flow
            timestep = flow.get_timestep(target_grid)
            x_end = flow.get_x_end(target_grid.shape, target_grid.device)
            alpha = flow.get_alphas(timestep).view(batch_size, 1, 1, 1)
            sigma = flow.get_sigmas(timestep).view(batch_size, 1, 1, 1)
            x_t = alpha * target_grid + sigma * x_end

            terms = flow.training_losses(
                model.diff_head.model,
                target_grid.repeat(3, 1, 1, 1),
                timestep.repeat(3),
                model_kwargs={
                    "y": torch.cat([full_condition, zero_condition, swap_condition], dim=0)
                },
                x_end=x_end.repeat(3, 1, 1, 1),
                x_t=x_t.repeat(3, 1, 1, 1),
            )
            mse_full, mse_zero, mse_swap = terms["mse_map"].chunk(3, dim=0)
            pred_full, pred_zero, pred_swap = terms["model_pred"].chunk(3, dim=0)
            outside = (1.0 - selected_masks).clamp(0.0, 1.0)

            full_inside = _area_mean(mse_full, selected_masks)
            zero_inside = _area_mean(mse_zero, selected_masks)
            swap_inside = _area_mean(mse_swap, selected_masks)
            zero_delta_in = _area_mean((pred_zero - pred_full).square(), selected_masks)
            zero_delta_out = _area_mean((pred_zero - pred_full).square(), outside)
            swap_delta_in = _area_mean((pred_swap - pred_full).square(), selected_masks)
            swap_delta_out = _area_mean((pred_swap - pred_full).square(), outside)

            for b in torch.nonzero(selected_valid, as_tuple=False).flatten().tolist():
                base = max(float(full_inside[b]), 1e-8)
                rows["full_selected_mse"].append(float(full_inside[b]))
                rows["zero_selected_mse"].append(float(zero_inside[b]))
                rows["swap_selected_mse"].append(float(swap_inside[b]))
                rows["zero_selected_error_ratio"].append(float(zero_inside[b]) / base)
                rows["swap_selected_error_ratio"].append(float(swap_inside[b]) / base)
                rows["zero_selected_error_delta"].append(float(zero_inside[b] - full_inside[b]))
                rows["swap_selected_error_delta"].append(float(swap_inside[b] - full_inside[b]))
                rows["zero_prediction_delta_inside"].append(float(zero_delta_in[b]))
                rows["zero_prediction_delta_outside"].append(float(zero_delta_out[b]))
                rows["swap_prediction_delta_inside"].append(float(swap_delta_in[b]))
                rows["swap_prediction_delta_outside"].append(float(swap_delta_out[b]))
                rows["zero_delta_localization_ratio"].append(
                    float(zero_delta_in[b] / zero_delta_out[b].clamp_min(1e-8))
                )
                rows["swap_delta_localization_ratio"].append(
                    float(swap_delta_in[b] / swap_delta_out[b].clamp_min(1e-8))
                )
            images_with_swap += int(selected_valid.sum())

        # Add current-image memories only after interventions are selected.
        for b in range(batch_size):
            for k in torch.nonzero(valid[b], as_tuple=False).flatten().tolist():
                cat = int(categories[b, int(k)])
                if cat >= 0:
                    donor_bank[cat].append(memory[b, int(k)].detach().cpu())
                    if len(donor_bank[cat]) > args.max_donors_per_category:
                        donor_bank[cat].pop(0)
        images_seen += batch_size

    summary = {
        "model_path": args.model_path,
        "num_images_seen": images_seen,
        "num_images_with_same_category_swap": images_with_swap,
        "paired_protocol": "same semantic keys/registers/timestep/noise; one visual memory changed",
        "metrics": {key: _summary(values) for key, values in sorted(rows.items())},
    }
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(json.dumps(summary, indent=2))
    log.info("Wrote %s", output / "summary.json")
    print(json.dumps(summary, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--val_jsonl", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--max_samples", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--grid_size", type=int, default=32)
    parser.add_argument("--max_caption_tokens", type=int, default=1024)
    parser.add_argument("--n_ovt_per_object", type=int, default=1)
    parser.add_argument("--max_objects", type=int, default=50)
    parser.add_argument("--image_preprocess_mode", default="coda_center_crop")
    parser.add_argument("--coda_crop_size", type=int, default=512)
    parser.add_argument("--dtype", choices=["fp32", "bf16"], default="fp32")
    parser.add_argument("--diffusion_inference_steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--max_donors_per_category", type=int, default=32)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
