"""Focused checkpoint-only follow-up diagnostics for E11 Dual-M4.

This module answers four questions in one model/data pass:

1. How much reconstruction quality remains when only the top 1/2/3/4
   memories per semantic owner are kept?
2. Do the four memories under one owner attend to different spatial regions?
3. Does a same-category memory swap move the generated object toward the
   donor appearance, rather than merely perturbing the prediction?
4. Why does the register-only branch still produce recognizable objects?

No checkpoint weights are changed.  All interventions are inference-only.
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
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

from pgot.eval.eval_recon_oracles import build_loader, load_model_and_tokenizer
from pgot.eval.pgot_inference import generate_siglip_latent, pgot_forward_eval


log = logging.getLogger("pgot.diagnose_e11_followup")


def _font(size: int = 14) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except Exception:
        return ImageFont.load_default()


def _label(image: Image.Image, title: str, height: int = 34) -> Image.Image:
    image = image.convert("RGB")
    canvas = Image.new("RGB", (image.width, image.height + height), "white")
    canvas.paste(image, (0, height))
    ImageDraw.Draw(canvas).text((6, 6), title, fill="black", font=_font())
    return canvas


def _grid(images: list[Image.Image], columns: int = 3) -> Image.Image:
    width = max(image.width for image in images)
    height = max(image.height for image in images)
    rows = math.ceil(len(images) / columns)
    canvas = Image.new("RGB", (columns * width, rows * height), (245, 245, 245))
    for index, image in enumerate(images):
        if image.size != (width, height):
            image = image.resize((width, height), Image.Resampling.BILINEAR)
        canvas.paste(image, ((index % columns) * width, (index // columns) * height))
    return canvas


def _source_image(tensor: torch.Tensor, processor) -> Image.Image:
    mean = torch.tensor(processor.image_mean).view(-1, 1, 1)
    std = torch.tensor(processor.image_std).view(-1, 1, 1)
    image = (tensor.detach().cpu().float() * std + mean).clamp(0, 1)
    array = (image.permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
    return Image.fromarray(array, mode="RGB")


def _heat_overlay(source: Image.Image, flat_heat: torch.Tensor) -> Image.Image:
    side = int(round(math.sqrt(flat_heat.numel())))
    if side * side != flat_heat.numel():
        raise ValueError("Spatial visualization expects a square Writer grid")
    heat = flat_heat.detach().cpu().float().reshape(side, side)
    heat = heat - heat.min()
    heat = heat / heat.max().clamp_min(1e-6)
    heat_image = Image.fromarray(
        (heat.numpy().clip(0, 1) * 255).astype(np.uint8), mode="L"
    ).resize(source.size, Image.Resampling.BILINEAR)
    src = np.asarray(source.convert("RGB"), dtype=np.float32)
    h = np.asarray(heat_image, dtype=np.float32) / 255.0
    color = np.zeros_like(src)
    color[..., 0] = 255.0
    color[..., 1] = 55.0
    color[..., 2] = 25.0
    out = src * (1.0 - 0.68 * h[..., None]) + color * (0.68 * h[..., None])
    return Image.fromarray(out.clip(0, 255).astype(np.uint8), mode="RGB")


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


def _object_categories(batch: dict, k_objects: int, n_ovt: int, device) -> torch.Tensor:
    raw = batch["ovt_category_ids"].to(device=device, dtype=torch.long)
    raw = raw[:, : k_objects * n_ovt]
    return raw.reshape(raw.shape[0], k_objects, n_ovt)[:, :, 0]


def _object_masks(
    batch: dict,
    k_objects: int,
    target_count: int,
    device: torch.device,
) -> torch.Tensor:
    masks = batch["gt_masks_per_ovt"][:, :k_objects].to(
        device=device, dtype=torch.float32
    )
    source_side = int(round(math.sqrt(masks.shape[-1])))
    target_side = int(round(math.sqrt(target_count)))
    if source_side * source_side != masks.shape[-1] or target_side * target_side != target_count:
        raise ValueError("E11 follow-up analysis expects square mask/latent grids")
    return F.interpolate(
        masks.reshape(-1, 1, source_side, source_side),
        size=(target_side, target_side),
        mode="area",
    ).reshape(masks.shape[0], k_objects, target_count)


def _reader_output(
    model,
    out: dict,
    memory: torch.Tensor,
    *,
    semantic_slots: torch.Tensor | None = None,
    object_keys_valid: bool = True,
) -> dict:
    object_valid = out["ovt_object_valid"].bool()
    batch_size, k_objects = object_valid.shape
    n_register = memory.shape[1] - k_objects
    if semantic_slots is None:
        semantic_slots = out["semantic_slots"]
    slot_valid = torch.cat(
        [
            object_valid if object_keys_valid else torch.zeros_like(object_valid),
            torch.ones(
                batch_size,
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
    )


def _top_n_memory(memory: torch.Tensor, utilization: torch.Tensor, n: int) -> torch.Tensor:
    if memory.ndim != 4:
        raise ValueError("Memory-count ablation requires [B,S,J,D] Dual-M4 memory")
    n = min(max(int(n), 1), memory.shape[2])
    top = utilization.float().topk(n, dim=-1).indices
    keep = torch.zeros_like(utilization, dtype=torch.bool)
    keep.scatter_(-1, top, True)
    return memory * keep.unsqueeze(-1).to(memory.dtype)


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    # values: [B,C,H,W], mask: [B,H,W]
    weighted = values * mask[:, None]
    denominator = mask.sum(dim=(1, 2)) * values.shape[1]
    return weighted.sum(dim=(1, 2, 3)) / denominator.clamp_min(1.0)


def _branch_flow_metrics(
    model,
    conditions: dict[str, torch.Tensor],
    target: torch.Tensor,
    foreground: torch.Tensor,
    *,
    seed: int,
    chunk_size: int,
) -> dict[str, dict[str, torch.Tensor]]:
    """Evaluate all conditions with one shared RF timestep/noise realization."""
    prepared = {
        name: model._captionslot_prepare_diffusion_condition(condition).float()
        for name, condition in conditions.items()
    }
    batch_size = target.shape[0]
    if getattr(model.diff_head, "normalize_data", False):
        mean = model.diff_head.data_mean.to(target.device)
        std = model.diff_head.data_std.to(target.device)
        while mean.dim() < target.dim():
            mean = mean.unsqueeze(0)
            std = std.unsqueeze(0)
        target = (target - mean) / std
    else:
        target = F.layer_norm(target, (target.shape[-1],))

    side = int(round(math.sqrt(target.shape[1])))
    target_grid = target.reshape(batch_size, side, side, -1).permute(0, 3, 1, 2)
    foreground_grid = foreground.reshape(batch_size, side, side).clamp(0, 1)
    background_grid = (1.0 - foreground_grid).clamp(0, 1)
    flow = model.diff_head.train_flow
    devices = [target.device.index] if target.is_cuda else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(int(seed))
        timestep = flow.get_timestep(target_grid)
        x_end = flow.get_x_end(target_grid.shape, target_grid.device)
    alpha = flow.get_alphas(timestep).view(batch_size, 1, 1, 1)
    sigma = flow.get_sigmas(timestep).view(batch_size, 1, 1, 1)
    x_t = alpha * target_grid + sigma * x_end

    branch_outputs: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    names = list(prepared)
    for start in range(0, len(names), max(int(chunk_size), 1)):
        chunk_names = names[start : start + max(int(chunk_size), 1)]
        count = len(chunk_names)
        terms = flow.training_losses(
            model.diff_head.model,
            target_grid.repeat(count, 1, 1, 1),
            timestep.repeat(count),
            model_kwargs={"y": torch.cat([prepared[name] for name in chunk_names], dim=0)},
            x_end=x_end.repeat(count, 1, 1, 1),
            x_t=x_t.repeat(count, 1, 1, 1),
        )
        mse_chunks = terms["mse_map"].chunk(count, dim=0)
        pred_chunks = terms["model_pred"].chunk(count, dim=0)
        branch_outputs.update(
            {
                name: (mse, pred)
                for name, mse, pred in zip(chunk_names, mse_chunks, pred_chunks)
            }
        )

    full_pred = branch_outputs["full"][1]
    metrics = {}
    for name, (mse, prediction) in branch_outputs.items():
        pred_delta = (prediction - full_pred).square()
        metrics[name] = {
            "mse": mse.mean(dim=(1, 2, 3)),
            "foreground_mse": _masked_mean(mse, foreground_grid),
            "background_mse": _masked_mean(mse, background_grid),
            "prediction_delta_foreground": _masked_mean(pred_delta, foreground_grid),
            "prediction_delta_background": _masked_mean(pred_delta, background_grid),
        }
    return metrics


def _pool_features(features: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
    denominator = masks.sum(dim=-1, keepdim=True).clamp_min(1e-6)
    return torch.einsum("bp,bpd->bd", masks, features.float()) / denominator


def _record_spatial_statistics(
    record: dict,
    valid: torch.Tensor,
    k_objects: int,
    rows: dict[str, list[float]],
    centroids_by_kind: dict[str, dict[int, list[list[float]]]],
) -> None:
    owner = record["owner_probs"].float()
    memory_probs = record["memory_probs"].float()
    if memory_probs.ndim != 4:
        raise ValueError("E11 spatial analysis requires [B,S,J,P] memory probabilities")
    _, slots, memories, patches = memory_probs.shape
    side = int(round(math.sqrt(patches)))
    if side * side != patches:
        raise ValueError("Writer memory probabilities must use a square grid")
    axis = torch.linspace(-1.0, 1.0, side, device=owner.device)
    yy, xx = torch.meshgrid(axis, axis, indexing="ij")
    coordinates = torch.stack([xx.flatten(), yy.flatten()], dim=-1)
    joint = owner.unsqueeze(2) * memory_probs
    write_weights = joint / joint.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    centroids = torch.einsum("bsjp,pd->bsjd", write_weights, coordinates)
    utilization = joint.sum(dim=-1) / owner.sum(dim=-1, keepdim=True).clamp_min(1e-8)

    for b in range(owner.shape[0]):
        for slot in range(slots):
            if slot < k_objects and not bool(valid[b, slot]):
                continue
            kind = "object" if slot < k_objects else "register"
            weights = write_weights[b, slot]
            normed = F.normalize(weights, dim=-1)
            cosine = normed @ normed.T
            triangle = torch.triu_indices(memories, memories, offset=1, device=owner.device)
            rows[f"{kind}_write_weight_pair_cosine"].extend(
                cosine[triangle[0], triangle[1]].cpu().tolist()
            )
            distance = torch.cdist(centroids[b, slot], centroids[b, slot])
            rows[f"{kind}_centroid_pair_distance"].extend(
                distance[triangle[0], triangle[1]].cpu().tolist()
            )
            entropy = -(utilization[b, slot] * utilization[b, slot].clamp_min(1e-8).log()).sum()
            entropy = entropy / math.log(float(memories))
            rows[f"{kind}_utilization_entropy"].append(float(entropy))
            for memory_id in range(memories):
                centroids_by_kind[kind][memory_id].append(
                    centroids[b, slot, memory_id].cpu().tolist()
                )


def _save_assignment_visuals(
    *,
    batch: dict,
    record: dict,
    valid: torch.Tensor,
    k_objects: int,
    target_processor,
    output_dir: Path,
    start_index: int,
    remaining: int,
) -> tuple[list[str], int]:
    if remaining <= 0:
        return [], start_index
    owner = record["owner_probs"].float()
    memory_probs = record["memory_probs"].float()
    utilization = record.get("memory_utilization")
    paths = []
    output_dir.mkdir(parents=True, exist_ok=True)
    for b in range(owner.shape[0]):
        if len(paths) >= remaining:
            break
        object_indices = torch.nonzero(valid[b], as_tuple=False).flatten().tolist()
        candidate_slots = []
        if object_indices:
            candidate_slots.append(("object", int(object_indices[0])))
        if owner.shape[1] > k_objects:
            register_mass = owner[b, k_objects:].sum(dim=-1)
            register_id = int(register_mass.argmax())
            candidate_slots.append(("register", k_objects + register_id))
        source = _source_image(batch["target_images"][b], target_processor)
        for kind, slot in candidate_slots:
            if len(paths) >= remaining:
                break
            tiles = [_label(source, f"source | {kind} owner {slot}")]
            tiles.append(_label(_heat_overlay(source, owner[b, slot]), "owner mass"))
            for memory_id in range(memory_probs.shape[2]):
                joint = owner[b, slot] * memory_probs[b, slot, memory_id]
                use = (
                    float(utilization[b, slot, memory_id])
                    if utilization is not None
                    else float(joint.sum() / owner[b, slot].sum().clamp_min(1e-8))
                )
                tiles.append(
                    _label(
                        _heat_overlay(source, joint),
                        f"memory {memory_id} | use={use:.3f}",
                    )
                )
            filename = f"{start_index:04d}_{kind}_owner{slot}.png"
            path = output_dir / filename
            _grid(tiles, columns=3).save(path)
            paths.append(str(path))
            start_index += 1
    return paths, start_index


@torch.no_grad()
def run(args: argparse.Namespace) -> dict:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    model, tokenizer, device, _ = load_model_and_tokenizer(args)
    if not bool(getattr(model.config, "pgot_e11_dual_m4_enable", False)):
        raise ValueError("This diagnostic requires an E11 Dual-M4 checkpoint")

    loader = build_loader(
        args,
        tokenizer,
        model,
        args.val_jsonl,
        shuffle=False,
        max_samples=args.max_samples,
    )
    target_processor = model.get_vision_tower_aux_list()[-1].image_processor
    output_dir = Path(args.output_dir)
    visual_dir = output_dir / "spatial_assignments"
    metric_rows: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    spatial_rows: dict[str, list[float]] = defaultdict(list)
    centroids_by_kind: dict[str, dict[int, list[list[float]]]] = {
        "object": defaultdict(list),
        "register": defaultdict(list),
    }
    transfer_rows: dict[str, list[float]] = defaultdict(list)
    donor_bank: dict[int, list[dict]] = defaultdict(list)
    previous_registers: torch.Tensor | None = None
    visual_paths: list[str] = []
    visual_index = 0
    samples_seen = 0

    for batch_index, batch in enumerate(tqdm(loader, desc="E11 follow-up")):
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
        if memory.ndim != 4 or memory.shape[2] <= 1:
            raise ValueError(f"Expected Dual-M4 memory, got {tuple(memory.shape)}")
        valid = out["ovt_object_valid"].bool()
        batch_size, k_objects = valid.shape
        register_count = memory.shape[1] - k_objects
        final_record = out["e8_write_records"][-1]
        utilization = final_record["memory_utilization"].float()
        masks = _object_masks(batch, k_objects, out["gt_siglip"].shape[1], device)
        foreground = masks.amax(dim=1)

        conditions: dict[str, torch.Tensor] = {"full": out["rae_hidden"].float()}
        for count in range(1, memory.shape[2] + 1):
            top_memory = _top_n_memory(memory, utilization, count)
            conditions[f"keep_top_{count}_all_owners"] = _reader_output(
                model, out, top_memory
            )["condition_hidden"].float()

            object_limited = memory.clone()
            object_limited[:, :k_objects] = top_memory[:, :k_objects]
            conditions[f"keep_top_{count}_object_owners"] = _reader_output(
                model, out, object_limited
            )["condition_hidden"].float()

            register_limited = memory.clone()
            register_limited[:, k_objects:] = top_memory[:, k_objects:]
            conditions[f"keep_top_{count}_register_owners"] = _reader_output(
                model, out, register_limited
            )["condition_hidden"].float()

        register_only = memory.clone()
        register_only[:, :k_objects] = 0
        conditions["register_only_keys_present"] = _reader_output(
            model, out, register_only
        )["condition_hidden"].float()

        null_semantic = out["semantic_slots"].clone()
        null_semantic[:, :k_objects] = 0
        conditions["register_only_null_object_keys"] = _reader_output(
            model, out, register_only, semantic_slots=null_semantic
        )["condition_hidden"].float()
        conditions["pure_register_only_object_keys_masked"] = _reader_output(
            model, out, register_only, object_keys_valid=False
        )["condition_hidden"].float()
        conditions["all_memory_zero"] = _reader_output(
            model, out, torch.zeros_like(memory)
        )["condition_hidden"].float()

        if previous_registers is not None and previous_registers.shape[1:] == memory[:, k_objects:].shape[1:]:
            swapped_registers = memory.clone()
            donor = previous_registers.to(device=device, dtype=memory.dtype)
            if donor.shape[0] < batch_size:
                repeats = math.ceil(batch_size / donor.shape[0])
                donor = donor.repeat(repeats, 1, 1, 1)
            swapped_registers[:, k_objects:] = donor[:batch_size]
            conditions["full_with_registers_swapped"] = _reader_output(
                model, out, swapped_registers
            )["condition_hidden"].float()

        branch_metrics = _branch_flow_metrics(
            model,
            conditions,
            out["gt_siglip"].float(),
            foreground,
            seed=args.seed + batch_index,
            chunk_size=args.branch_chunk_size,
        )
        full_batch_metrics = branch_metrics["full"]
        for branch, metrics in branch_metrics.items():
            for metric, values in metrics.items():
                metric_rows[branch][metric].extend(values.detach().cpu().tolist())
            metric_rows[branch]["mse_delta_vs_full"].extend(
                (metrics["mse"] - full_batch_metrics["mse"]).detach().cpu().tolist()
            )
            metric_rows[branch]["foreground_mse_delta_vs_full"].extend(
                (
                    metrics["foreground_mse"]
                    - full_batch_metrics["foreground_mse"]
                ).detach().cpu().tolist()
            )
            metric_rows[branch]["background_mse_delta_vs_full"].extend(
                (
                    metrics["background_mse"]
                    - full_batch_metrics["background_mse"]
                ).detach().cpu().tolist()
            )

        _record_spatial_statistics(
            final_record,
            valid,
            k_objects,
            spatial_rows,
            centroids_by_kind,
        )
        if len(visual_paths) < args.max_visualizations:
            new_paths, visual_index = _save_assignment_visuals(
                batch=batch,
                record=final_record,
                valid=valid,
                k_objects=k_objects,
                target_processor=target_processor,
                output_dir=visual_dir,
                start_index=visual_index,
                remaining=args.max_visualizations - len(visual_paths),
            )
            visual_paths.extend(new_paths)

        categories = _object_categories(
            batch, k_objects, int(model.pgot_n_ovt_per_object), device
        )
        transfer_masks = masks
        transfer_candidates: list[tuple[int, int, dict]] = []
        if len(transfer_rows["donor_similarity_gain"]) < args.transfer_max_pairs:
            for b in range(batch_size):
                candidates = [
                    int(k)
                    for k in torch.nonzero(valid[b], as_tuple=False).flatten().tolist()
                    if int(categories[b, int(k)]) >= 0
                    and donor_bank.get(int(categories[b, int(k)]))
                ]
                if candidates:
                    k = candidates[random.randrange(len(candidates))]
                    cat = int(categories[b, k])
                    donor = donor_bank[cat][random.randrange(len(donor_bank[cat]))]
                    transfer_candidates.append((b, k, donor))
            remaining = args.transfer_max_pairs - len(transfer_rows["donor_similarity_gain"])
            transfer_candidates = transfer_candidates[:remaining]

        if transfer_candidates:
            swap_memory = memory.clone()
            selected_batch = []
            selected_masks = []
            donor_features = []
            for b, k, donor in transfer_candidates:
                swap_memory[b, k] = donor["memory"].to(device=device, dtype=memory.dtype)
                selected_batch.append(b)
                selected_masks.append(transfer_masks[b, k])
                donor_features.append(donor["target_feature"])
            selected = torch.tensor(selected_batch, device=device, dtype=torch.long)
            selected_mask = torch.stack(selected_masks)
            donor_feature = torch.stack(donor_features).to(device=device, dtype=torch.float32)
            full_condition = out["rae_hidden"].float()[selected]
            swap_condition = _reader_output(model, out, swap_memory)["condition_hidden"].float()[selected]
            generation_seed = args.seed + 100000 + batch_index
            torch.manual_seed(generation_seed)
            full_generated = generate_siglip_latent(
                model, full_condition, guidance_level=args.guidance_scale
            ).float()
            torch.manual_seed(generation_seed)
            swap_generated = generate_siglip_latent(
                model, swap_condition, guidance_level=args.guidance_scale
            ).float()
            full_feature = _pool_features(full_generated, selected_mask)
            swap_feature = _pool_features(swap_generated, selected_mask)
            target_feature = _pool_features(
                out["gt_siglip"].float()[selected], selected_mask
            )
            full_n = F.normalize(full_feature, dim=-1)
            swap_n = F.normalize(swap_feature, dim=-1)
            target_n = F.normalize(target_feature, dim=-1)
            donor_n = F.normalize(donor_feature, dim=-1)
            full_donor = (full_n * donor_n).sum(dim=-1)
            swap_donor = (swap_n * donor_n).sum(dim=-1)
            full_target = (full_n * target_n).sum(dim=-1)
            swap_target = (swap_n * target_n).sum(dim=-1)
            direction = F.cosine_similarity(
                swap_feature - full_feature,
                donor_feature - target_feature,
                dim=-1,
            )
            transfer_rows["full_to_donor_cosine"].extend(full_donor.cpu().tolist())
            transfer_rows["swap_to_donor_cosine"].extend(swap_donor.cpu().tolist())
            transfer_rows["donor_similarity_gain"].extend((swap_donor - full_donor).cpu().tolist())
            transfer_rows["full_to_target_cosine"].extend(full_target.cpu().tolist())
            transfer_rows["swap_to_target_cosine"].extend(swap_target.cpu().tolist())
            transfer_rows["target_similarity_change"].extend((swap_target - full_target).cpu().tolist())
            transfer_rows["donor_direction_alignment"].extend(direction.cpu().tolist())
            transfer_rows["donor_closer"].extend((swap_donor > full_donor).float().cpu().tolist())

        # Populate the donor bank only after choosing donors for this batch.
        gt_features = out["gt_siglip"].float()
        for b in range(batch_size):
            for k in torch.nonzero(valid[b], as_tuple=False).flatten().tolist():
                cat = int(categories[b, int(k)])
                if cat < 0:
                    continue
                target_feature = _pool_features(
                    gt_features[b : b + 1], transfer_masks[b : b + 1, int(k)]
                )[0]
                donor_bank[cat].append(
                    {
                        "memory": memory[b, int(k)].detach().cpu(),
                        "target_feature": target_feature.detach().cpu(),
                    }
                )
                if len(donor_bank[cat]) > args.max_donors_per_category:
                    donor_bank[cat].pop(0)

        previous_registers = memory[:, k_objects:].detach().cpu()
        samples_seen += batch_size

    branch_summary = {
        branch: {metric: _summary(values) for metric, values in metrics.items()}
        for branch, metrics in metric_rows.items()
    }

    centroid_summary = {}
    for kind, by_memory in centroids_by_kind.items():
        centroid_summary[kind] = {}
        for memory_id, values in sorted(by_memory.items()):
            array = np.asarray(values, dtype=np.float64)
            centroid_summary[kind][str(memory_id)] = {
                "mean_x": float(array[:, 0].mean()),
                "mean_y": float(array[:, 1].mean()),
                "std_x": float(array[:, 0].std()),
                "std_y": float(array[:, 1].std()),
                "count": int(array.shape[0]),
            }

    memory_count_names = [
        name for name in branch_summary if name.startswith("keep_top_")
    ]
    register_names = [
        "full",
        "register_only_keys_present",
        "register_only_null_object_keys",
        "pure_register_only_object_keys_masked",
        "all_memory_zero",
        "full_with_registers_swapped",
    ]
    summary = {
        "model_path": args.model_path,
        "num_samples": samples_seen,
        "protocol": {
            "memory_count": (
                "Top-N is selected independently for every semantic owner using "
                "the final Writer utilization; object-only and register-only "
                "capacity reductions are also reported."
            ),
            "spatial_assignment": (
                "Final Writer owner mass multiplied by within-owner memory softmax."
            ),
            "donor_transfer": (
                "Same-category previous-image donor; fixed diffusion noise; masked "
                "generated SigLIP feature compared with donor/target GT features."
            ),
            "register_decomposition": {
                "register_only_keys_present": "Object values zero; object semantic keys remain valid.",
                "register_only_null_object_keys": "Object values and semantic content zero; zero object slots remain valid.",
                "pure_register_only_object_keys_masked": "Object values zero and object slots removed from Reader attention.",
                "all_memory_zero": "All visual values zero while semantic keys remain.",
                "full_with_registers_swapped": "Target object memories with registers from a previous image.",
            },
        },
        "memory_count_ablation": {
            name: branch_summary[name] for name in sorted(memory_count_names)
        },
        "spatial_assignment": {
            "metrics": {key: _summary(values) for key, values in sorted(spatial_rows.items())},
            "centroids_by_memory_id": centroid_summary,
            "visualizations": visual_paths,
        },
        "donor_appearance_transfer": {
            key: _summary(values) for key, values in sorted(transfer_rows.items())
        },
        "register_decomposition": {
            name: branch_summary[name]
            for name in register_names
            if name in branch_summary
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    log.info("Wrote %s", summary_path)
    print(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--val_jsonl", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--max_samples", type=int, default=512)
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
    parser.add_argument("--branch_chunk_size", type=int, default=4)
    parser.add_argument("--transfer_max_pairs", type=int, default=64)
    parser.add_argument("--max_donors_per_category", type=int, default=32)
    parser.add_argument("--max_visualizations", type=int, default=12)
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s :: %(message)s",
    )
    run(args)


if __name__ == "__main__":
    main()
