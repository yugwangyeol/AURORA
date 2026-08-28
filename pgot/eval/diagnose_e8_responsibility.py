"""Priority diagnostics for E8 object-memory responsibility.

This checkpoint-only analysis separates three failure modes:

1. discovery vs named grounding at every Writer layer;
2. causal exclusivity of each object memory at the DiT prediction;
3. unique appearance information in self/other/register memories.

No weights are updated.  Every causal branch shares timestep and noise.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm

from pgot.eval.diagnose_e8_visual_memory import _reader_condition
from pgot.eval.eval_recon_oracles import build_loader, load_model_and_tokenizer
from pgot.eval.pgot_inference import pgot_forward_eval
from pgot.eval.probe_ovt_appearance import pool_with_mask, resize_mask, word_embeddings


log = logging.getLogger("pgot.diagnose_e8_responsibility")


def _summary(values: Iterable[float]) -> dict:
    values = list(values)
    if not values:
        return {"count": 0, "mean": None, "median": None, "std": None, "p10": None, "p90": None}
    x = np.asarray(values, dtype=np.float64)
    return {
        "count": int(x.size),
        "mean": float(x.mean()),
        "median": float(np.median(x)),
        "std": float(x.std()),
        "p10": float(np.quantile(x, 0.10)),
        "p90": float(np.quantile(x, 0.90)),
    }


def _area_mean_single(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Channel/area-normalized mean for values [C,H,W], mask [H,W]."""
    return (values * mask[None]).sum() / (mask.sum() * values.shape[0]).clamp_min(1.0)


def _resize_masks(masks: torch.Tensor, target_count: int) -> torch.Tensor:
    source_side = int(round(math.sqrt(masks.shape[-1])))
    target_side = int(round(math.sqrt(target_count)))
    if source_side * source_side != masks.shape[-1] or target_side * target_side != target_count:
        raise ValueError("E8 responsibility analysis requires square grids")
    return F.interpolate(
        masks.reshape(-1, 1, source_side, source_side).float(),
        size=(target_side, target_side),
        mode="area",
    ).reshape(masks.shape[0], masks.shape[1], target_side, target_side)


def _mean_or_zero(x: torch.Tensor, dim: int = 0) -> torch.Tensor:
    if x.shape[dim] == 0:
        shape = list(x.shape)
        del shape[dim]
        return x.new_zeros(shape)
    return x.mean(dim=dim)


def _layer_writer_metrics(
    record: dict,
    masks: torch.Tensor,
    valid: torch.Tensor,
) -> tuple[dict, list[dict]]:
    """Return aggregate named/oracle IoU and per-image rows for one Writer layer."""
    probs = record["owner_probs"].float()
    batch_size, _, patch_count = probs.shape
    object_count = valid.shape[1]
    masks_grid = _resize_masks(masks[:, :object_count], patch_count).flatten(2)
    hard_owner = probs.argmax(dim=1)
    rows = []
    named_ious, oracle_ious, oracle_gains = [], [], []
    soft_named_ious, target_inside, target_outside = [], [], []

    for b in range(batch_size):
        indices = torch.nonzero(valid[b], as_tuple=False).flatten().tolist()
        if not indices:
            continue
        gt = masks_grid[b, indices]
        pred = torch.stack([(hard_owner[b] == k).float() for k in indices], dim=0)
        inter = pred[:, None].mul(gt[None]).sum(dim=-1)
        union = pred[:, None].sum(dim=-1) + gt[None].sum(dim=-1) - inter
        iou = inter / union.clamp_min(1e-8)
        named = float(iou.diag().mean())
        rr, cc = linear_sum_assignment((-iou).detach().cpu().numpy())
        oracle = float(iou[rr, cc].mean())
        named_ious.append(named)
        oracle_ious.append(oracle)
        oracle_gains.append(oracle - named)

        for local_idx, k in enumerate(indices):
            p = probs[b, k]
            m = gt[local_idx]
            soft_inter = (p * m).sum()
            soft_union = p.sum() + m.sum() - soft_inter
            soft_named_ious.append(float(soft_inter / soft_union.clamp_min(1e-8)))
            target_inside.append(float((p * m).sum() / m.sum().clamp_min(1e-8)))
            outside = 1.0 - m
            target_outside.append(float((p * outside).sum() / outside.sum().clamp_min(1e-8)))
        rows.append(
            {
                "named_hard_iou": named,
                "oracle_hard_iou": oracle,
                "oracle_minus_named": oracle - named,
                "num_objects": len(indices),
            }
        )

    return {
        "named_hard_iou": _summary(named_ious),
        "oracle_hard_iou": _summary(oracle_ious),
        "oracle_minus_named": _summary(oracle_gains),
        "named_soft_iou_per_object": _summary(soft_named_ious),
        "target_probability_inside": _summary(target_inside),
        "target_probability_outside": _summary(target_outside),
    }, rows


def _writer_transition_metrics(
    previous: dict,
    current: dict,
    masks: torch.Tensor,
) -> dict:
    p0 = previous["owner_probs"].float()
    p1 = current["owner_probs"].float()
    if p0.shape != p1.shape:
        return {}
    hard0, hard1 = p0.argmax(dim=1), p1.argmax(dim=1)
    fg = _resize_masks(masks, p0.shape[-1]).amax(dim=1).flatten(1) > 0
    all_consistency = (hard0 == hard1).float().mean(dim=1)
    fg_values, bg_values = [], []
    for b in range(p0.shape[0]):
        if bool(fg[b].any()):
            fg_values.append(float((hard0[b, fg[b]] == hard1[b, fg[b]]).float().mean()))
        if bool((~fg[b]).any()):
            bg_values.append(float((hard0[b, ~fg[b]] == hard1[b, ~fg[b]]).float().mean()))
    return {
        "hard_assignment_consistency_all": _summary(all_consistency.cpu().tolist()),
        "hard_assignment_consistency_fg": _summary(fg_values),
        "hard_assignment_consistency_bg": _summary(bg_values),
        "probability_l1": float((p1 - p0).abs().mean()),
    }


def _normalize_target(model, target: torch.Tensor) -> torch.Tensor:
    target = target.float()
    if getattr(model.diff_head, "normalize_data", False):
        mean = model.diff_head.data_mean.to(target.device)
        std = model.diff_head.data_std.to(target.device)
        while mean.dim() < target.dim():
            mean = mean.unsqueeze(0)
            std = std.unsqueeze(0)
        return (target - mean) / std
    return F.layer_norm(target, (target.shape[-1],))


@torch.no_grad()
def _causal_influence(
    model,
    out: dict,
    masks: torch.Tensor,
    valid: torch.Tensor,
    categories: torch.Tensor,
    image_offset: int,
    seed: int,
    intervention_batch_size: int,
) -> list[dict]:
    """Zero each valid object memory and measure changes on every object and BG."""
    memory = out["visual_memory"].float()
    batch_size, object_count = valid.shape
    target = _normalize_target(model, out["gt_siglip"].to(memory.device))
    side = int(round(math.sqrt(target.shape[1])))
    target_grid = target.reshape(batch_size, side, side, -1).permute(0, 3, 1, 2).contiguous()
    masks_grid = _resize_masks(masks[:, :object_count], target.shape[1])
    foreground = masks_grid.amax(dim=1).clamp(0.0, 1.0)

    devices = [memory.device.index] if memory.is_cuda else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(int(seed))
        flow = model.diff_head.train_flow
        timestep = flow.get_timestep(target_grid)
        x_end = flow.get_x_end(target_grid.shape, target_grid.device)
    alpha = flow.get_alphas(timestep).view(batch_size, 1, 1, 1)
    sigma = flow.get_sigmas(timestep).view(batch_size, 1, 1, 1)
    x_t = alpha * target_grid + sigma * x_end
    full_condition = model._captionslot_prepare_diffusion_condition(out["rae_hidden"]).float()
    full_terms = flow.training_losses(
        model.diff_head.model,
        target_grid,
        timestep,
        model_kwargs={"y": full_condition},
        x_end=x_end,
        x_t=x_t,
    )
    full_pred = full_terms["model_pred"]

    sources = [
        (b, int(k))
        for b in range(batch_size)
        for k in torch.nonzero(valid[b], as_tuple=False).flatten().tolist()
    ]
    rows = []
    for start in range(0, len(sources), intervention_batch_size):
        chunk = sources[start : start + intervention_batch_size]
        source_b = torch.tensor([x[0] for x in chunk], device=memory.device, dtype=torch.long)
        source_k = torch.tensor([x[1] for x in chunk], device=memory.device, dtype=torch.long)
        ablated_memory = memory[source_b].clone()
        ablated_memory[torch.arange(len(chunk), device=memory.device), source_k] = 0.0
        row_out = {
            "ovt_object_valid": out["ovt_object_valid"][source_b],
            "raw_rae_hidden": out["raw_rae_hidden"][source_b],
            "semantic_slots": out["semantic_slots"][source_b],
            "memory_centroids": (
                out["memory_centroids"][source_b]
                if out.get("memory_centroids") is not None
                else None
            ),
        }
        condition = model._captionslot_prepare_diffusion_condition(
            _reader_condition(model, row_out, ablated_memory)
        ).float()
        terms = flow.training_losses(
            model.diff_head.model,
            target_grid[source_b],
            timestep[source_b],
            model_kwargs={"y": condition},
            x_end=x_end[source_b],
            x_t=x_t[source_b],
        )
        delta = (terms["model_pred"] - full_pred[source_b]).square()

        for row_idx, (b, k) in enumerate(chunk):
            target_indices = torch.nonzero(valid[b], as_tuple=False).flatten().tolist()
            object_values = [
                float(_area_mean_single(delta[row_idx], masks_grid[b, j]))
                for j in target_indices
            ]
            local_k = target_indices.index(k)
            diagonal = object_values[local_k]
            other_values = [v for j, v in enumerate(object_values) if j != local_k]
            background = float(
                _area_mean_single(delta[row_idx], (1.0 - foreground[b]).clamp(0.0, 1.0))
            )
            other_mean = float(np.mean(other_values)) if other_values else 0.0
            competitor_values = other_values + [background]
            dominant = diagonal >= max(competitor_values) if competitor_values else True
            rows.append(
                {
                    "image_index": image_offset + b,
                    "source_slot": k,
                    "source_category": int(categories[b, k]),
                    "source_mask_area": float(masks_grid[b, k].sum()),
                    "target_slots": target_indices,
                    "target_categories": [int(categories[b, j]) for j in target_indices],
                    "object_influence": object_values,
                    "background_influence": background,
                    "diagonal_influence": diagonal,
                    "other_object_mean_influence": other_mean,
                    "other_object_max_influence": max(other_values) if other_values else 0.0,
                    "diag_over_other_mean": diagonal / max(other_mean, 1e-12),
                    "diag_over_background": diagonal / max(background, 1e-12),
                    "offdiag_over_diag": other_mean / max(diagonal, 1e-12),
                    "background_over_diag": background / max(diagonal, 1e-12),
                    "selected_region_is_dominant": bool(dominant),
                    "num_objects": len(target_indices),
                }
            )
    return rows


def _solve_ridge(
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_eval: torch.Tensor,
    ridge_lambda: float,
) -> torch.Tensor:
    """Centered ridge prediction; switch between primal and dual forms."""
    n, d = x_train.shape
    if d <= n:
        gram = x_train.T @ x_train
        eye = torch.eye(d, device=x_train.device, dtype=x_train.dtype)
        weight = torch.linalg.solve(gram + ridge_lambda * eye, x_train.T @ y_train)
        return x_eval @ weight
    gram = x_train @ x_train.T
    eye = torch.eye(n, device=x_train.device, dtype=x_train.dtype)
    alpha = torch.linalg.solve(gram + ridge_lambda * eye, y_train)
    return (x_eval @ x_train.T) @ alpha


def _ridge_probe_split(
    x_cpu: torch.Tensor,
    y_cpu: torch.Tensor,
    train_mask: torch.Tensor,
    val_mask: torch.Tensor,
    test_mask: torch.Tensor,
    lambdas: list[float],
    device: torch.device,
    return_pred_all: bool = False,
) -> tuple[dict, torch.Tensor | None]:
    """Choose lambda on image-disjoint validation, refit on train+val, test once."""
    x = x_cpu.to(device=device, dtype=torch.float64)
    y = y_cpu.to(device=device, dtype=torch.float64)

    def standardize(train: torch.Tensor, eval_x: torch.Tensor):
        mean = train.mean(dim=0, keepdim=True)
        std = train.std(dim=0, unbiased=False, keepdim=True).clamp_min(1e-6)
        return (train - mean) / std, (eval_x - mean) / std

    x_tr, x_val = standardize(x[train_mask], x[val_mask])
    y_mean = y[train_mask].mean(dim=0, keepdim=True)
    y_tr = y[train_mask] - y_mean
    y_val = y[val_mask]
    denom_val = ((y_val - y_mean) ** 2).sum().clamp_min(1e-12)
    best_lambda, best_val_r2 = None, -float("inf")
    for ridge_lambda in lambdas:
        pred_val = _solve_ridge(x_tr, y_tr, x_val, ridge_lambda) + y_mean
        val_r2 = float(1.0 - ((y_val - pred_val) ** 2).sum() / denom_val)
        if val_r2 > best_val_r2:
            best_lambda, best_val_r2 = float(ridge_lambda), val_r2

    fit_mask = train_mask | val_mask
    x_fit, x_all = standardize(x[fit_mask], x)
    y_mean = y[fit_mask].mean(dim=0, keepdim=True)
    y_fit = y[fit_mask] - y_mean
    pred_all = _solve_ridge(x_fit, y_fit, x_all, best_lambda) + y_mean
    y_test, pred_test = y[test_mask], pred_all[test_mask]
    denom = ((y_test - y_mean) ** 2).sum().clamp_min(1e-12)
    r2 = float(1.0 - ((y_test - pred_test) ** 2).sum() / denom)
    cosine = float(F.cosine_similarity(pred_test, y_test, dim=-1).mean())
    rel_l2 = float(
        ((pred_test - y_test).norm(dim=-1) / y_test.norm(dim=-1).clamp_min(1e-8)).mean()
    )
    result = {
        "r2": r2,
        "cosine": cosine,
        "rel_l2": rel_l2,
        "lambda": best_lambda,
        "validation_r2": best_val_r2,
    }
    return result, pred_all.float().cpu() if return_pred_all else None


def _split_by_image(img_ids: torch.Tensor, seed: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
    unique = torch.unique(img_ids).cpu().numpy()
    rng = np.random.default_rng(seed)
    rng.shuffle(unique)
    n = len(unique)
    n_train = max(1, int(0.70 * n))
    n_val = max(1, int(0.10 * n))
    train_ids = set(unique[:n_train].tolist())
    val_ids = set(unique[n_train : n_train + n_val].tolist())
    test_ids = set(unique[n_train + n_val :].tolist())
    train = torch.tensor([int(x) in train_ids for x in img_ids.tolist()], dtype=torch.bool)
    val = torch.tensor([int(x) in val_ids for x in img_ids.tolist()], dtype=torch.bool)
    test = torch.tensor([int(x) in test_ids for x in img_ids.tolist()], dtype=torch.bool)
    return train, val, test, {
        "train_images": len(train_ids),
        "validation_images": len(val_ids),
        "test_images": len(test_ids),
        "train_objects": int(train.sum()),
        "validation_objects": int(val.sum()),
        "test_objects": int(test.sum()),
    }


def _run_appearance_probes(cache: Dict[str, list], args, device: torch.device) -> dict:
    tensors = {
        key: torch.stack(value).float() if key != "img_ids" else torch.tensor(value, dtype=torch.long)
        for key, value in cache.items()
    }
    tensors["direct_nonself"] = torch.cat(
        [tensors["direct_other_mean"], tensors["direct_register_flat"]], dim=-1
    )
    tensors["direct_all"] = torch.cat(
        [tensors["direct_self"], tensors["direct_nonself"]], dim=-1
    )
    tensors["routed_nonself"] = torch.cat(
        [tensors["routed_other"], tensors["routed_register"]], dim=-1
    )
    tensors["routed_all_concat"] = torch.cat(
        [tensors["routed_self"], tensors["routed_nonself"]], dim=-1
    )
    tensors["routed_sum"] = (
        tensors["routed_self"] + tensors["routed_other"] + tensors["routed_register"]
    )
    train, val, test, split_summary = _split_by_image(tensors["img_ids"], args.seed)
    lambdas = [float(x) for x in args.probe_lambdas.split(",") if x.strip()]

    word_result, word_prediction = _ridge_probe_split(
        tensors["word"], tensors["target"], train, val, test, lambdas, device, True
    )
    residual = tensors["target"] - word_prediction
    residual_fraction = float(
        (residual[test] ** 2).sum()
        / ((tensors["target"][test] - tensors["target"][train | val].mean(0)) ** 2).sum().clamp_min(1e-12)
    )
    probe_keys = [
        "direct_self",
        "direct_other_mean",
        "direct_register_flat",
        "direct_nonself",
        "direct_all",
        "routed_self",
        "routed_other",
        "routed_register",
        "routed_nonself",
        "routed_all_concat",
        "routed_sum",
    ]
    results = {}
    for key in probe_keys:
        log.info("Fitting appearance probe: %s (dim=%d)", key, tensors[key].shape[1])
        results[key], _ = _ridge_probe_split(
            tensors[key], residual, train, val, test, lambdas, device, False
        )
    results["direct_unique_self_delta_r2"] = {
        "r2": results["direct_all"]["r2"] - results["direct_nonself"]["r2"]
    }
    results["routed_unique_self_delta_r2"] = {
        "r2": results["routed_all_concat"]["r2"] - results["routed_nonself"]["r2"]
    }
    return {
        "split": split_summary,
        "num_objects": int(tensors["target"].shape[0]),
        "word_probe": word_result,
        "instance_residual_variance_fraction": residual_fraction,
        "probes_instance_residual": results,
        "reader_mass_on_target_region": {
            "self": _summary(tensors["reader_mass_self"].flatten().tolist()),
            "other_objects": _summary(tensors["reader_mass_other"].flatten().tolist()),
            "registers": _summary(tensors["reader_mass_register"].flatten().tolist()),
        },
    }


def _aggregate_influence(rows: list[dict]) -> dict:
    keys = [
        "diagonal_influence",
        "other_object_mean_influence",
        "other_object_max_influence",
        "background_influence",
        "diag_over_other_mean",
        "diag_over_background",
        "offdiag_over_diag",
        "background_over_diag",
    ]
    multi = [row for row in rows if row["num_objects"] > 1]
    diagonal_share = [
        row["diagonal_influence"]
        / max(
            row["diagonal_influence"]
            + row["other_object_mean_influence"]
            + row["background_influence"],
            1e-12,
        )
        for row in rows
    ]
    diagonal_sum = sum(row["diagonal_influence"] for row in rows)
    other_sum = sum(row["other_object_mean_influence"] for row in rows)
    background_sum = sum(row["background_influence"] for row in rows)
    return {
        "num_interventions": len(rows),
        "num_multi_object_interventions": len(multi),
        **{key: _summary(row[key] for row in rows) for key in keys},
        "diagonal_share_of_selected_other_background": _summary(diagonal_share),
        "global_sum_diag_over_other_mean": diagonal_sum / max(other_sum, 1e-12),
        "global_sum_diag_over_background": diagonal_sum / max(background_sum, 1e-12),
        "multi_object_only": {
            key: _summary(row[key] for row in multi) for key in keys
        },
        "selected_region_dominant_fraction": float(
            np.mean([row["selected_region_is_dominant"] for row in rows])
        ) if rows else None,
        "selected_region_dominant_fraction_multi_object": float(
            np.mean([row["selected_region_is_dominant"] for row in multi])
        ) if multi else None,
    }


def _save_plots(output: Path, summary: dict) -> None:
    influence = summary["causal_influence"]
    labels = ["selected object", "other objects", "background"]
    values = [
        influence["diagonal_influence"]["mean"],
        influence["other_object_mean_influence"]["mean"],
        influence["background_influence"]["mean"],
    ]
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.bar(labels, values, color=["#e76f51", "#4c78a8", "#9aa0a6"])
    ax.set_ylabel("same-noise prediction change")
    ax.set_title("Single-memory zero: causal responsibility")
    fig.tight_layout()
    fig.savefig(output / "causal_responsibility.png", dpi=180)
    plt.close(fig)

    layers = sorted(summary["writer_by_layer"], key=int)
    named = [summary["writer_by_layer"][x]["named_hard_iou"]["mean"] for x in layers]
    oracle = [summary["writer_by_layer"][x]["oracle_hard_iou"]["mean"] for x in layers]
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.plot(layers, named, marker="o", label="named slot IoU")
    ax.plot(layers, oracle, marker="o", label="oracle-matched IoU")
    ax.set_xlabel("Writer layer")
    ax.set_ylabel("hard IoU")
    ax.set_ylim(0.0, 1.0)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / "writer_layer_progression.png", dpi=180)
    plt.close(fig)

    probes = summary["appearance_information"]["probes_instance_residual"]
    probe_names = [
        "direct_self",
        "direct_other_mean",
        "direct_register_flat",
        "direct_nonself",
        "direct_all",
        "routed_self",
        "routed_other",
        "routed_register",
    ]
    probe_values = [probes[x]["r2"] for x in probe_names]
    fig, ax = plt.subplots(figsize=(9.5, 4.5))
    ax.bar(range(len(probe_names)), probe_values, color="#4c78a8")
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_xticks(range(len(probe_names)), probe_names, rotation=35, ha="right")
    ax.set_ylabel("held-out residual $R^2$")
    ax.set_title("Where is object appearance information?")
    fig.tight_layout()
    fig.savefig(output / "appearance_information_sources.png", dpi=180)
    plt.close(fig)


@torch.no_grad()
def run(args) -> dict:
    model, tokenizer, device, _ = load_model_and_tokenizer(args)
    if not bool(getattr(model.config, "pgot_e8_visual_memory_enable", False)):
        raise ValueError("This analysis requires an E8 visual-memory checkpoint")
    loader = build_loader(
        args, tokenizer, model, args.val_jsonl, shuffle=False, max_samples=args.max_samples
    )
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    layer_rows: dict[str, list[dict]] = defaultdict(list)
    transition_rows: dict[str, list[dict]] = defaultdict(list)
    causal_rows: list[dict] = []
    cache: Dict[str, list] = defaultdict(list)
    image_offset = 0
    model_object_slot_count = 0
    visible_object_count = 0

    for batch_idx, batch in enumerate(tqdm(loader, desc="E8 priority responsibility analysis")):
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
        batch_size, object_count = valid.shape
        masks = batch["gt_masks_per_ovt"][:, :object_count].to(device).float().clamp(0.0, 1.0)
        # Captions can mention objects that disappear after CODA center-crop.  They
        # remain valid model slots and therefore remain competitors, but they are
        # not valid GT targets for IoU, causal locality, or appearance probes.
        analysis_valid = valid & (masks.sum(dim=-1) > 1e-6)
        model_object_slot_count += int(valid.sum())
        visible_object_count += int(analysis_valid.sum())
        raw_categories = batch.get("ovt_category_ids")
        if raw_categories is None:
            categories = torch.full_like(valid, -1, dtype=torch.long)
        else:
            categories = raw_categories[:, :object_count].to(device=device, dtype=torch.long)

        records = sorted(out["e8_write_records"], key=lambda x: int(x["layer"].item()))
        for record in records:
            layer = str(int(record["layer"].item()))
            metrics, rows = _layer_writer_metrics(record, masks, analysis_valid)
            layer_rows[layer].append({"batch_size": batch_size, "metrics": metrics})
            for row in rows:
                row["image_batch_offset"] = image_offset
        for previous, current in zip(records, records[1:]):
            name = f"{int(previous['layer'].item())}_to_{int(current['layer'].item())}"
            transition_rows[name].append(_writer_transition_metrics(previous, current, masks))

        causal_rows.extend(
            _causal_influence(
                model,
                out,
                masks,
                analysis_valid,
                categories,
                image_offset,
                args.seed + batch_idx,
                args.intervention_batch_size,
            )
        )

        target = out["gt_siglip"].float()
        target_side = int(round(math.sqrt(target.shape[1])))
        query_count = out["reader_attention"].shape[1]
        query_masks = _resize_masks(masks, query_count).flatten(2)
        owner_attention = out["reader_attention"].float()
        memory_attention = out.get("reader_memory_attention")
        if memory_attention is None:
            memory_attention = owner_attention.unsqueeze(-1)
        else:
            memory_attention = memory_attention.float().reshape(
                batch_size,
                owner_attention.shape[1],
                memory.shape[1],
                memory.shape[2] if memory.ndim == 4 else 1,
            )
        value_memory = model.pgot_e8_reader.value(
            model.pgot_e8_reader.memory_norm(memory)
        ).float()
        if value_memory.ndim == 3:
            value_memory = value_memory.unsqueeze(2)
        register_start = object_count
        register_flat = memory[:, register_start:].flatten(1)

        for b in range(batch_size):
            target_indices = torch.nonzero(analysis_valid[b], as_tuple=False).flatten().tolist()
            model_object_indices = torch.nonzero(valid[b], as_tuple=False).flatten().tolist()
            for k in target_indices:
                target_mask = resize_mask(masks[b, k].flatten(), target_side)
                if float(target_mask.sum()) <= 1e-6:
                    continue
                cache["target"].append(pool_with_mask(target[b], target_mask).cpu())
                cache["word"].append(
                    word_embeddings(
                        model,
                        batch["caption_input_ids"],
                        batch["ovt_positions_in_caption"],
                        k,
                        int(model.pgot_n_ovt_per_object),
                        b,
                    )
                )
                cache["img_ids"].append(image_offset + b)
                self_memory = memory[b, k]
                cache["direct_self"].append(self_memory.flatten().cpu())
                # Non-self includes every object slot actually exposed to the
                # Reader, including captioned slots whose GT mask left the crop.
                other_indices = [j for j in model_object_indices if j != k]
                cache["direct_other_mean"].append(
                    _mean_or_zero(memory[b, other_indices], dim=0).flatten().cpu()
                )
                cache["direct_register_flat"].append(register_flat[b].cpu())

                qmask = query_masks[b, k]
                slot_weight = (
                    memory_attention[b] * qmask[:, None, None]
                ).sum(dim=0) / qmask.sum().clamp_min(1e-8)
                self_contribution = (
                    slot_weight[k, :, None] * value_memory[b, k]
                ).sum(dim=0)
                other_contribution = (
                    (
                        slot_weight[other_indices, :, None]
                        * value_memory[b, other_indices]
                    ).sum(dim=(0, 1))
                    if other_indices else value_memory.new_zeros(value_memory.shape[-1])
                )
                register_contribution = (
                    slot_weight[register_start:, :, None]
                    * value_memory[b, register_start:]
                ).sum(dim=(0, 1))
                cache["routed_self"].append(self_contribution.cpu())
                cache["routed_other"].append(other_contribution.cpu())
                cache["routed_register"].append(register_contribution.cpu())
                cache["reader_mass_self"].append(slot_weight[k].sum().reshape(1).cpu())
                cache["reader_mass_other"].append(slot_weight[other_indices].sum().reshape(1).cpu())
                cache["reader_mass_register"].append(slot_weight[register_start:].sum().reshape(1).cpu())
        image_offset += batch_size

    # Weighted aggregation for per-layer summaries.
    writer_summary = {}
    for layer, batches in layer_rows.items():
        metric_names = batches[0]["metrics"].keys()
        writer_summary[layer] = {}
        for metric_name in metric_names:
            values = []
            for batch in batches:
                metric = batch["metrics"][metric_name]
                if metric["mean"] is not None:
                    values.extend([metric["mean"]] * int(batch["batch_size"]))
            writer_summary[layer][metric_name] = _summary(values)

    transition_summary = {}
    for name, batches in transition_rows.items():
        keys = batches[0].keys()
        transition_summary[name] = {}
        for key in keys:
            if isinstance(batches[0][key], dict):
                values = [x[key]["mean"] for x in batches if x[key]["mean"] is not None]
                transition_summary[name][key] = _summary(values)
            else:
                transition_summary[name][key] = float(np.mean([x[key] for x in batches]))

    del model
    torch.cuda.empty_cache()
    appearance = _run_appearance_probes(cache, args, device)
    summary = {
        "model_path": args.model_path,
        "num_images": image_offset,
        "num_model_object_slots": model_object_slot_count,
        "num_visible_gt_objects": visible_object_count,
        "num_cropped_out_object_slots": model_object_slot_count - visible_object_count,
        "protocol": {
            "causal": "zero one object memory; identical RF timestep/noise; compare every GT object and background",
            "writer": "named slot IoU plus permutation-invariant Hungarian oracle IoU",
            "appearance": "image-disjoint 70/10/20 train/validation/test ridge on word-residual target",
        },
        "writer_by_layer": writer_summary,
        "writer_transitions": transition_summary,
        "causal_influence": _aggregate_influence(causal_rows),
        "appearance_information": appearance,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2))
    with (output / "influence_matrices.jsonl").open("w") as handle:
        for row in causal_rows:
            handle.write(json.dumps(row) + "\n")
    with (output / "appearance_probes.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["probe", "r2", "cosine", "rel_l2", "lambda", "validation_r2"],
        )
        writer.writeheader()
        for name, values in appearance["probes_instance_residual"].items():
            writer.writerow({"probe": name, **values})
    _save_plots(output, summary)
    log.info("Wrote priority analysis to %s", output)
    print(json.dumps(summary, indent=2))
    return summary


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
    parser.add_argument("--intervention_batch_size", type=int, default=16)
    parser.add_argument("--probe_lambdas", default="1e-2,1e0,1e2,1e4,1e6")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s :: %(message)s")
    run(args)


if __name__ == "__main__":
    main()
