"""Structure-aware diagnostics for the one-shot PGOT Reader.

Unlike the E8/E11 diagnostics, this script never intervenes on a persistent
visual memory: the one-shot model has none.  It instead perturbs the predicted
owner map and the spatial information in frozen raw SigLIP patches, then
measures fixed-noise diffusion loss and routing/locality statistics.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from pgot.eval.diagnose_e8_visual_memory import _fixed_recon_loss
from pgot.eval.eval_recon_oracles import build_loader, load_model_and_tokenizer
from pgot.eval.pgot_inference import pgot_forward_eval


log = logging.getLogger("pgot.diagnose_one_shot_reader")


def _mean(rows: list[float]) -> float:
    return float(np.mean(rows)) if rows else float("nan")


def _condition(
    model,
    out: dict,
    *,
    raw_patches: torch.Tensor | None = None,
    owner_probs: torch.Tensor | None = None,
    semantic_slots: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict]:
    valid = out["ovt_object_valid"].bool()
    n_register = out["semantic_slots"].shape[1] - valid.shape[1]
    slot_valid = torch.cat(
        [
            valid,
            torch.ones(
                valid.shape[0], n_register, device=valid.device, dtype=torch.bool
            ),
        ],
        dim=1,
    )
    if owner_probs is None:
        owner_probs = torch.cat(
            [out["ovt_object_probs"], out["ovt_void_probs"]], dim=1
        )
    reader = model.pgot_e8_reader(
        rae_queries=out["raw_rae_hidden"],
        semantic_slots=(
            out["semantic_slots"] if semantic_slots is None else semantic_slots
        ),
        raw_patches=(
            out["raw_img_features"] if raw_patches is None else raw_patches
        ),
        owner_probs=owner_probs,
        slot_valid=slot_valid,
    )
    return reader["condition_hidden"], reader


def _gt_owner_probs(batch: dict, predicted: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """Build an oracle object/background map while preserving register splits."""
    B, S, P = predicted.shape
    K = valid.shape[1]
    masks = batch["gt_masks_per_ovt"][:, :K].to(predicted.device).float()
    if masks.shape[-1] != P:
        side = int(round(math.sqrt(masks.shape[-1])))
        target_side = int(round(math.sqrt(P)))
        masks = F.interpolate(
            masks.reshape(B * K, 1, side, side),
            size=(target_side, target_side),
            mode="area",
        ).reshape(B, K, P)
    masks = masks * valid.unsqueeze(-1).float()
    # COCO instances can overlap.  At each patch, keep total object mass <= 1.
    object_sum = masks.sum(dim=1, keepdim=True)
    object_probs = masks / object_sum.clamp_min(1.0)
    background = (1.0 - object_probs.sum(dim=1, keepdim=True)).clamp_min(0.0)
    registers = predicted[:, K:]
    registers = registers / registers.sum(dim=1, keepdim=True).clamp_min(1e-6)
    return torch.cat([object_probs, background * registers], dim=1)


def _spatial_mean_ablation(
    raw: torch.Tensor, patch_owner: torch.Tensor, keep_object: bool, k_objects: int
) -> torch.Tensor:
    """Remove local detail outside one owner type without introducing zero OOD values."""
    mean = raw.mean(dim=1, keepdim=True)
    object_patch = patch_owner < k_objects
    keep = object_patch if keep_object else ~object_patch
    return torch.where(keep.unsqueeze(-1), raw, mean.expand_as(raw))


@torch.no_grad()
def run(args: argparse.Namespace) -> dict:
    model, tokenizer, device, _ = load_model_and_tokenizer(args)
    if not bool(getattr(model.config, "pgot_one_shot_reader_enable", False)):
        raise ValueError("This diagnostic requires a one-shot Reader checkpoint")
    loader = build_loader(
        args, tokenizer, model, args.val_jsonl, shuffle=False, max_samples=args.max_samples
    )

    branch_names = [
        "baseline",
        "object_region_local_only",
        "register_region_local_only",
        "all_spatial_mean",
        "owner_map_spatial_shuffle",
        "raw_patch_spatial_shuffle",
        "gt_owner_routing",
    ]
    loss_sums = {name: 0.0 for name in branch_names}
    condition_changes = {name: [] for name in branch_names if name != "baseline"}
    route_stats: dict[str, list[float]] = {
        "owner_correct_object_patch_top1": [],
        "owner_register_on_gt_foreground_top1": [],
        "owner_object_on_gt_background_top1": [],
        "reader_correct_object_query_top1": [],
        "reader_register_on_gt_foreground_top1": [],
        "reader_object_on_gt_background_top1": [],
        "patch_attention_same_object_mass": [],
        "patch_attention_background_mass_on_bg_query": [],
        "patch_attention_expected_distance": [],
        "reader_object_owner_utilization": [],
        "reader_register_owner_utilization": [],
        "patch_attention_effective_count": [],
    }
    sample_count = 0

    for batch_idx, batch in enumerate(tqdm(loader, desc="one-shot diagnostic")):
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
        if raw is None:
            raise RuntimeError("one-shot forward did not expose raw SigLIP patches")
        valid = out["ovt_object_valid"].bool()
        B, K = valid.shape
        predicted = torch.cat(
            [out["ovt_object_probs"], out["ovt_void_probs"]], dim=1
        ).float()
        patch_owner = predicted.argmax(dim=1)
        spatial_mean = raw.mean(dim=1, keepdim=True).expand_as(raw)
        generator = torch.Generator(device=raw.device)
        generator.manual_seed(int(args.seed) + batch_idx)
        permutation = torch.randperm(raw.shape[1], generator=generator, device=raw.device)
        branches: dict[str, tuple[torch.Tensor | None, torch.Tensor | None]] = {
            "baseline": (None, None),
            "object_region_local_only": (
                _spatial_mean_ablation(raw, patch_owner, True, K), None
            ),
            "register_region_local_only": (
                _spatial_mean_ablation(raw, patch_owner, False, K), None
            ),
            "all_spatial_mean": (spatial_mean, None),
            "owner_map_spatial_shuffle": (None, predicted[:, :, permutation]),
            "raw_patch_spatial_shuffle": (raw[:, permutation], None),
            "gt_owner_routing": (
                None, _gt_owner_probs(batch, predicted, valid)
            ),
        }
        conditions = {}
        readers = {}
        for name, (raw_branch, owner_branch) in branches.items():
            conditions[name], readers[name] = _condition(
                model, out, raw_patches=raw_branch, owner_probs=owner_branch
            )
        # Verify that the diagnostic exactly reconstructs the actual forward.
        max_error = (conditions["baseline"] - out["rae_hidden"].float()).abs().max()
        if float(max_error) > 5e-4:
            raise RuntimeError(f"Reader recompute mismatch: {float(max_error):.6g}")
        fixed_seed = int(args.seed) + batch_idx
        for name, condition in conditions.items():
            loss_sums[name] += B * _fixed_recon_loss(
                model, condition, out["gt_siglip"], fixed_seed
            )
            if name != "baseline":
                delta = condition - conditions["baseline"]
                relative = delta.flatten(1).norm(dim=1) / conditions[
                    "baseline"
                ].flatten(1).norm(dim=1).clamp_min(1e-8)
                condition_changes[name].extend(relative.cpu().tolist())
        sample_count += B

        # Hard ownership accuracy on 32x32 source patches.
        gt_masks = batch["gt_masks_per_ovt"][:, :K].to(device).float()
        gt_present = gt_masks * valid.unsqueeze(-1).float()
        gt_max, gt_owner = gt_present.max(dim=1)
        gt_fg = gt_max > 0.5
        for b in range(B):
            if bool(gt_fg[b].any()):
                route_stats["owner_correct_object_patch_top1"].append(
                    float((patch_owner[b][gt_fg[b]] == gt_owner[b][gt_fg[b]]).float().mean())
                )
                route_stats["owner_register_on_gt_foreground_top1"].append(
                    float((patch_owner[b][gt_fg[b]] >= K).float().mean())
                )
            if bool((~gt_fg[b]).any()):
                route_stats["owner_object_on_gt_background_top1"].append(
                    float((patch_owner[b][~gt_fg[b]] < K).float().mean())
                )

        reader = readers["baseline"]
        owner_attention = reader["reader_attention_heads"].mean(dim=1)
        query_owner = owner_attention.argmax(dim=-1)
        patch_attention = reader["reader_patch_attention_heads"].mean(dim=1).float()
        Q, P = patch_attention.shape[1:]
        q_side, p_side = int(round(math.sqrt(Q))), int(round(math.sqrt(P)))
        query_masks = F.interpolate(
            gt_present.reshape(B * K, 1, p_side, p_side),
            size=(q_side, q_side),
            mode="area",
        ).reshape(B, K, Q)
        q_max, q_gt_owner = query_masks.max(dim=1)
        q_fg = q_max > 0.5
        patch_bg = (~gt_fg).float()
        for b in range(B):
            if bool(q_fg[b].any()):
                route_stats["reader_correct_object_query_top1"].append(
                    float((query_owner[b][q_fg[b]] == q_gt_owner[b][q_fg[b]]).float().mean())
                )
                route_stats["reader_register_on_gt_foreground_top1"].append(
                    float((query_owner[b][q_fg[b]] >= K).float().mean())
                )
                same = []
                for q in torch.nonzero(q_fg[b], as_tuple=False).flatten():
                    obj = int(q_gt_owner[b, q])
                    same.append((patch_attention[b, q] * gt_present[b, obj]).sum())
                route_stats["patch_attention_same_object_mass"].append(
                    float(torch.stack(same).mean())
                )
            if bool((~q_fg[b]).any()):
                route_stats["reader_object_on_gt_background_top1"].append(
                    float((query_owner[b][~q_fg[b]] < K).float().mean())
                )
                bg_mass = (
                    patch_attention[b, ~q_fg[b]] * patch_bg[b].unsqueeze(0)
                ).sum(dim=-1)
                route_stats["patch_attention_background_mass_on_bg_query"].append(
                    float(bg_mass.mean())
                )
            used = torch.unique(query_owner[b])
            route_stats["reader_object_owner_utilization"].append(
                float((used < K).sum() / valid[b].sum().clamp_min(1))
            )
            route_stats["reader_register_owner_utilization"].append(
                float((used >= K).sum() / max(predicted.shape[1] - K, 1))
            )

        q_y, q_x = torch.meshgrid(
            (torch.arange(q_side, device=device) + 0.5) / q_side,
            (torch.arange(q_side, device=device) + 0.5) / q_side,
            indexing="ij",
        )
        p_y, p_x = torch.meshgrid(
            (torch.arange(p_side, device=device) + 0.5) / p_side,
            (torch.arange(p_side, device=device) + 0.5) / p_side,
            indexing="ij",
        )
        q_coord = torch.stack([q_y.flatten(), q_x.flatten()], dim=-1)
        p_coord = torch.stack([p_y.flatten(), p_x.flatten()], dim=-1)
        expected = torch.einsum("bqp,pd->bqd", patch_attention, p_coord)
        route_stats["patch_attention_expected_distance"].extend(
            (expected - q_coord.unsqueeze(0)).norm(dim=-1).mean(dim=1).cpu().tolist()
        )
        entropy = -(
            patch_attention.clamp_min(1e-8).log() * patch_attention
        ).sum(dim=-1)
        route_stats["patch_attention_effective_count"].extend(
            entropy.exp().mean(dim=1).cpu().tolist()
        )

    losses = {name: value / max(sample_count, 1) for name, value in loss_sums.items()}
    baseline = losses["baseline"]
    summary = {
        "model_path": args.model_path,
        "readout_mode": str(getattr(model.config, "pgot_one_shot_readout_mode", "")),
        "num_samples": sample_count,
        "fixed_noise_seed_base": int(args.seed),
        "diffusion_training_loss": losses,
        "diffusion_training_loss_delta_vs_baseline": {
            name: value - baseline for name, value in losses.items() if name != "baseline"
        },
        "condition_relative_l2_change": {
            name: _mean(values) for name, values in condition_changes.items()
        },
        "routing_and_locality": {
            name: _mean(values) for name, values in route_stats.items()
        },
        "interpretation": {
            "object_region_local_only": "register-owned raw patches replaced by image mean",
            "register_region_local_only": "object-owned raw patches replaced by image mean",
            "all_spatial_mean": "all raw patches replaced by the image-mean raw feature",
            "owner_map_spatial_shuffle": "owner probabilities permuted over patch positions",
            "raw_patch_spatial_shuffle": "raw feature positions permuted while owner map stays fixed",
            "gt_owner_routing": "GT object masks; predicted relative split among background registers",
        },
    }
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(json.dumps(summary, indent=2))
    log.info("Wrote %s", output / "summary.json")
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
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s :: %(message)s")
    print(json.dumps(run(args), indent=2))


if __name__ == "__main__":
    main()
