"""Checkpoint-only causal diagnostics for PGOT E8 visual memory.

The diagnostic answers three questions without changing model weights:

1. Does the typed RAE reader spatially bind each object memory to its mask?
2. Does reconstruction loss depend on object memories, register memories, or both?
3. Does removing one object memory perturb the corresponding RAE query region?
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from pgot.eval.diagnose_e2_d1_d4 import _binding_metrics
from pgot.eval.eval_recon_oracles import build_loader, load_model_and_tokenizer
from pgot.eval.pgot_inference import pgot_forward_eval


log = logging.getLogger("pgot.diagnose_e8_visual_memory")


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
    semantic_slots = (
        memory
        if str(getattr(model.config, "pgot_e8_update_mode", "")) == "final_ovt"
        else out["semantic_slots"]
    )
    return model.pgot_e8_reader(
        rae_queries=out["raw_rae_hidden"],
        semantic_slots=semantic_slots,
        visual_memory=memory,
        slot_valid=slot_valid,
        memory_centroids=out.get("memory_centroids"),
        object_count=object_valid.shape[1],
    )["condition_hidden"]


def _fixed_recon_loss(model, condition, target, seed: int) -> float:
    devices = [condition.device.index] if condition.is_cuda else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(int(seed))
        loss = model._captionslot_compute_diffusion_loss(
            hidden=condition,
            target_features=target,
            slot_context=None,
            slot_mask=None,
        )
    return float(loss.detach())


def _resize_object_masks(batch, k_objects: int, query_count: int, device):
    masks = batch["gt_masks_per_ovt"][:, :k_objects].to(device=device).float()
    source_side = int(round(math.sqrt(masks.shape[-1])))
    query_side = int(round(math.sqrt(query_count)))
    if source_side * source_side != masks.shape[-1] or query_side * query_side != query_count:
        raise ValueError("E8 binding diagnostic expects square patch/query grids")
    return F.interpolate(
        masks.reshape(-1, 1, source_side, source_side),
        size=(query_side, query_side),
        mode="area",
    ).reshape(masks.shape[0], k_objects, query_count)


def _mean_dict(rows):
    if not rows:
        return {}
    return {
        key: float(np.nanmean([float(row[key]) for row in rows]))
        for key in rows[0]
    }


@torch.no_grad()
def run(args):
    model, tokenizer, device, _ = load_model_and_tokenizer(args)
    if not bool(getattr(model.config, "pgot_e8_visual_memory_enable", False)):
        raise ValueError("This diagnostic requires an E8 visual-memory checkpoint")
    loader = build_loader(
        args,
        tokenizer,
        model,
        args.val_jsonl,
        shuffle=False,
        max_samples=args.max_samples,
    )

    loss_sums = {
        "baseline": 0.0,
        "object_memory_only": 0.0,
        "register_memory_only": 0.0,
        "all_memory_zero": 0.0,
    }
    sample_count = 0
    reader_binding = []
    zero_object_binding = []
    reader_register_on_fg = []
    reader_object_on_bg = []
    condition_stats = {key: [] for key in loss_sums if key != "baseline"}
    memory_stats = {
        "object_norm": [],
        "register_norm": [],
        # Keep this historical name comparable across E8/E10/E11 by comparing
        # owner-level representations.  E11 first averages its J memories.
        "object_pair_cosine": [],
        # E11-specific: similarity among the J memories under the same owner.
        "within_owner_memory_pair_cosine": [],
    }
    exact_reader_max_error = 0.0
    writer_layer_sums = {}
    writer_layer_counts = {}

    for batch_idx, batch in enumerate(tqdm(loader, desc="E8 causal diagnostic")):
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
        n_register = memory.shape[1] - k_objects

        for record in out["e8_write_records"]:
            layer = str(int(record["layer"].item()))
            owner_stats = model._pgot_v14_compute_route_loss(
                owner_logits=record["owner_logits"],
                owner_probs=record["owner_probs"],
                gt_masks_per_ovt=batch["gt_masks_per_ovt"].to(device).float(),
                ovt_valid_mask=batch["ovt_valid_mask"].to(device).bool(),
            )
            values = {
                "loss": float(owner_stats["loss"]),
                "fg_acc": float(owner_stats["fg_acc"]),
                "bg_acc": float(owner_stats["bg_acc"]),
                "register_prob_on_fg": float(owner_stats["void_prob_on_fg"]),
                "object_prob_on_bg": float(owner_stats["object_prob_on_bg"]),
                "entropy": float(owner_stats["entropy"]),
            }
            if layer not in writer_layer_sums:
                writer_layer_sums[layer] = {key: 0.0 for key in values}
                writer_layer_counts[layer] = 0
            for key, value in values.items():
                writer_layer_sums[layer][key] += batch_size * value
            writer_layer_counts[layer] += batch_size

        object_only_memory = memory.clone()
        object_only_memory[:, k_objects:] = 0
        register_only_memory = memory.clone()
        register_only_memory[:, :k_objects] = 0
        zero_memory = torch.zeros_like(memory)
        conditions = {
            "baseline": out["rae_hidden"].float(),
            "object_memory_only": _reader_condition(model, out, object_only_memory).float(),
            "register_memory_only": _reader_condition(model, out, register_only_memory).float(),
            "all_memory_zero": _reader_condition(model, out, zero_memory).float(),
        }
        recomputed = _reader_condition(model, out, memory).float()
        exact_reader_max_error = max(
            exact_reader_max_error,
            float((recomputed - conditions["baseline"]).abs().max()),
        )
        seed = int(args.seed) + batch_idx
        for name, condition in conditions.items():
            loss_sums[name] += batch_size * _fixed_recon_loss(
                model, condition, out["gt_siglip"], seed
            )
            if name != "baseline":
                base = conditions["baseline"]
                delta = condition - base
                condition_stats[name].append(
                    {
                        "relative_l2": float(delta.norm() / base.norm().clamp_min(1e-8)),
                        "cosine": float(
                            F.cosine_similarity(
                                condition.flatten(1), base.flatten(1), dim=1
                            ).mean()
                        ),
                    }
                )
        sample_count += batch_size

        masks = _resize_object_masks(
            batch, k_objects, conditions["baseline"].shape[1], device
        )
        attention = out["reader_attention"].float()
        object_attention = attention[:, :, :k_objects]
        register_attention = attention[:, :, k_objects:].sum(dim=-1)
        fg_union = masks.amax(dim=1) > 0
        total_object_attention = object_attention.sum(dim=-1)
        for b in range(batch_size):
            if bool(fg_union[b].any()):
                reader_register_on_fg.append(float(register_attention[b][fg_union[b]].mean()))
            if bool((~fg_union[b]).any()):
                reader_object_on_bg.append(float(total_object_attention[b][~fg_union[b]].mean()))
            valid_indices = valid[b].nonzero(as_tuple=False).flatten().tolist()
            obj_mem = memory[b, :k_objects][valid[b]]
            if obj_mem.numel():
                memory_stats["object_norm"].extend(
                    obj_mem.reshape(-1, obj_mem.shape[-1]).norm(dim=-1).cpu().tolist()
                )
            if n_register > 0:
                memory_stats["register_norm"].extend(
                    memory[b, k_objects:]
                    .reshape(-1, memory.shape[-1])
                    .norm(dim=-1)
                    .cpu()
                    .tolist()
                )
            owner_mem = obj_mem.mean(dim=-2) if obj_mem.ndim == 3 else obj_mem
            if owner_mem.shape[0] > 1:
                normed = F.normalize(owner_mem, dim=-1)
                pair = normed @ normed.T
                tri = torch.triu_indices(pair.shape[0], pair.shape[1], offset=1)
                memory_stats["object_pair_cosine"].extend(pair[tri[0], tri[1]].cpu().tolist())
            if memory.ndim == 4 and memory.shape[2] > 1:
                all_owner_mem = memory[b][
                    torch.cat(
                        [
                            valid[b],
                            torch.ones(
                                n_register,
                                device=valid.device,
                                dtype=torch.bool,
                            ),
                        ]
                    )
                ]
                normed = F.normalize(all_owner_mem, dim=-1)
                pair = torch.einsum("sjd,skd->sjk", normed, normed)
                tri = torch.triu_indices(
                    memory.shape[2], memory.shape[2], offset=1, device=pair.device
                )
                memory_stats["within_owner_memory_pair_cosine"].extend(
                    pair[:, tri[0], tri[1]].flatten().cpu().tolist()
                )

            for k in valid_indices:
                reader_binding.append(
                    _binding_metrics(
                        object_attention[b, :, k].cpu(), masks[b, k].cpu()
                    )
                )

                ablated_memory = memory[b : b + 1].clone()
                ablated_memory[:, k] = 0
                single_out = {
                    "ovt_object_valid": out["ovt_object_valid"][b : b + 1],
                    "raw_rae_hidden": out["raw_rae_hidden"][b : b + 1],
                    "semantic_slots": out["semantic_slots"][b : b + 1],
                    "memory_centroids": (
                        out["memory_centroids"][b : b + 1]
                        if out.get("memory_centroids") is not None
                        else None
                    ),
                }
                ablated = _reader_condition(model, single_out, ablated_memory).float()
                delta_map = (
                    ablated - conditions["baseline"][b : b + 1]
                ).norm(dim=-1)[0]
                zero_object_binding.append(
                    _binding_metrics(delta_map.cpu(), masks[b, k].cpu())
                )

    losses = {key: value / max(sample_count, 1) for key, value in loss_sums.items()}
    base_loss = losses["baseline"]
    summary = {
        "model_path": args.model_path,
        "num_samples": sample_count,
        "fixed_noise_seed_base": int(args.seed),
        "diffusion_training_loss": losses,
        "diffusion_training_loss_delta_vs_baseline": {
            key: value - base_loss for key, value in losses.items() if key != "baseline"
        },
        "condition_change_vs_baseline": {
            key: _mean_dict(rows) for key, rows in condition_stats.items()
        },
        "reader_object_attention_binding": _mean_dict(reader_binding),
        "single_object_memory_zero_binding": _mean_dict(zero_object_binding),
        "reader_register_attention_on_gt_foreground": float(np.mean(reader_register_on_fg)),
        "reader_object_attention_on_gt_background": float(np.mean(reader_object_on_bg)),
        "memory": {
            key: {
                "mean": float(np.mean(values)) if values else float("nan"),
                "std": float(np.std(values)) if values else float("nan"),
                "count": len(values),
            }
            for key, values in memory_stats.items()
        },
        "reader_recompute_max_abs_error": exact_reader_max_error,
        "num_object_binding_records": len(reader_binding),
        "writer_ownership_by_layer": {
            layer: {
                key: value / max(writer_layer_counts[layer], 1)
                for key, value in values.items()
            }
            for layer, values in writer_layer_sums.items()
        },
    }
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(json.dumps(summary, indent=2))
    log.info("Wrote %s", output / "summary.json")
    return summary


def main():
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
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s :: %(message)s")
    print(json.dumps(run(args), indent=2))


if __name__ == "__main__":
    main()
