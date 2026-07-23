"""Diagnose whether OVTs read visual information from image patches.

This is an eval-only probe for V15/V20-style checkpoints. It measures:
  - source mass of OVT queries in exact LLM attention (image/text/self/void/etc.)
  - inside/outside image attention relative to each object's GT mask
  - CODA-style segmentation metrics from OVT->image attention maps
  - the usual dot-product threshold metrics on the same samples for comparison
"""

import argparse
import gc
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from pgot.eval.pgot_inference import pgot_forward_eval
from pgot.eval.pgot_metrics import fari_metric, mbo_metric, miou_metric, ovt_logits_to_pred_mask
from pgot.eval.run_eval import CocoInstanceMaskCache, load_gt_panoptic_mask, load_thing_categories
from pgot.eval.visualize_ovt_overlays import _load_model
from pgot.model.pgot_utils import build_pred_mask_llm_attention_eval
from pgot.train.pgot_dataset import PGOTDataCollator, Pix2CapPGOTDataset


def _mean(xs: Iterable[float]) -> float:
    vals = [float(x) for x in xs if np.isfinite(float(x))]
    return float(np.mean(vals)) if vals else float("nan")


def _parse_layers(model, spec: str) -> List[int]:
    if hasattr(model, "_resolve_llm_qk_outside_layers"):
        return list(model._resolve_llm_qk_outside_layers(spec))
    n = int(getattr(model.config, "num_hidden_layers", len(model.model.layers)))
    if spec.startswith("last"):
        count = int(spec[4:] or "1")
        return list(range(max(0, n - count), n))
    return [int(x.strip()) for x in spec.split(",") if x.strip()]


def _resize_masks_to_p(masks: torch.Tensor, target_p: int) -> torch.Tensor:
    if masks.shape[-1] == target_p:
        return masks
    src = int(round(float(masks.shape[-1]) ** 0.5))
    dst = int(round(float(target_p) ** 0.5))
    if src * src != masks.shape[-1] or dst * dst != target_p:
        raise ValueError(f"Expected square masks, got {masks.shape[-1]} -> {target_p}.")
    B, M, _ = masks.shape
    return F.interpolate(
        masks.reshape(B * M, 1, src, src).float(),
        size=(dst, dst),
        mode="area",
    ).reshape(B, M, target_p).clamp(0.0, 1.0)


def _remap_thing_flags(batch, samples_iter, global_indices, thing_categories, n_ovt_per_object):
    valid = batch["ovt_valid_mask"]
    ovt_is_thing = batch.get("ovt_is_thing")
    if ovt_is_thing is None:
        ovt_is_thing = torch.zeros_like(valid, dtype=torch.bool)
    else:
        ovt_is_thing = ovt_is_thing.clone().to(dtype=torch.bool)
    if thing_categories is None:
        return ovt_is_thing
    for b, global_idx in enumerate(global_indices):
        segs = samples_iter[int(global_idx)]["segments"]
        for k, seg in enumerate(segs):
            s = k * n_ovt_per_object
            e = s + n_ovt_per_object
            if e <= ovt_is_thing.shape[1] and seg.get("category") in thing_categories:
                ovt_is_thing[b, s:e] = True
    return ovt_is_thing


def _safe_gather_mass(probs: torch.Tensor, positions: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """probs: (B,H,M,L), positions/valid: (B,M). Returns (B,H,M)."""
    B, H, M, _ = probs.shape
    idx = positions.clamp_min(0).to(probs.device)
    selected = probs.gather(dim=-1, index=idx[:, None, :, None].expand(B, H, M, 1)).squeeze(-1)
    return selected * valid[:, None, :].to(selected.dtype)


def _exact_ovt_full_probs_for_layer(model, *, layer_input, layer_idx, attention_bias, positions, ovt_abs_positions, ovt_valid_mask):
    """Reconstruct exact full-key OVT-row attention for one layer.

    Returns probs over all keys with object OVT rows only: (B,H,M,L).
    """
    block = model.model.layers[layer_idx]
    attn = block.self_attn
    attn_input = block.input_layernorm(layer_input)
    B, L, _ = attn_input.shape
    M = ovt_abs_positions.shape[1]

    safe_q_pos = ovt_abs_positions.clamp(min=0, max=L - 1)
    q_input = attn_input.gather(
        dim=1,
        index=safe_q_pos.unsqueeze(-1).expand(-1, -1, attn_input.shape[-1]),
    )
    q_input = q_input * ovt_valid_mask.unsqueeze(-1).to(q_input.dtype)
    q_proj = attn.q_proj(q_input)
    k_proj = attn.k_proj(attn_input)

    num_heads = int(getattr(attn, "num_heads", model.config.num_attention_heads))
    num_kv_heads = int(getattr(attn, "num_key_value_heads", model.config.num_key_value_heads))
    head_dim = int(getattr(attn, "head_dim", q_proj.shape[-1] // num_heads))
    q = q_proj.view(B, M, num_heads, head_dim).transpose(1, 2)
    k = k_proj.view(B, L, num_kv_heads, head_dim).transpose(1, 2)

    cos, sin = attn.rotary_emb(k, seq_len=L)
    cos = cos.to(device=q.device, dtype=q.dtype)
    sin = sin.to(device=q.device, dtype=q.dtype)
    all_pos = torch.arange(L, device=q.device, dtype=torch.long)
    k_cos = cos[all_pos].view(1, 1, L, head_dim)
    k_sin = sin[all_pos].view(1, 1, L, head_dim)
    q_cos = cos[safe_q_pos].unsqueeze(1)
    q_sin = sin[safe_q_pos].unsqueeze(1)
    q = (q * q_cos) + (model._pgot_rotate_half(q) * q_sin)
    k = (k * k_cos) + (model._pgot_rotate_half(k) * k_sin)

    if num_kv_heads != num_heads:
        groups = int(getattr(attn, "num_key_value_groups", num_heads // num_kv_heads))
        k = k.repeat_interleave(groups, dim=1)

    scores = torch.matmul(q.float(), k.float().transpose(2, 3)) / math.sqrt(float(head_dim))
    bias_rows = attention_bias.gather(
        dim=2,
        index=safe_q_pos[:, None, :, None].expand(B, 1, M, L),
    )
    scores = scores + bias_rows.float()
    probs = F.softmax(scores, dim=-1, dtype=torch.float32)
    return probs * ovt_valid_mask[:, None, :, None].float()


def _source_masses(model, out, batch, layers: List[int], n_ovt_per_object: int, patch_temperature: float):
    device = out["attn_bias"].device
    gt_masks = _resize_masks_to_p(
        batch["gt_masks_per_ovt"].to(device).float(),
        int(out["positions"]["img_e"] - out["positions"]["img_s"]),
    )
    ovt_valid = out["ovt_valid_mask"].to(device, dtype=torch.bool)
    valid_f = ovt_valid.float()
    denom = valid_f.sum().clamp_min(1.0)

    per_layer = []
    object_patch_acc = None
    void_patch_acc = None
    used = 0
    for layer_idx in layers:
        if layer_idx >= len(out["hidden_states"]) - 1:
            continue
        probs = _exact_ovt_full_probs_for_layer(
            model,
            layer_input=out["hidden_states"][layer_idx],
            layer_idx=layer_idx,
            attention_bias=out["attn_bias"],
            positions=out["positions"],
            ovt_abs_positions=out["ovt_abs_positions"],
            ovt_valid_mask=ovt_valid,
        )
        object_full, void_full, object_patch, void_patch = model._compute_exact_llm_attention_components_for_layer(
            layer_input=out["hidden_states"][layer_idx],
            layer_idx=layer_idx,
            attention_bias=out["attn_bias"],
            positions=out["positions"],
            ovt_abs_positions=out["ovt_abs_positions"],
            ovt_valid_mask=ovt_valid,
            patch_temperature=patch_temperature,
        )
        object_patch_acc = object_patch.mean(dim=1) if object_patch_acc is None else object_patch_acc + object_patch.mean(dim=1)
        void_patch_acc = void_patch.mean(dim=1) if void_patch_acc is None else void_patch_acc + void_patch.mean(dim=1)
        used += 1

        pos = out["positions"]
        img = probs[..., pos["img_s"]:pos["img_e"]].sum(dim=-1)
        inside = (object_full * gt_masks[:, None]).sum(dim=-1)
        outside = (object_full * (1.0 - gt_masks[:, None]).clamp(0, 1)).sum(dim=-1)
        patch_inside = (object_patch * gt_masks[:, None]).sum(dim=-1)
        patch_outside = (object_patch * (1.0 - gt_masks[:, None]).clamp(0, 1)).sum(dim=-1)

        self_mass = _safe_gather_mass(probs, out["ovt_abs_positions"], ovt_valid)
        all_ovt_mass = probs.gather(
            dim=-1,
            index=out["ovt_abs_positions"].clamp_min(0)[:, None, None, :].expand(
                probs.shape[0], probs.shape[1], probs.shape[2], ovt_valid.shape[1]
            ),
        )
        all_ovt_mass = (all_ovt_mass * ovt_valid[:, None, None, :].float()).sum(dim=-1)

        def key_slice_mass(name_s, name_e):
            s = int(pos.get(name_s, 0))
            e = int(pos.get(name_e, s))
            if e <= s:
                return probs.new_zeros(probs.shape[:3])
            return probs[..., s:e].sum(dim=-1)

        void_mass = key_slice_mass("null_bg_s", "null_bg_e")
        rae_mass = key_slice_mass("rae_s", "rae_e")
        reg_mass = key_slice_mass("reg_s", "reg_e")
        cap_mass = key_slice_mass("cap_s", "cap_e")
        nonimage = 1.0 - img

        def valid_mean(x):
            return float(((x.mean(dim=1) * valid_f).sum() / denom).detach().cpu())

        per_layer.append({
            "layer": int(layer_idx),
            "image_mass": valid_mean(img),
            "inside_image_mass": valid_mean(inside),
            "outside_image_mass": valid_mean(outside),
            "conditional_inside_mass": valid_mean(patch_inside),
            "conditional_outside_mass": valid_mean(patch_outside),
            "nonimage_mass": valid_mean(nonimage),
            "self_mass": valid_mean(self_mass),
            "other_ovt_mass": valid_mean((all_ovt_mass - self_mass).clamp_min(0.0)),
            "caption_segment_mass_including_ovt": valid_mean(cap_mass),
            "void_mass": valid_mean(void_mass),
            "rae_mass": valid_mean(rae_mass),
            "register_mass": valid_mean(reg_mass),
        })

    if used <= 0:
        return per_layer, None, None
    return per_layer, (object_patch_acc / float(used)).detach(), (void_patch_acc / float(used)).detach()


def _evaluate_metrics(gt_masks, pred_masks, overlap_masks):
    out = {"fARI": [], "mBO": [], "mIoU": []}
    for b in range(gt_masks.shape[0]):
        overlap = overlap_masks[b:b + 1] if overlap_masks is not None else None
        gt = gt_masks[b:b + 1]
        pred = pred_masks[b:b + 1]
        out["fARI"].append(fari_metric(gt, pred, overlap))
        out["mBO"].append(mbo_metric(gt, pred, overlap))
        out["mIoU"].append(miou_metric(gt, pred, overlap))
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--val_jsonl", default="/home/jovyan/PGOT/data/pgot_val.jsonl")
    parser.add_argument("--sample_indices", default="")
    parser.add_argument("--max_samples", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--layers", default="last4")
    parser.add_argument("--patch_temperature", type=float, default=1.0)
    parser.add_argument("--grid_size", type=int, default=32)
    parser.add_argument("--eval_size", type=int, default=224)
    parser.add_argument("--n_ovt_per_object", type=int, default=2)
    parser.add_argument("--max_objects", type=int, default=50)
    parser.add_argument("--gt_source", choices=["coco_instance", "panoptic"], default="coco_instance")
    parser.add_argument("--coco_mask_cache", default="/home/jovyan/PGOT/data/coco_inst_mask_cache_coda256")
    parser.add_argument("--image_preprocess_mode", choices=["default", "coda_center_crop"], default="coda_center_crop")
    parser.add_argument("--coda_crop_size", type=int, default=512)
    parser.add_argument("--dtype", choices=["fp32", "bf16"], default="fp32")
    parser.add_argument("--bg_threshold", type=float, default=0.5)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32 if args.dtype == "fp32" else torch.bfloat16

    model, tokenizer = _load_model(args.model_path, dtype, device)
    layers = _parse_layers(model, args.layers)
    vt_list = model.get_vision_tower_aux_list()
    image_proc = vt_list[0].image_processor
    target_proc = vt_list[1].image_processor if len(vt_list) > 1 else image_proc
    dataset = Pix2CapPGOTDataset(
        jsonl_path=args.val_jsonl,
        tokenizer=tokenizer,
        image_processor=image_proc,
        target_image_processor=target_proc,
        grid_size=args.grid_size,
        max_caption_tokens=2048,
        n_ovt_per_object=args.n_ovt_per_object,
        max_objects=args.max_objects,
        panoptic_categories_json="/home/jovyan/data/coco/annotations/panoptic_val2017.json",
        image_preprocess_mode=args.image_preprocess_mode,
        coda_crop_size=args.coda_crop_size,
    )
    with open(args.val_jsonl) as f:
        samples_iter = [json.loads(line) for line in f]

    if args.sample_indices.strip():
        indices = [int(x) for x in args.sample_indices.split(",") if x.strip()]
    else:
        indices = list(range(min(args.max_samples, len(dataset))))
    subset = Subset(dataset, indices)
    loader = DataLoader(
        subset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=PGOTDataCollator(pad_token_id=tokenizer.pad_token_id),
    )
    thing_categories = (
        load_thing_categories("/home/jovyan/data/coco/annotations/panoptic_val2017.json")
        if args.gt_source == "coco_instance"
        else None
    )
    coco_cache = CocoInstanceMaskCache(args.coco_mask_cache) if args.gt_source == "coco_instance" else None
    if coco_cache is not None:
        args.eval_size = coco_cache.size

    source_records = []
    attn_metrics = {"fARI": [], "mBO": [], "mIoU": []}
    dot_metrics = {"fARI": [], "mBO": [], "mIoU": []}
    per_sample = []
    cursor = 0
    for batch in tqdm(loader, desc="ovt-attn-diagnostic"):
        global_indices = indices[cursor:cursor + batch["images"].shape[0]]
        cursor += batch["images"].shape[0]
        with torch.no_grad():
            out = pgot_forward_eval(
                model,
                images=batch["images"],
                target_images=batch["target_images"],
                caption_input_ids=batch["caption_input_ids"],
                caption_attention_mask=batch["caption_attention_mask"],
                ovt_positions_in_caption=batch["ovt_positions_in_caption"],
                ovt_valid_mask=batch["ovt_valid_mask"],
                return_hidden_states=True,
            )
            layer_records, attn_maps, void_maps = _source_masses(
                model,
                out,
                batch,
                layers,
                args.n_ovt_per_object,
                args.patch_temperature,
            )
            register_maps = model._compute_llm_register_patch_attention_maps(
                hidden_states=out["hidden_states"],
                attention_bias=out["attn_bias"],
                positions=out["positions"],
                layers_spec=args.layers,
                temperature=args.patch_temperature,
            ) if hasattr(model, "_compute_llm_register_patch_attention_maps") else None

        source_records.extend(layer_records)
        ovt_is_thing = _remap_thing_flags(
            batch,
            samples_iter,
            global_indices,
            thing_categories,
            args.n_ovt_per_object,
        ).to(out["ovt_valid_mask"].device)
        attn_grid = int(round(float(attn_maps.shape[-1]) ** 0.5))
        bg_maps = void_maps
        if bg_maps is None or bg_maps.numel() == 0:
            bg_maps = register_maps
        pred_attn = build_pred_mask_llm_attention_eval(
            attn_maps,
            bg_maps,
            out["ovt_valid_mask"],
            ovt_is_thing,
            target_size=args.eval_size,
            n_ovt_per_object=args.n_ovt_per_object,
            patch_grid=attn_grid,
            merge="mean",
            map_stuff_to_bg=(args.gt_source == "coco_instance"),
        )
        valid_thing = out["ovt_valid_mask"] & ovt_is_thing.to(out["ovt_valid_mask"].device)
        pred_dot = ovt_logits_to_pred_mask(
            out["ovt_logits"],
            valid_thing,
            target_size=args.eval_size,
            n_ovt_per_object=args.n_ovt_per_object,
            patch_grid=args.grid_size,
            bg_threshold=args.bg_threshold,
        )

        gt_list = []
        overlap_list = []
        for local_i, global_idx in enumerate(global_indices):
            sample = samples_iter[int(global_idx)]
            if args.gt_source == "coco_instance":
                image_id = int(batch["image_ids"][local_i])
                gt = coco_cache.get(image_id)
                if gt is None:
                    seg_ids = [int(s["segment_id"]) for s in sample["segments"]]
                    gt = load_gt_panoptic_mask(sample["panoptic_mask_path"], seg_ids, args.eval_size)
                overlap = coco_cache.get_overlap(image_id)
                if overlap is None:
                    overlap = torch.zeros_like(gt, dtype=torch.uint8)
            else:
                seg_ids = [int(s["segment_id"]) for s in sample["segments"]]
                gt = load_gt_panoptic_mask(sample["panoptic_mask_path"], seg_ids, args.eval_size)
                overlap = None
            gt_list.append(gt)
            if overlap is not None:
                overlap_list.append(overlap)
        gt_masks = torch.stack(gt_list).to(pred_attn.device)
        overlap_masks = torch.stack(overlap_list).to(pred_attn.device) if overlap_list else None
        a = _evaluate_metrics(gt_masks, pred_attn, overlap_masks)
        d = _evaluate_metrics(gt_masks, pred_dot, overlap_masks)
        for key in attn_metrics:
            attn_metrics[key].extend(a[key])
            dot_metrics[key].extend(d[key])
        for i, global_idx in enumerate(global_indices):
            per_sample.append({
                "sample_index": int(global_idx),
                "image_id": int(batch["image_ids"][i]),
                "n_objects": int(batch["n_objects_list"][i]),
                "attention": {k: float(a[k][i]) for k in a},
                "dot_threshold": {k: float(d[k][i]) for k in d},
            })

    by_layer: Dict[str, Dict[str, float]] = {}
    for layer_idx in sorted({r["layer"] for r in source_records}):
        rows = [r for r in source_records if r["layer"] == layer_idx]
        by_layer[str(layer_idx)] = {
            key: _mean(r[key] for r in rows)
            for key in rows[0]
            if key != "layer"
        }
    overall = {
        key: _mean(r[key] for r in source_records)
        for key in source_records[0]
        if key != "layer"
    } if source_records else {}
    summary = {
        "model_path": args.model_path,
        "sample_indices": indices,
        "layers": layers,
        "image_preprocess_mode": args.image_preprocess_mode,
        "coda_crop_size": args.coda_crop_size,
        "gt_source": args.gt_source,
        "source_mass_overall": overall,
        "source_mass_by_layer": by_layer,
        "attention_readout": {k: _mean(v) for k, v in attn_metrics.items()},
        "dot_threshold_readout_same_samples": {k: _mean(v) for k, v in dot_metrics.items()},
        "per_sample": per_sample,
    }
    with (out_dir / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2)
    with (out_dir / "per_layer_records.json").open("w") as f:
        json.dump(source_records, f, indent=2)
    print(out_dir / "summary.json")
    del model, tokenizer, dataset
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
