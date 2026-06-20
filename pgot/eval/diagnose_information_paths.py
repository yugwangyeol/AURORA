"""Diagnose whether PGOT reconstruction uses OVTs or register tokens.

The script loads each checkpoint once and, for every requested sample:
  - reruns reconstruction conditioning with baseline / OVT-only / register-only /
    self-only RAE access;
  - measures reconstruction loss and RAE hidden-state change;
  - measures reconstruction gradients at image, OVT, and register input tokens;
  - saves register-to-image maps from final dot products and selected LLM Q/K layers;
  - optionally decodes the four reconstruction variants and performs an OVT swap.
"""

import argparse
import gc
import json
import logging
import math
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont

from pgot.eval.pgot_inference import (
    generate_siglip_latent,
    ovt_swap_inference,
    pgot_forward_eval,
)
from pgot.eval.run_eval import decode_to_image, denormalize_images, load_rae_decoder
from pgot.eval.visualize_ovt_overlays import _load_model
from pgot.train.pgot_dataset import PGOTDataCollator, Pix2CapPGOTDataset

log = logging.getLogger("pgot.information_paths")


def parse_model(spec: str) -> tuple[str, str]:
    label, path = spec.split("|", 1)
    return label, path


def selected_layers(model, spec: str) -> list[int]:
    if hasattr(model, "_resolve_llm_qk_outside_layers"):
        return model._resolve_llm_qk_outside_layers(spec)
    n = len(model.model.layers)
    if spec.startswith("last"):
        count = int(spec[4:] or "1")
        return list(range(max(0, n - count), n))
    return [int(x) for x in spec.split(",") if x.strip()]


def project_qk_map(model, hidden_states, positions, query_slice, layer_ids):
    """Patch-normalized Q/K preference map, averaged over heads and layers."""
    maps = []
    for layer_idx in layer_ids:
        if layer_idx >= len(hidden_states) - 1:
            continue
        block = model.model.layers[layer_idx]
        attn = block.self_attn
        x = block.input_layernorm(hidden_states[layer_idx])
        query = x[:, query_slice[0]:query_slice[1]]
        image = x[:, positions["img_s"]:positions["img_e"]]
        q_proj = attn.q_proj(query)
        k_proj = attn.k_proj(image)
        n_heads = int(getattr(attn, "num_heads", model.config.num_attention_heads))
        n_kv = int(getattr(attn, "num_key_value_heads", model.config.num_key_value_heads))
        head_dim = int(getattr(attn, "head_dim", q_proj.shape[-1] // n_heads))
        q = q_proj.reshape(q_proj.shape[0], q_proj.shape[1], n_heads, head_dim)
        k = k_proj.reshape(k_proj.shape[0], k_proj.shape[1], n_kv, head_dim)
        if n_kv != n_heads:
            k = k.repeat_interleave(max(n_heads // n_kv, 1), dim=2)[:, :, :n_heads]
        score = torch.einsum("bqhd,bphd->bqhp", q.float(), k.float())
        maps.append((score / math.sqrt(head_dim)).softmax(dim=-1).mean(dim=2))
    return torch.stack(maps).mean(dim=0) if maps else None


def rae_source_preference(model, hidden_states, positions, ovt_positions, ovt_valid, layer_ids):
    """Relative Q/K preference of RAE queries over valid OVTs versus registers."""
    records = []
    for layer_idx in layer_ids:
        if layer_idx >= len(hidden_states) - 1:
            continue
        block = model.model.layers[layer_idx]
        attn = block.self_attn
        x = block.input_layernorm(hidden_states[layer_idx])
        rae = x[:, positions["rae_s"]:positions["rae_e"]]
        reg = x[:, positions["reg_s"]:positions["reg_e"]]
        valid_pos = ovt_positions[0][ovt_valid[0]]
        ovt = x[:, valid_pos]
        sources = torch.cat([ovt, reg], dim=1)
        q_proj, k_proj = attn.q_proj(rae), attn.k_proj(sources)
        n_heads = int(getattr(attn, "num_heads", model.config.num_attention_heads))
        n_kv = int(getattr(attn, "num_key_value_heads", model.config.num_key_value_heads))
        head_dim = int(getattr(attn, "head_dim", q_proj.shape[-1] // n_heads))
        q = q_proj.reshape(1, q_proj.shape[1], n_heads, head_dim)
        k = k_proj.reshape(1, k_proj.shape[1], n_kv, head_dim)
        if n_kv != n_heads:
            k = k.repeat_interleave(max(n_heads // n_kv, 1), dim=2)[:, :, :n_heads]
        probs = (torch.einsum("bqhd,bshd->bqhs", q.float(), k.float()) / math.sqrt(head_dim))
        probs = probs.softmax(dim=-1)
        n_ovt = ovt.shape[1]
        records.append({
            "layer": layer_idx,
            "ovt_mass": float(probs[..., :n_ovt].sum(dim=-1).mean().detach()),
            "register_mass": float(probs[..., n_ovt:].sum(dim=-1).mean().detach()),
        })
    return records


def fixed_recon_loss(model, rae_hidden, target, seed):
    devices = [rae_hidden.device.index] if rae_hidden.is_cuda else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(seed)
        return model._captionslot_compute_diffusion_loss(
            hidden=rae_hidden,
            target_features=target,
        )


def input_gradient_stats(model, baseline, seed):
    for param in model.parameters():
        param.requires_grad_(False)
    inputs = baseline["inputs_embeds"].detach().requires_grad_(True)
    out = model.model(
        inputs_embeds=inputs,
        attention_bias=baseline["attn_bias"],
        use_cache=False,
        return_dict=True,
    )
    pos = baseline["positions"]
    rae = out.last_hidden_state[:, pos["rae_s"]:pos["rae_e"]]
    loss = fixed_recon_loss(model, rae, baseline["gt_siglip"], seed)
    grad = torch.autograd.grad(loss, inputs, retain_graph=False)[0].float().norm(dim=-1)
    valid_pos = baseline["ovt_abs_positions"][0][baseline["ovt_valid_mask"][0]]

    def stats(values):
        return {
            "mean": float(values.mean()),
            "sum": float(values.sum()),
            "max": float(values.max()),
        } if values.numel() else {"mean": 0.0, "sum": 0.0, "max": 0.0}

    return {
        "loss": float(loss.detach()),
        "image": stats(grad[:, pos["img_s"]:pos["img_e"]]),
        "ovt": stats(grad[:, valid_pos]),
        "register": stats(grad[:, pos["reg_s"]:pos["reg_e"]]),
        "rae_query": stats(grad[:, pos["rae_s"]:pos["rae_e"]]),
    }


def heat_overlay(source: Image.Image, values: torch.Tensor, title: str, size=384):
    side = int(round(values.numel() ** 0.5))
    heat = values.float().reshape(1, 1, side, side)
    heat = F.interpolate(heat, size=(size, size), mode="bilinear", align_corners=False)[0, 0]
    heat = (heat - heat.min()) / (heat.max() - heat.min() + 1e-8)
    base = np.asarray(source.resize((size, size), Image.BILINEAR)).astype(np.float32)
    h = heat.detach().cpu().numpy()
    color = np.stack([255 * h, 220 * np.sqrt(h), 40 * (1 - h)], axis=-1)
    out = Image.fromarray(np.clip(0.48 * base + 0.52 * color, 0, 255).astype(np.uint8))
    canvas = Image.new("RGB", (size, size + 34), "white")
    canvas.paste(out, (0, 34))
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 15)
    except Exception:
        font = None
    draw.text((8, 8), title, fill="black", font=font)
    return canvas


def save_map_grid(source, maps, path, prefix, top_k):
    if maps is None or maps.shape[1] == 0:
        return None
    mean_map = maps.mean(dim=1)[0]
    entropy = -(maps.clamp_min(1e-8) * maps.clamp_min(1e-8).log()).sum(dim=-1)[0]
    indices = entropy.argsort()[:min(top_k, maps.shape[1])].tolist()
    tiles = [heat_overlay(source, mean_map, f"{prefix}: mean over registers")]
    tiles.extend(heat_overlay(source, maps[0, idx], f"{prefix}: register {idx}") for idx in indices)
    cols = min(3, len(tiles))
    rows = math.ceil(len(tiles) / cols)
    w, h = tiles[0].size
    grid = Image.new("RGB", (cols * w, rows * h), "white")
    for i, tile in enumerate(tiles):
        grid.paste(tile, ((i % cols) * w, (i // cols) * h))
    grid.save(path)
    return {"path": str(path), "lowest_entropy_registers": indices}


def decode_variants(model, decoder, variants, target_image, target_proc, output_path, seed, guidance):
    images = []
    mean = torch.tensor(target_proc.image_mean)
    std = torch.tensor(target_proc.image_std)
    source = denormalize_images(target_image.float(), mean, std)
    images.append(source[0])
    for name, out in variants.items():
        torch.manual_seed(seed)
        latent = generate_siglip_latent(model, out["rae_hidden"], guidance_level=guidance)
        images.append(decode_to_image(decoder, latent, out["rae_hidden"].device)[0])
    hw = images[1].shape[-2:]
    images = [F.interpolate(x[None], size=hw, mode="bilinear", align_corners=False)[0] for x in images]
    grid = torch.cat(images, dim=2).permute(1, 2, 0).cpu().clamp(0, 1).numpy()
    Image.fromarray((grid * 255).astype(np.uint8)).save(output_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", action="append", required=True, help="label|checkpoint")
    parser.add_argument("--sample_indices", default="0,1020,1407")
    parser.add_argument("--swap_sample_index", type=int, default=None)
    parser.add_argument("--swap_target_object", type=int, default=0)
    parser.add_argument("--swap_source_object", type=int, default=0)
    parser.add_argument(
        "--object_ovt_drop",
        action="store_true",
        help="Also rerun reconstruction after blocking each object's OVT pair.",
    )
    parser.add_argument("--val_jsonl", default="/home/jovyan/PGOT/data/pgot_val.jsonl")
    parser.add_argument("--output_dir", default="/home/jovyan/PGOT/outputs/information_path_diagnostic")
    parser.add_argument("--layers", default="last4")
    parser.add_argument("--top_registers", type=int, default=5)
    parser.add_argument("--compute_gradients", action="store_true")
    parser.add_argument("--decode_recon", action="store_true")
    parser.add_argument("--diffusion_inference_steps", type=int, default=25)
    parser.add_argument("--guidance_scale", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--dtype", choices=["fp32", "bf16"], default="fp32")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s :: %(message)s")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    sample_indices = [int(x) for x in args.sample_indices.split(",") if x.strip()]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32 if args.dtype == "fp32" else torch.bfloat16
    with open(args.val_jsonl) as f:
        raw_samples = [json.loads(line) for line in f]
    all_results = {}

    for label, model_path in map(parse_model, args.model):
        log.info("Loading %s: %s", label, model_path)
        model, tokenizer = _load_model(model_path, dtype, device)
        vt_list = model.get_vision_tower_aux_list()
        image_proc = vt_list[0].image_processor
        target_proc = vt_list[1].image_processor if len(vt_list) > 1 else image_proc
        dataset = Pix2CapPGOTDataset(
            jsonl_path=args.val_jsonl,
            tokenizer=tokenizer,
            image_processor=image_proc,
            target_image_processor=target_proc,
            grid_size=32,
            max_caption_tokens=2048,
            n_ovt_per_object=2,
            max_objects=50,
            panoptic_categories_json="/home/jovyan/data/coco/annotations/panoptic_val2017.json",
        )
        collator = PGOTDataCollator(pad_token_id=tokenizer.pad_token_id)
        decoder = load_rae_decoder(model, device, dtype) if args.decode_recon else None
        if args.decode_recon and args.diffusion_inference_steps != 50:
            from scale_rae.model.diffusion_loss.diffusion import create_diffusion
            inf = model.diff_head.inference_flow
            model.diff_head.inference_flow = create_diffusion(
                str(args.diffusion_inference_steps),
                noise_schedule="linear", use_kl=False, sigma_small=False,
                predict_xstart=False, learn_sigma=False, rescale_learned_sigmas=False,
                diffusion_steps=int(getattr(inf, "diffusion_steps", 1000)),
                input_base_dimension_ratio=float(getattr(inf, "size_ratio", 1.0)),
                diffusion_type="rf", use_loss_weighting=False,
            )
        model_dir = output_dir / label
        model_dir.mkdir(exist_ok=True)
        model_results = []

        for sample_idx in sample_indices:
            batch = collator([dataset[sample_idx]])
            kwargs = dict(
                images=batch["images"],
                target_images=batch["target_images"],
                caption_input_ids=batch["caption_input_ids"],
                caption_attention_mask=batch["caption_attention_mask"],
                ovt_positions_in_caption=batch["ovt_positions_in_caption"],
                ovt_valid_mask=batch["ovt_valid_mask"],
            )
            variants = {
                mode: pgot_forward_eval(
                    model, **kwargs, rae_access_mode=mode,
                    return_hidden_states=(mode == "baseline"),
                )
                for mode in ("baseline", "ovt_only", "register_only", "self_only")
            }
            base = variants["baseline"]
            layer_ids = selected_layers(model, args.layers)
            losses = {}
            shifts = {}
            for mode, out in variants.items():
                losses[mode] = float(fixed_recon_loss(model, out["rae_hidden"], out["gt_siglip"], args.seed))
                shifts[mode] = {
                    "cosine_to_baseline": float(F.cosine_similarity(
                        base["rae_hidden"].float().flatten(1),
                        out["rae_hidden"].float().flatten(1),
                    ).mean()),
                    "relative_l2_to_baseline": float(
                        (out["rae_hidden"].float() - base["rae_hidden"].float()).norm()
                        / base["rae_hidden"].float().norm().clamp_min(1e-8)
                    ),
                }

            source = Image.open(raw_samples[sample_idx]["image_path"]).convert("RGB")
            reg_dot = base["reg_logits"].float().softmax(dim=-1)
            reg_qk = project_qk_map(
                model, base["hidden_states"], base["positions"],
                (base["positions"]["reg_s"], base["positions"]["reg_e"]), layer_ids,
            )
            dot_info = save_map_grid(
                source, reg_dot, model_dir / f"sample{sample_idx}_register_dot.png",
                "final register-patch dot", args.top_registers,
            )
            qk_info = save_map_grid(
                source, reg_qk, model_dir / f"sample{sample_idx}_register_llm_qk.png",
                f"LLM QK {args.layers}", args.top_registers,
            )
            record = {
                "sample_index": sample_idx,
                "image_id": raw_samples[sample_idx]["image_id"],
                "recon_loss": losses,
                "rae_hidden_shift": shifts,
                "rae_source_preference": rae_source_preference(
                    model, base["hidden_states"], base["positions"],
                    base["ovt_abs_positions"], base["ovt_valid_mask"], layer_ids,
                ),
                "register_dot_maps": dot_info,
                "register_llm_qk_maps": qk_info,
            }
            if args.object_ovt_drop:
                n_valid_ovt = int(base["ovt_valid_mask"][0].sum().item())
                object_drop = []
                for obj_idx in range(n_valid_ovt // 2):
                    dropped = pgot_forward_eval(
                        model,
                        **kwargs,
                        rae_block_ovt_indices=(2 * obj_idx, 2 * obj_idx + 1),
                    )
                    drop_loss = float(
                        fixed_recon_loss(model, dropped["rae_hidden"], dropped["gt_siglip"], args.seed)
                    )
                    object_drop.append({
                        "object_index": obj_idx,
                        "recon_loss": drop_loss,
                        "delta_from_baseline": drop_loss - losses["baseline"],
                        "relative_l2_to_baseline": float(
                            (dropped["rae_hidden"].float() - base["rae_hidden"].float()).norm()
                            / base["rae_hidden"].float().norm().clamp_min(1e-8)
                        ),
                    })
                record["object_ovt_drop"] = object_drop
            decode_inputs = dict(variants)
            if args.swap_sample_index is not None:
                swap_batch = collator([dataset[args.swap_sample_index]])
                swap_out = pgot_forward_eval(
                    model,
                    images=swap_batch["images"],
                    target_images=swap_batch["target_images"],
                    caption_input_ids=swap_batch["caption_input_ids"],
                    caption_attention_mask=swap_batch["caption_attention_mask"],
                    ovt_positions_in_caption=swap_batch["ovt_positions_in_caption"],
                    ovt_valid_mask=swap_batch["ovt_valid_mask"],
                )
                swapped_rae, _ = ovt_swap_inference(
                    model,
                    base,
                    swap_out,
                    [(args.swap_target_object, args.swap_source_object)],
                    2,
                )
                record["object_swap"] = {
                    "source_sample_index": args.swap_sample_index,
                    "target_object_index": args.swap_target_object,
                    "source_object_index": args.swap_source_object,
                    "recon_loss": float(fixed_recon_loss(model, swapped_rae, base["gt_siglip"], args.seed)),
                    "relative_l2_to_baseline": float(
                        (swapped_rae.float() - base["rae_hidden"].float()).norm()
                        / base["rae_hidden"].float().norm().clamp_min(1e-8)
                    ),
                }
                decode_inputs["object_swap"] = {
                    "rae_hidden": swapped_rae,
                }
            if args.compute_gradients:
                record["input_gradient"] = input_gradient_stats(model, base, args.seed)
            if args.decode_recon:
                recon_path = model_dir / f"sample{sample_idx}_recon_ablation.png"
                decode_variants(
                    model, decoder, decode_inputs, batch["target_images"].to(device),
                    target_proc, recon_path, args.seed, args.guidance_scale,
                )
                record["recon_ablation_image"] = str(recon_path)
                record["recon_ablation_order"] = ["source", *decode_inputs.keys()]
            model_results.append(record)
            log.info("%s sample=%d losses=%s", label, sample_idx, losses)

        all_results[label] = model_results
        with (model_dir / "summary.json").open("w") as f:
            json.dump(model_results, f, indent=2)
        del decoder, dataset, tokenizer, model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    with (output_dir / "summary.json").open("w") as f:
        json.dump(all_results, f, indent=2)
    print(output_dir / "summary.json")


if __name__ == "__main__":
    main()
