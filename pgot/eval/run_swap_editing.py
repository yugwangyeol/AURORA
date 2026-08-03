"""PGOT OVT-swap editing inference (qualitative).

For N pairs (A, B) of validation images, swaps OVT_i(A) ← OVT_j(B) and
generates an edited image. Saves a grid:

   [Source A | A→Decoder (GT-SigLIP recon, sanity) | A self-recon (no swap) | A-with-OVT(B) | Source B]

Usage:
    PYTHONPATH=/home/jovyan/PGOT python -m pgot.eval.run_swap_editing \
        --model_path /home/jovyan/PGOT/checkpoints/pgot_main_v3/checkpoint-14000 \
        --val_jsonl  /home/jovyan/PGOT/data/pgot_val.jsonl \
        --output_dir /home/jovyan/PGOT/outputs/edit_v3_14k \
        --n_pairs 16 --diffusion_inference_steps 25 --guidance_scale 1.0
"""
import argparse
import json
import logging
import os
import random
import sys
from typing import List

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, "/home/jovyan/PGOT")
from pgot.constants import OVT_TOKEN, SCENE_END_TOKEN, NEW_SPECIAL_TOKENS
from pgot.model.pgot_qwen2 import PGOTQwen2ForCausalLM
from pgot.train.pgot_dataset import Pix2CapPGOTDataset, PGOTDataCollator
from pgot.eval.pgot_inference import (
    generate_siglip_latent,
    ovt_swap_all_layers_inference,
    pgot_forward_eval,
)
from pgot.eval.visualize_ovt_overlays import _load_model

from transformers import AutoTokenizer, AutoConfig

logging.basicConfig(level=logging.INFO,
                    format="[%(asctime)s] %(levelname)s :: %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("pgot.swap_editing")


def load_rae_decoder(pgot_model, device, dtype=torch.float32):
    from huggingface_hub import hf_hub_download
    from scale_rae.model.multimodal_decoder import MultimodalDecoder
    repo_id = "nyu-visionx/siglip2_decoder"
    vision_towers = list(getattr(pgot_model.config, "mm_vision_tower_aux_list",
                                 ["google/siglip2-so400m-patch14-224"]))
    encoder_path = vision_towers[1] if len(vision_towers) > 1 else vision_towers[0]
    encoder_path = encoder_path.split("-interp")[0]
    num_patches = int(getattr(pgot_model.config, "diffusion_target_token_len", 256))
    log.info(f"Loading RAE decoder ({repo_id}) for encoder={encoder_path} P={num_patches}")
    config_path = hf_hub_download(repo_id=repo_id, filename="config.json")
    ckpt_path = hf_hub_download(repo_id=repo_id, filename="model.pt")
    dec = MultimodalDecoder(
        pretrained_encoder_path=encoder_path,
        general_decoder_config=config_path,
        num_patches=num_patches,
        drop_cls_token=True,
        decoder_path=ckpt_path,
    )
    dec.eval()
    dec.to(device=device, dtype=dtype)
    return dec


@torch.no_grad()
def decode_to_image(decoder, generated, device):
    if generated is None:
        return torch.empty(0, 3, 1, 1, device=device)
    decoder_dtype = next(decoder.parameters()).dtype
    if generated.dtype != decoder_dtype:
        generated = generated.to(dtype=decoder_dtype)
    if hasattr(decoder, "image_mean"):
        decoder.image_mean = decoder.image_mean.to(device=device, dtype=decoder_dtype)
        decoder.image_std = decoder.image_std.to(device=device, dtype=decoder_dtype)
    empty_cls = torch.zeros((generated.shape[0], 1, generated.shape[-1]),
                            device=device, dtype=decoder_dtype)
    feats = torch.cat([empty_cls, generated], dim=1)
    recon = decoder(feats)
    recon = torch.nan_to_num(recon, nan=0.0, posinf=1.0, neginf=0.0)
    return recon.clamp(0.0, 1.0).detach().float()


def denormalize(images, mean, std):
    mean = mean.view(1, -1, 1, 1).to(images.device).to(images.dtype)
    std = std.view(1, -1, 1, 1).to(images.device).to(images.dtype)
    return (images * std + mean).clamp(0.0, 1.0)


def _labeled_tile(image_chw: torch.Tensor, label: str) -> Image.Image:
    array = (
        image_chw.detach()
        .float()
        .cpu()
        .clamp(0, 1)
        .permute(1, 2, 0)
        .numpy()
        * 255
    ).astype(np.uint8)
    tile = Image.fromarray(array)
    canvas = Image.new("RGB", (tile.width, tile.height + 30), "white")
    canvas.paste(tile, (0, 30))
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 14)
    except Exception:
        font = None
    draw.text((6, 6), label, fill=(0, 0, 0), font=font)
    return canvas


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", required=True)
    p.add_argument("--val_jsonl", default="/home/jovyan/PGOT/data/pgot_val.jsonl")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--n_pairs", type=int, default=16)
    p.add_argument("--n_ovt_per_object", type=int, default=2)
    p.add_argument("--max_objects", type=int, default=50)
    p.add_argument("--max_caption_tokens", type=int, default=2048)
    p.add_argument("--grid_size", type=int, default=32)
    p.add_argument(
        "--image_preprocess_mode",
        choices=["default", "coda_center_crop"],
        default="default",
    )
    p.add_argument("--coda_crop_size", type=int, default=512)
    p.add_argument("--diffusion_inference_steps", type=int, default=25)
    p.add_argument("--guidance_scale", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--obj_a_idx", type=int, default=0, help="which object (0-based) in A to replace")
    p.add_argument("--obj_b_idx", type=int, default=0, help="which object (0-based) in B to inject")
    # Manual pair selection (for qualitative inspection of a single pair).
    # Either pass --image_id_a / --image_id_b (COCO image ids in pgot_val.jsonl)
    # OR --idx_a / --idx_b (dataset row indices). If set, n_pairs is forced to 1.
    p.add_argument("--image_id_a", type=int, default=None)
    p.add_argument("--image_id_b", type=int, default=None)
    p.add_argument("--idx_a", type=int, default=None)
    p.add_argument("--idx_b", type=int, default=None)
    # Multi-pair manual mode: "imgA:imgB:objA:objB,imgA:imgB:objA:objB,..."
    p.add_argument("--pairs_csv", type=str, default=None,
                   help='Comma-separated list of "image_id_A:image_id_B:obj_a:obj_b" tuples.')
    p.add_argument("--dtype", choices=["fp32", "bf16"], default="fp32")
    args = p.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32 if args.dtype == "fp32" else torch.bfloat16

    # ---- Load model
    log.info(f"Loading model: {args.model_path}")
    model, tokenizer = _load_model(args.model_path, dtype=dtype, device=device)
    log.info("Model loaded.")

    # ---- Decoder
    decoder = load_rae_decoder(model, device=device, dtype=dtype)

    # ---- Patch diff_head inference steps
    if args.diffusion_inference_steps and args.diffusion_inference_steps != 50:
        from scale_rae.model.diffusion_loss.diffusion import create_diffusion
        inf = model.diff_head.inference_flow
        size_ratio = float(getattr(inf, "size_ratio", 1.0))
        d_steps = int(getattr(inf, "diffusion_steps", 1000))
        model.diff_head.inference_flow = create_diffusion(
            str(args.diffusion_inference_steps),
            noise_schedule="linear", use_kl=False, sigma_small=False,
            predict_xstart=False, learn_sigma=False, rescale_learned_sigmas=False,
            diffusion_steps=d_steps, input_base_dimension_ratio=size_ratio,
            diffusion_type="rf", use_loss_weighting=False,
        )
        log.info(f"[diff] patched inference_flow to {args.diffusion_inference_steps} steps")

    # ---- Dataset
    vt_list = model.get_vision_tower_aux_list()
    image_proc = vt_list[0].image_processor
    target_proc = vt_list[1].image_processor if len(vt_list) > 1 else image_proc
    dataset = Pix2CapPGOTDataset(
        jsonl_path=args.val_jsonl,
        tokenizer=tokenizer,
        image_processor=image_proc,
        target_image_processor=target_proc,
        grid_size=args.grid_size,
        max_caption_tokens=args.max_caption_tokens,
        n_ovt_per_object=args.n_ovt_per_object,
        max_objects=args.max_objects,
        panoptic_categories_json="/home/jovyan/data/coco/annotations/panoptic_val2017.json",
        image_preprocess_mode=args.image_preprocess_mode,
        coda_crop_size=args.coda_crop_size,
    )
    log.info(f"Dataset: {len(dataset)} samples")

    # ---- Sample pairs
    def _resolve_idx_from_image_id(image_id):
        for i, s in enumerate(dataset.samples):
            if int(s["image_id"]) == int(image_id):
                return i
        return None

    # Multi-pair mode (pairs_csv): runs all listed pairs sequentially, model loaded once.
    pair_overrides = None
    if args.pairs_csv:
        pair_overrides = []
        for chunk in args.pairs_csv.split(","):
            parts = chunk.strip().split(":")
            if len(parts) != 4:
                log.error(f"Bad pairs_csv chunk: {chunk}  (expected 'imgA:imgB:objA:objB')")
                return
            ia, ib, oa, ob = map(int, parts)
            i_a = _resolve_idx_from_image_id(ia)
            i_b = _resolve_idx_from_image_id(ib)
            if i_a is None or i_b is None:
                log.error(f"image_id not found: A={ia} (idx={i_a}), B={ib} (idx={i_b})")
                return
            pair_overrides.append((i_a, i_b, oa, ob))
        log.info(f"[pairs_csv] {len(pair_overrides)} pairs queued")
        pair_indices = [(p[0], p[1]) for p in pair_overrides]
        n_pairs = len(pair_overrides)

    manual_pair = (args.image_id_a is not None and args.image_id_b is not None) or \
                  (args.idx_a is not None and args.idx_b is not None)
    if manual_pair and pair_overrides is None:
        if args.image_id_a is not None and args.image_id_b is not None:
            i_a = _resolve_idx_from_image_id(args.image_id_a)
            i_b = _resolve_idx_from_image_id(args.image_id_b)
            if i_a is None or i_b is None:
                log.error(f"image_id not found: A={args.image_id_a} (idx={i_a}), B={args.image_id_b} (idx={i_b})")
                return
        else:
            i_a, i_b = int(args.idx_a), int(args.idx_b)
        log.info(f"[manual pair] idx_A={i_a} (image_id={dataset.samples[i_a]['image_id']}), "
                 f"idx_B={i_b} (image_id={dataset.samples[i_b]['image_id']})")
        # Validate that both have enough objects
        for label, idx, obj_idx in [("A", i_a, args.obj_a_idx), ("B", i_b, args.obj_b_idx)]:
            n_objects = len(dataset.samples[idx].get("segments", []))
            if n_objects <= obj_idx:
                log.error(f"{label} (image_id={dataset.samples[idx]['image_id']}) "
                          f"has only {n_objects} object(s); "
                          f"requested obj_{label.lower()}_idx={obj_idx}.")
                return
        pair_indices = [(i_a, i_b)]
        n_pairs = 1
    elif pair_overrides is None:
        min_objs = max(args.obj_a_idx, args.obj_b_idx) + 1
        eligible = [
            i
            for i in range(len(dataset))
            if len(dataset.samples[i].get("segments", [])) >= min_objs
        ]
        log.info(f"Eligible samples (>= {min_objs} objects): {len(eligible)}")
        if len(eligible) < 2:
            log.error("Not enough eligible samples.")
            return
        random.shuffle(eligible)
        n_pairs = min(args.n_pairs, len(eligible) // 2)
        pair_indices = [(eligible[2 * i], eligible[2 * i + 1]) for i in range(n_pairs)]

    # ---- Reconstruction normalization stats (for denormalizing target_image)
    t_mean = torch.tensor(target_proc.image_mean)
    t_std = torch.tensor(target_proc.image_std)

    collator = PGOTDataCollator(pad_token_id=tokenizer.pad_token_id)

    metadata = []
    log.info(f"Running {n_pairs} swap edits...")
    for pair_i, (i_a, i_b) in enumerate(pair_indices):
        # Per-pair obj indices (multi-pair mode) or global fallback
        if pair_overrides is not None:
            _oa, _ob = pair_overrides[pair_i][2], pair_overrides[pair_i][3]
        else:
            _oa, _ob = args.obj_a_idx, args.obj_b_idx
        sample_a = dataset[i_a]
        sample_b = dataset[i_b]
        batch_a = collator([sample_a])
        batch_b = collator([sample_b])

        out_a = pgot_forward_eval(
            model,
            images=batch_a["images"].to(device),
            target_images=batch_a["target_images"].to(device),
            caption_input_ids=batch_a["caption_input_ids"].to(device),
            caption_attention_mask=batch_a["caption_attention_mask"].to(device),
            ovt_positions_in_caption=batch_a["ovt_positions_in_caption"].to(device),
            ovt_valid_mask=batch_a["ovt_valid_mask"].to(device),
            return_hidden_states=True,
        )
        out_b = pgot_forward_eval(
            model,
            images=batch_b["images"].to(device),
            target_images=batch_b["target_images"].to(device),
            caption_input_ids=batch_b["caption_input_ids"].to(device),
            caption_attention_mask=batch_b["caption_attention_mask"].to(device),
            ovt_positions_in_caption=batch_b["ovt_positions_in_caption"].to(device),
            ovt_valid_mask=batch_b["ovt_valid_mask"].to(device),
            return_hidden_states=True,
        )

        # 1) A self-recon (no swap, baseline)
        rae_A = out_a["rae_hidden"]
        torch.manual_seed(args.seed)
        gen_A_self = generate_siglip_latent(model, rae_A, guidance_level=args.guidance_scale)
        img_A_self = decode_to_image(decoder, gen_A_self, device)  # (1, 3, H, W)

        # 2) A with OVT_obj_a <- OVT_obj_b (from B)
        rae_mixed, _ = ovt_swap_all_layers_inference(
            model, out_A=out_a, out_B=out_b,
            swap_pairs=[(_oa, _ob)],
            n_ovt_per_object=args.n_ovt_per_object,
        )
        torch.manual_seed(args.seed)
        gen_mixed = generate_siglip_latent(model, rae_mixed, guidance_level=args.guidance_scale)
        img_mixed = decode_to_image(decoder, gen_mixed, device)

        # 3) GT-SigLIP -> decoder (sanity / oracle)
        gt_dec_A = decode_to_image(decoder, out_a["gt_siglip"], device)

        # Source images (denormalized)
        src_a = denormalize(batch_a["target_images"].to(device).float(), t_mean, t_std)
        src_b = denormalize(batch_b["target_images"].to(device).float(), t_mean, t_std)

        # Resize all to common H,W
        target_hw = img_A_self.shape[-2:]
        def _match(x):
            if x.shape[-2:] != target_hw:
                x = F.interpolate(x, size=target_hw, mode="bilinear", align_corners=False)
            return x
        src_a = _match(src_a)
        src_b = _match(src_b)
        gt_dec_A = _match(gt_dec_A)

        # Build a labeled 5-up horizontal grid with fixed-noise reconstructions.
        category_a = dataset.samples[i_a]["segments"][_oa]["category"]
        category_b = dataset.samples[i_b]["segments"][_ob]["category"]
        tiles = [
            _labeled_tile(src_a[0], f"Source A (replace {category_a})"),
            _labeled_tile(gt_dec_A[0], "GT SigLIP decoder"),
            _labeled_tile(img_A_self[0], "A self-reconstruction"),
            _labeled_tile(
                img_mixed[0],
                f"Swap: {category_a} OVT <- {category_b} OVT",
            ),
            _labeled_tile(src_b[0], f"Source B (donor {category_b})"),
        ]
        grid = Image.new(
            "RGB",
            (sum(tile.width for tile in tiles), max(tile.height for tile in tiles)),
            "white",
        )
        left = 0
        for tile in tiles:
            grid.paste(tile, (left, 0))
            left += tile.width
        out_png = os.path.join(args.output_dir, f"pair_{pair_i:03d}_A{sample_a['image_id']}_B{sample_b['image_id']}_swapA{_oa}fromB{_ob}.png")
        grid.save(out_png)

        # Caption decoding for metadata
        def first_chunk_label(samp, idx):
            segs = samp.get("segments", [])
            if idx < len(segs):
                return f"{segs[idx]['category'].capitalize()} {segs[idx].get('category_index', idx+1)}"
            return f"obj{idx}"

        meta = {
            "pair": pair_i,
            "image_id_A": sample_a["image_id"],
            "image_id_B": sample_b["image_id"],
            "n_objects_A": sample_a["n_objects"],
            "n_objects_B": sample_b["n_objects"],
            "swap": {"obj_A_idx": _oa, "obj_B_idx": _ob,
                     "obj_A": first_chunk_label(dataset.samples[i_a], _oa),
                     "obj_B": first_chunk_label(dataset.samples[i_b], _ob)},
            "swap_mode": "all_layers",
            "guidance": args.guidance_scale,
            "image": out_png,
        }
        metadata.append(meta)
        log.info(f"  [{pair_i+1}/{n_pairs}] A={sample_a['image_id']}({meta['swap']['obj_A']}) "
                 f"<- B={sample_b['image_id']}({meta['swap']['obj_B']}) -> {os.path.basename(out_png)}")

    with open(os.path.join(args.output_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)
    log.info(f"Saved {n_pairs} edits to {args.output_dir}")
    log.info("Grid columns: [Source A | GT-SigLIP→Decoder | A self-recon | A with OVT(B) | Source B]")


if __name__ == "__main__":
    main()
