"""Visualize PGOT OVT/readout behavior on one validation image.

Example:
    python -m pgot.eval.visualize_ovt_overlays \
      --sample_index 0 \
      --gt_source coco_instance \
      --model 'V3|/path/to/checkpoint|threshold|mean' \
      --model 'V12|/path/to/checkpoint|ovt_owner|mean'
"""

import argparse
import gc
import glob
import json
import logging
import os
import re
import sys
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, "/home/jovyan/PGOT")
from pgot.constants import NEW_SPECIAL_TOKENS, OVT_TOKEN, SCENE_END_TOKEN
from pgot.eval.pgot_inference import pgot_forward_eval
from pgot.eval.pgot_metrics import (
    build_pred_mask_spatial_readout,
    ovt_logits_to_pred_mask,
    preproc_masks_overlap,
)
from pgot.eval.run_eval import CocoInstanceMaskCache, load_gt_panoptic_mask, load_thing_categories
from pgot.model.pgot_qwen2 import PGOTQwen2ForCausalLM
from pgot.model.pgot_utils import (
    build_pred_mask_competition_eval,
    build_pred_mask_null_bg_eval,
    build_pred_mask_ovt_owner_eval,
)
from pgot.train.pgot_dataset import PGOTDataCollator, Pix2CapPGOTDataset
from transformers import AutoConfig, AutoTokenizer

log = logging.getLogger("pgot.viz")


def _parse_model_spec(spec: str):
    parts = spec.split("|")
    if len(parts) not in (2, 4):
        raise ValueError("--model must be 'label|path' or 'label|path|readout|merge'")
    label, path = parts[0], parts[1]
    readout = parts[2] if len(parts) == 4 else "threshold"
    merge = parts[3] if len(parts) == 4 else "mean"
    return {"label": label, "path": path, "readout": readout, "merge": merge}


def _load_model(model_path: str, dtype: torch.dtype, device: str):
    config = AutoConfig.from_pretrained(model_path)
    model = PGOTQwen2ForCausalLM.from_pretrained(
        model_path,
        config=config,
        torch_dtype=dtype,
        ignore_mismatched_sizes=True,
    )
    model.config.use_cache = False
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False, padding_side="right")

    import safetensors.torch as safe_torch
    has_lora = False
    for shard in sorted(glob.glob(os.path.join(model_path, "*.safetensors"))):
        with safe_torch.safe_open(shard, framework="pt", device="cpu") as f:
            if any("lora_" in k for k in f.keys()):
                has_lora = True
                break
    if has_lora:
        from peft import LoraConfig, inject_adapter_in_model
        lora_cfg = LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.0,
            bias="none",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            task_type="CAUSAL_LM",
        )
        inject_adapter_in_model(lora_cfg, model, adapter_name="default")
        state = {}
        for shard in sorted(glob.glob(os.path.join(model_path, "*.safetensors"))):
            with safe_torch.safe_open(shard, framework="pt", device="cpu") as f:
                for k in f.keys():
                    state[k] = f.get_tensor(k)
        missing, unexpected = model.load_state_dict(state, strict=False)
        log.info("[%s] LoRA reload missing=%d unexpected=%d", model_path, len(missing), len(unexpected))

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = "<|endoftext|>"
        tokenizer.pad_token_id = 151643
    if OVT_TOKEN not in tokenizer.get_vocab():
        tokenizer.add_special_tokens({"additional_special_tokens": NEW_SPECIAL_TOKENS})
        model.resize_token_embeddings(len(tokenizer))

    parsed_towers = getattr(config, "mm_vision_tower_aux_list", None) or json.loads(
        getattr(config, "vision_tower_aux_list", '["google/siglip2-so400m-patch14-224"]')
    )
    parsed_token_lens = getattr(config, "mm_vision_tower_aux_token_len_list", None) or [256]
    vt_args = SimpleNamespace(
        vision_tower_aux_list=parsed_towers,
        vision_tower_aux_token_len_list=parsed_token_lens,
        mm_vision_select_layer=-1,
        mm_vision_select_feature="patch",
        mm_projector_type="mlp2x_gelu",
        mm_use_im_start_end=True,
        mm_use_im_patch_token=False,
        unfreeze_mm_vision_tower=False,
        vision_hidden_size=1024,
        connector_only=True,
        pretrain_mm_mlp_adapter=None,
        pretrain_adapter_and_vision_head=None,
        diffusion_norm_stats_path=getattr(config, "diffusion_norm_stats_path", None),
    )
    model.get_model().initialize_vision_modules(model_args=vt_args, fsdp=None)
    model.load_vision_head(model_args=vt_args)
    for vt in model.get_vision_tower_aux_list():
        vt.to(dtype=dtype, device=device)

    model.pgot_ovt_token_id = tokenizer.convert_tokens_to_ids(OVT_TOKEN)
    model.pgot_scene_end_token_id = tokenizer.convert_tokens_to_ids(SCENE_END_TOKEN)
    blocks = {
        "pgot_system_prefix_ids": "<|im_start|>system\nYou are a vision assistant that describes scenes with grounded objects.",
        "pgot_system_suffix_ids": "<|im_end|>\n",
        "pgot_user_prefix_ids": "<|im_start|>user\n",
        "pgot_user_suffix_ids": "\nDescribe all objects and regions in this scene with grounded tokens.<|im_end|>\n",
        "pgot_assistant_prefix_ids": "<|im_start|>assistant\n",
        "pgot_assistant_suffix_ids": "<|im_end|>",
    }
    for attr, txt in blocks.items():
        setattr(model, attr, tokenizer.encode(txt, add_special_tokens=False))

    model.to(device=device, dtype=dtype)
    model.eval()
    return model, tokenizer


def _palette(n: int):
    colors = []
    for i in range(max(n, 1)):
        hue = (i * 0.61803398875) % 1.0
        rgb = np.array(_hsv_to_rgb(hue, 0.72, 0.95)) * 255
        colors.append(rgb.astype(np.uint8))
    return np.stack(colors, axis=0)


def _hsv_to_rgb(h, s, v):
    i = int(h * 6.0)
    f = h * 6.0 - i
    p = v * (1.0 - s)
    q = v * (1.0 - f * s)
    t = v * (1.0 - (1.0 - f) * s)
    i = i % 6
    vals = [(v, t, p), (q, v, p), (p, v, t), (p, q, v), (t, p, v), (v, p, q)]
    return vals[i]


def _add_title(img: Image.Image, title: str, height: int = 32):
    out = Image.new("RGB", (img.width, img.height + height), (255, 255, 255))
    out.paste(img, (0, height))
    draw = ImageDraw.Draw(out)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 16)
    except Exception:
        font = None
    draw.text((8, 7), title[:80], fill=(0, 0, 0), font=font)
    return out


def _safe_name(text: str) -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text.strip())
    return text.strip("_") or "model"


def _load_font(size: int):
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except Exception:
        return None


def _mask_np(mask: torch.Tensor, size=None) -> np.ndarray:
    m = mask.detach().cpu().numpy().astype(np.int64)
    if size is not None and tuple(m.shape[:2]) != tuple(size):
        m_img = Image.fromarray(m.astype(np.uint16))
        m_img = m_img.resize((size[1], size[0]), Image.NEAREST)
        m = np.asarray(m_img).astype(np.int64)
    return m


def _source_from_target_tensor(image: torch.Tensor, processor) -> Image.Image:
    x = image.detach().cpu().float()
    mean = torch.tensor(getattr(processor, "image_mean", [0.5, 0.5, 0.5]), dtype=torch.float32).view(-1, 1, 1)
    std = torch.tensor(getattr(processor, "image_std", [0.5, 0.5, 0.5]), dtype=torch.float32).view(-1, 1, 1)
    x = (x * std + mean).clamp(0.0, 1.0)
    arr = (x.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
    return Image.fromarray(arr)


def _color_mask_overlay(source: Image.Image, mask: torch.Tensor, alpha: float = 0.55, size: int = None):
    if size is not None:
        src_img = source.convert("RGB").resize((size, size), Image.BILINEAR)
    else:
        src_img = source.convert("RGB")
    src = np.asarray(src_img).astype(np.float32)
    m = _mask_np(mask, size=(src.shape[0], src.shape[1]))
    colors = _palette(int(m.max()) + 1)
    color = np.zeros_like(src)
    fg = m > 0
    if fg.any():
        color[fg] = colors[(m[fg] - 1) % len(colors)]
    out = src.copy()
    out[fg] = src[fg] * (1.0 - alpha) + color[fg] * alpha
    return Image.fromarray(out.clip(0, 255).astype(np.uint8)), m, colors


def _mask_overlay(source: Image.Image, mask: torch.Tensor, title: str, alpha: float = 0.55):
    overlay, _, _ = _color_mask_overlay(source, mask, alpha=alpha)
    return _add_title(overlay, title)


def _large_mask_overlay(
    source: Image.Image,
    mask: torch.Tensor,
    title: str,
    labels,
    alpha: float = 0.52,
    size: int = 640,
    max_legend_items: int = 18,
):
    overlay, m, colors = _color_mask_overlay(source, mask, alpha=alpha, size=size)
    present = [int(i) for i in np.unique(m) if int(i) > 0]
    present = present[:max_legend_items]

    title_h = 46
    legend_rows = int(np.ceil(len(present) / 3.0)) if present else 0
    legend_h = max(0, legend_rows * 25 + 16)
    canvas = Image.new("RGB", (overlay.width, overlay.height + title_h + legend_h), (255, 255, 255))
    canvas.paste(overlay, (0, title_h))

    draw = ImageDraw.Draw(canvas)
    title_font = _load_font(20)
    legend_font = _load_font(13)
    draw.text((14, 12), title[:95], fill=(0, 0, 0), font=title_font)

    if present:
        x0 = 14
        y0 = title_h + overlay.height + 10
        col_w = max(160, overlay.width // 3)
        for idx, mask_id in enumerate(present):
            row = idx // 3
            col = idx % 3
            x = x0 + col * col_w
            y = y0 + row * 25
            color = tuple(int(v) for v in colors[(mask_id - 1) % len(colors)])
            draw.rectangle((x, y + 3, x + 15, y + 18), fill=color)
            if labels and mask_id - 1 < len(labels):
                name = labels[mask_id - 1]
            else:
                name = f"region {mask_id}"
            draw.text((x + 21, y), f"{mask_id}: {name}"[:24], fill=(20, 20, 20), font=legend_font)
    return canvas


def _segment_labels(segments, thing_categories=None, thing_only: bool = False):
    counts = {}
    labels = []
    for seg in segments:
        cat = seg.get("category", "object")
        if thing_only and thing_categories is not None and cat not in thing_categories:
            continue
        counts[cat] = counts.get(cat, 0) + 1
        labels.append(f"{cat} {counts[cat]}")
    return labels


def _gt_eval_mask(gt: torch.Tensor, overlap: torch.Tensor | None) -> torch.Tensor:
    if overlap is None:
        return gt
    gt_eval, _ = preproc_masks_overlap(gt, gt.clone(), overlap)
    return gt_eval


def _pred_labels_for_readout(args, spec, raw, thing_categories):
    compact_thing_readouts = {"spatial", "competition", "nullbg", "ovt_owner"}
    if args.gt_source == "coco_instance" and spec["readout"] in compact_thing_readouts:
        return _segment_labels(raw["segments"], thing_categories=thing_categories, thing_only=True)
    return _segment_labels(raw["segments"])


def _heat_overlay(source: Image.Image, heat: torch.Tensor, title: str):
    src = np.asarray(source.convert("RGB")).astype(np.float32)
    h = heat.detach().cpu().float()
    h = h - h.min()
    h = h / h.max().clamp_min(1e-6)
    h_np = h.numpy()
    color = np.zeros_like(src)
    color[..., 0] = 255.0
    color[..., 1] = 40.0
    color[..., 2] = 40.0
    out = src * (1.0 - 0.62 * h_np[..., None]) + color * (0.62 * h_np[..., None])
    return _add_title(Image.fromarray(out.clip(0, 255).astype(np.uint8)), title)


def _concat_grid(images, cols: int):
    if not images:
        raise ValueError("no images to concatenate")
    w = max(i.width for i in images)
    h = max(i.height for i in images)
    rows = int(np.ceil(len(images) / float(cols)))
    canvas = Image.new("RGB", (cols * w, rows * h), (245, 245, 245))
    for idx, img in enumerate(images):
        x = (idx % cols) * w
        y = (idx // cols) * h
        canvas.paste(img, (x, y))
    return canvas


def _build_pred(args, spec, out, batch, samples_iter, thing_categories):
    valid = out["ovt_valid_mask"].clone()
    ovt_is_thing = batch["ovt_is_thing"].to(valid.device, dtype=torch.bool)
    if thing_categories is not None:
        for b in range(valid.shape[0]):
            segs = samples_iter[args.sample_index]["segments"]
            for k, seg in enumerate(segs):
                s = k * args.n_ovt_per_object
                e = s + args.n_ovt_per_object
                if e <= ovt_is_thing.shape[1] and seg["category"] in thing_categories:
                    ovt_is_thing[b, s:e] = True

    readout = spec["readout"]
    if readout == "threshold":
        valid_for_threshold = valid
        if args.gt_source == "coco_instance":
            valid_for_threshold = valid & ovt_is_thing
        return ovt_logits_to_pred_mask(
            out["ovt_logits"],
            valid_for_threshold,
            target_size=args.eval_size,
            n_ovt_per_object=args.n_ovt_per_object,
            patch_grid=args.grid_size,
            bg_threshold=args.bg_threshold,
        )
    if readout == "spatial":
        return build_pred_mask_spatial_readout(
            out["ovt_logits"],
            valid,
            target_size=args.eval_size,
            n_ovt_per_object=args.n_ovt_per_object,
            patch_grid=args.grid_size,
            merge=spec["merge"],
            temp=args.spatial_temperature,
            ovt_is_thing=ovt_is_thing,
            map_stuff_to_bg=(args.gt_source == "coco_instance"),
        )
    if readout == "nullbg":
        return build_pred_mask_null_bg_eval(
            out["ovt_logits"],
            out["null_bg_logits"],
            valid,
            ovt_is_thing,
            target_size=args.eval_size,
            n_ovt_per_object=args.n_ovt_per_object,
            patch_grid=args.grid_size,
            merge=spec["merge"],
        )
    if readout == "ovt_owner":
        return build_pred_mask_ovt_owner_eval(
            ovt_object_probs=out["ovt_object_probs"],
            ovt_void_probs=out["ovt_void_probs"],
            ovt_valid_mask=valid,
            ovt_is_thing=ovt_is_thing,
            target_size=args.eval_size,
            n_ovt_per_object=args.n_ovt_per_object,
            patch_grid=args.grid_size,
            map_stuff_to_bg=(args.gt_source == "coco_instance"),
        )
    return build_pred_mask_competition_eval(
        out["ovt_logits"],
        out["reg_logits"],
        valid,
        ovt_is_thing,
        target_size=args.eval_size,
        n_ovt_per_object=args.n_ovt_per_object,
        patch_grid=args.grid_size,
        merge=spec["merge"],
    )


def _object_maps(args, spec, out):
    if spec["readout"] == "ovt_owner":
        obj = out["ovt_object_probs"].float()
        B, K, P = obj.shape
        side = args.grid_size
        obj_2d = obj.reshape(B, K, side, side)
        up = F.interpolate(
            obj_2d,
            size=(args.eval_size, args.eval_size),
            mode="bilinear",
            align_corners=False,
        )
        return up[0]

    logits = out["ovt_logits"].float()
    B, M, P = logits.shape
    n = args.n_ovt_per_object
    K = M // n
    logits = logits[:, : K * n].reshape(B, K, n, P)
    if spec["readout"] == "spatial":
        source = torch.softmax(logits / max(float(args.spatial_temperature), 1e-6), dim=-1)
    else:
        source = torch.sigmoid(logits)
    if spec["merge"] == "max":
        obj = source.amax(dim=2)
    else:
        obj = source.mean(dim=2)
    side = args.grid_size
    obj_2d = obj.reshape(B, K, side, side)
    up = F.interpolate(obj_2d, size=(args.eval_size, args.eval_size), mode="bilinear", align_corners=False)
    return up[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", action="append", required=True,
                        help="'label|checkpoint_path|readout|merge'. readout: threshold/spatial/competition/nullbg/ovt_owner.")
    parser.add_argument("--sample_index", type=int, default=0)
    parser.add_argument("--val_jsonl", default="/home/jovyan/PGOT/data/pgot_val.jsonl")
    parser.add_argument("--output_dir", default="/home/jovyan/PGOT/outputs/ovt_overlay_compare")
    parser.add_argument("--gt_source", choices=["pix2cap_panoptic", "coco_instance"], default="coco_instance")
    parser.add_argument("--coco_mask_cache", default="/home/jovyan/PGOT/data/coco_inst_mask_cache_coda256")
    parser.add_argument("--image_preprocess_mode", choices=["default", "coda_center_crop"], default="default")
    parser.add_argument("--coda_crop_size", type=int, default=512)
    parser.add_argument("--grid_size", type=int, default=32)
    parser.add_argument("--eval_size", type=int, default=256)
    parser.add_argument("--max_caption_tokens", type=int, default=2048)
    parser.add_argument("--n_ovt_per_object", type=int, default=2)
    parser.add_argument("--max_objects", type=int, default=50)
    parser.add_argument("--bg_threshold", type=float, default=0.05)
    parser.add_argument("--spatial_temperature", type=float, default=1.0)
    parser.add_argument("--max_object_overlays", type=int, default=8)
    parser.add_argument("--large_overlay_size", type=int, default=640)
    parser.add_argument("--large_overlay_alpha", type=float, default=0.52)
    parser.add_argument("--large_overlay_legend_items", type=int, default=18)
    parser.add_argument("--dtype", choices=["fp32", "bf16"], default="fp32")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s :: %(message)s")
    os.makedirs(args.output_dir, exist_ok=True)
    specs = [_parse_model_spec(s) for s in args.model]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32 if args.dtype == "fp32" else torch.bfloat16

    with open(args.val_jsonl) as f:
        raw_samples = [json.loads(line) for line in f]
    raw = raw_samples[args.sample_index]
    fallback_source = Image.open(raw["image_path"]).convert("RGB").resize((args.eval_size, args.eval_size), Image.BILINEAR)
    gt_from_cache = False
    overlap = None

    if args.gt_source == "coco_instance":
        cache = CocoInstanceMaskCache(args.coco_mask_cache)
        gt = cache.get(int(raw["image_id"]))
        overlap = cache.get_overlap(int(raw["image_id"]))
        if gt is None:
            seg_ids = [int(s["segment_id"]) for s in raw["segments"]]
            gt = load_gt_panoptic_mask(raw["panoptic_mask_path"], seg_ids, args.eval_size)
        else:
            gt_from_cache = True
            args.eval_size = cache.size
            fallback_source = fallback_source.resize((args.eval_size, args.eval_size), Image.BILINEAR)
        thing_categories = load_thing_categories("/home/jovyan/data/coco/annotations/panoptic_val2017.json")
    else:
        seg_ids = [int(s["segment_id"]) for s in raw["segments"]]
        gt = load_gt_panoptic_mask(raw["panoptic_mask_path"], seg_ids, args.eval_size)
        thing_categories = None

    gt_eval = _gt_eval_mask(gt, overlap if args.gt_source == "coco_instance" else None)
    gt_title = (
        f"GT {args.gt_source} (overlap removed)"
        if args.gt_source == "coco_instance" and overlap is not None
        else f"GT {args.gt_source}"
    )

    large_dir = os.path.join(args.output_dir, "large_pred_overlays")
    os.makedirs(large_dir, exist_ok=True)
    source = None
    comparison_tiles = None
    large_tiles = None
    summary = None

    def _init_artifacts(source_img: Image.Image, source_uses_eval_target_image: bool):
        comparison = [_add_title(source_img, f"source idx={args.sample_index} image_id={raw['image_id']}")]
        comparison.append(_mask_overlay(source_img, gt_eval, gt_title))
        source_large = _add_title(
            source_img.resize((args.large_overlay_size, args.large_overlay_size), Image.BILINEAR),
            f"source idx={args.sample_index} image_id={raw['image_id']}",
            height=46,
        )
        source_large_path = os.path.join(large_dir, "source.png")
        source_large.save(source_large_path)
        gt_labels = None if gt_from_cache else _segment_labels(raw["segments"])
        gt_large = _large_mask_overlay(
            source_img,
            gt_eval,
            gt_title,
            gt_labels,
            alpha=args.large_overlay_alpha,
            size=args.large_overlay_size,
            max_legend_items=args.large_overlay_legend_items,
        )
        gt_large_path = os.path.join(large_dir, "GT_overlay.png")
        gt_large.save(gt_large_path)
        summary_dict = {
            "sample_index": args.sample_index,
            "image_id": raw["image_id"],
            "source_uses_eval_target_image": bool(source_uses_eval_target_image),
            "gt_overlap_removed": bool(args.gt_source == "coco_instance" and overlap is not None),
            "large_source": source_large_path,
            "large_gt_overlay": gt_large_path,
            "models": [],
        }
        return comparison, [source_large, gt_large], summary_dict

    for spec in specs:
        log.info("Loading %s from %s", spec["label"], spec["path"])
        model, tokenizer = _load_model(spec["path"], dtype=dtype, device=device)
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
        collator = PGOTDataCollator(pad_token_id=tokenizer.pad_token_id)
        batch = collator([dataset[args.sample_index]])
        if source is None:
            source_uses_eval_target_image = True
            try:
                source = _source_from_target_tensor(batch["target_images"][0], target_proc)
                source = source.resize((args.eval_size, args.eval_size), Image.BILINEAR)
            except Exception as exc:
                log.warning("Falling back to raw resized source image: %s", exc)
                source_uses_eval_target_image = False
                source = fallback_source
            comparison_tiles, large_tiles, summary = _init_artifacts(source, source_uses_eval_target_image)
        with torch.no_grad():
            out = pgot_forward_eval(
                model,
                images=batch["images"],
                target_images=batch["target_images"],
                caption_input_ids=batch["caption_input_ids"],
                caption_attention_mask=batch["caption_attention_mask"],
                ovt_positions_in_caption=batch["ovt_positions_in_caption"],
                ovt_valid_mask=batch["ovt_valid_mask"],
                return_llm_qk_maps=True,
            )
            pred = _build_pred(args, spec, out, batch, raw_samples, thing_categories)[0].detach().cpu()
            obj_maps = _object_maps(args, spec, out).detach().cpu()

        title = f"{spec['label']} {spec['readout']}/{spec['merge']}"
        comparison_tiles.append(_mask_overlay(source, pred, title))
        pred_labels = _pred_labels_for_readout(args, spec, raw, thing_categories)
        large_pred = _large_mask_overlay(
            source,
            pred,
            title,
            pred_labels,
            alpha=args.large_overlay_alpha,
            size=args.large_overlay_size,
            max_legend_items=args.large_overlay_legend_items,
        )
        large_pred_path = os.path.join(large_dir, f"{_safe_name(spec['label'])}_pred_overlay.png")
        large_pred.save(large_pred_path)
        large_tiles.append(large_pred)

        overlays = [_add_title(source, f"{spec['label']} source")]
        labels = []
        for k, seg in enumerate(raw["segments"][: obj_maps.shape[0]]):
            label = f"{k + 1}:{seg.get('category', 'obj')}"
            labels.append(label)
            if len(overlays) - 1 >= args.max_object_overlays:
                continue
            overlays.append(_heat_overlay(source, obj_maps[k], label))
        overlay_path = os.path.join(args.output_dir, f"{spec['label']}_object_overlays.png")
        _concat_grid(overlays, cols=min(5, len(overlays))).save(overlay_path)
        summary["models"].append({
            "label": spec["label"],
            "path": spec["path"],
            "readout": spec["readout"],
            "merge": spec["merge"],
            "object_overlay": overlay_path,
            "large_pred_overlay": large_pred_path,
            "labels": labels,
        })
        del model, tokenizer, dataset, batch, out
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if summary is None:
        source = fallback_source
        comparison_tiles, large_tiles, summary = _init_artifacts(source, False)

    compare_path = os.path.join(args.output_dir, "comparison_segmentation.png")
    _concat_grid(comparison_tiles, cols=min(4, len(comparison_tiles))).save(compare_path)
    large_compare_path = os.path.join(large_dir, "comparison_large_pred_overlays.png")
    _concat_grid(large_tiles, cols=min(3, len(large_tiles))).save(large_compare_path)
    summary["large_comparison"] = large_compare_path
    with open(os.path.join(args.output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    log.info("Wrote %s", compare_path)
    log.info("Wrote %s", large_compare_path)


if __name__ == "__main__":
    main()
