#!/usr/bin/env python
"""
Build Stage A diagnostic cache using generation-time attention traces.

Pipeline:
image -> generate caption token-by-token while recording attention over image patches
      -> extract first noun phrase / noun head
      -> aggregate trace maps over the selected generated tokens
      -> evaluate weakly against COCO instance boxes

Outputs:
- metadata.jsonl: per-image diagnostic records
- shards/cache_XXXXX.pt: sharded phrase/head trace maps
- summary.json: run summary
- samples/<image_id>/*: optional debug visualizations
"""

import argparse
import json
import os
import random
import re
import sys
import types
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import spacy
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parents[1]
INFERENCE_ROOT = REPO_ROOT / "inference"
for path in (str(REPO_ROOT), str(INFERENCE_ROOT)):
    if path not in sys.path:
        sys.path.append(path)

if "IPython" not in sys.modules:
    ipython_stub = types.ModuleType("IPython")
    ipython_stub.get_ipython = lambda: None
    sys.modules["IPython"] = ipython_stub

from cli import (  # type: ignore
    build_prompt,
    ensure_output_dir,
    load_image_rgb,
    prepare_special_token_ids,
    preprocess_single_image,
    tokenize_prompt,
)
from scale_rae.constants import IMAGE_TOKEN_INDEX
from utils.load_model import load_scale_rae_model  # type: ignore


DEFAULT_MODEL_PATH = "nyu-visionx/Scale-RAE-Qwen1.5B_DiT2.4B"
DEFAULT_CAPTION_PROMPT = "Describe this image in one concise sentence."
SAFE_CATEGORY_ALIASES = {
    "table": "dining table",
    "phone": "cell phone",
    "sofa": "couch",
    "bike": "bicycle",
    "motorbike": "motorcycle",
    "plane": "airplane",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Stage A generation-trace diagnostics.")
    parser.add_argument("--image-dir", required=True, help="Directory containing images.")
    parser.add_argument("--output-dir", required=True, help="Output directory for the diagnostic run.")
    parser.add_argument("--coco-instances-json", required=True, help="COCO instances json for weak bbox evaluation.")
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--caption-prompt", default=DEFAULT_CAPTION_PROMPT)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=["fp16", "bf16", "fp32"], default="bf16")
    parser.add_argument("--caption-max-new-tokens", type=int, default=64)
    parser.add_argument("--max-caption-tokens", type=int, default=64)
    parser.add_argument("--trace-last-n-layers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--map-shard-size", type=int, default=1000)
    parser.add_argument("--ranking-top-k", type=int, default=10)
    parser.add_argument("--spacy-model", default="en_core_web_sm")
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--save-debug-images", action="store_true")
    parser.add_argument("--debug-limit", type=int, default=100)
    parser.add_argument("--force-eager-attention", action="store_true")
    parser.add_argument("--disable-kv-cache", action="store_true")
    parser.add_argument(
        "--attention-source",
        choices=["generation", "teacher_forced"],
        default="generation",
        help="generation: extract attention during caption generation (KV-cached). "
             "teacher_forced: first generate caption without attention tracking, "
             "then run one full teacher-forced forward to extract raw attention.",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_dtype(dtype_name: str, device_name: str) -> torch.dtype:
    if dtype_name == "fp16":
        dtype = torch.float16
    elif dtype_name == "bf16":
        dtype = torch.bfloat16
    else:
        dtype = torch.float32
    if device_name.startswith("cpu") and dtype != torch.float32:
        return torch.float32
    return dtype


def normalize_caption(text: str) -> str:
    text = " ".join(text.strip().split())
    return text if text else "<empty>"


def save_json(path: str, payload: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def list_image_files(image_dir: str) -> List[str]:
    allowed = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    return [
        str(path)
        for path in sorted(Path(image_dir).iterdir())
        if path.is_file() and path.suffix.lower() in allowed
    ]


def take_shard(items: Sequence[Any], num_shards: int, shard_index: int) -> List[Any]:
    if num_shards < 1:
        raise ValueError("--num-shards must be at least 1.")
    if shard_index < 0 or shard_index >= num_shards:
        raise ValueError("--shard-index must be in [0, num_shards).")
    return [item for idx, item in enumerate(items) if idx % num_shards == shard_index]


def tensor_to_pil(image_tensor: torch.Tensor) -> Image.Image:
    array = image_tensor.detach().float().cpu().clamp(0, 1).permute(1, 2, 0).numpy()
    array = np.clip(array * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(array)


def normalize_attention_for_visualization(
    attn: torch.Tensor,
    low_quantile: float = 0.80,
    high_quantile: float = 0.995,
    gamma: float = 0.75,
) -> torch.Tensor:
    arr = attn.detach().float().clamp(min=0.0)
    if arr.numel() == 0 or float(arr.max().item()) <= 0.0:
        return torch.zeros_like(arr, dtype=torch.float32)
    low = torch.quantile(arr, low_quantile)
    high = torch.quantile(arr, high_quantile)
    if float(high.item()) <= float(low.item()):
        low = arr.min()
        high = arr.max()
    vis = (arr - low) / (high - low + 1e-8)
    vis = vis.clamp(0.0, 1.0)
    if gamma != 1.0:
        vis = vis.pow(gamma)
    return vis


def make_attention_overlay(source: torch.Tensor, attn: torch.Tensor, color: Tuple[float, float, float]) -> torch.Tensor:
    c, height, width = source.shape
    n_patches = int(attn.numel())
    grid_side = int(n_patches ** 0.5)
    if grid_side * grid_side != n_patches:
        return source.clone()
    vis = normalize_attention_for_visualization(attn)
    attn_2d = vis.view(grid_side, grid_side).unsqueeze(0).unsqueeze(0)
    attn_map = F.interpolate(attn_2d, size=(height, width), mode="bilinear", align_corners=False)[0, 0]
    attn_map = attn_map.clamp(0.0, 1.0)
    color_t = torch.tensor(color, dtype=torch.float32).view(3, 1, 1)
    overlay = source.float() * (1.0 - 0.70 * attn_map.unsqueeze(0)) + color_t * 0.85 * attn_map.unsqueeze(0)
    return overlay.clamp(0.0, 1.0)


def make_attention_heatmap(attn: torch.Tensor, size: int = 224) -> Image.Image:
    n_patches = int(attn.numel())
    grid_side = int(n_patches ** 0.5)
    if grid_side * grid_side != n_patches:
        raise ValueError(f"Expected square patch count, got {n_patches}")
    vis = normalize_attention_for_visualization(attn)
    attn_2d = vis.view(grid_side, grid_side).detach().float().cpu().numpy()
    red = np.clip(attn_2d * 255.0 * 1.6, 0, 255)
    green = np.clip((attn_2d - 0.20) * 255.0 * 2.4, 0, 255)
    blue = np.clip((attn_2d - 0.60) * 255.0 * 3.0, 0, 255)
    heat = np.stack([red, green, blue], axis=-1).astype(np.uint8)
    return Image.fromarray(heat, mode="RGB").resize((size, size), resample=Image.BILINEAR)


def denormalize_image_tensor(image_tensor: torch.Tensor, image_processor) -> torch.Tensor:
    mean = torch.tensor(image_processor[0].image_mean, device=image_tensor.device, dtype=image_tensor.dtype).view(1, 3, 1, 1)
    std = torch.tensor(image_processor[0].image_std, device=image_tensor.device, dtype=image_tensor.dtype).view(1, 3, 1, 1)
    return (image_tensor * std + mean).clamp(0, 1)


def map_stats(attn: torch.Tensor) -> Dict[str, float]:
    arr = attn.detach().float().clamp(min=0.0)
    total = arr.sum().clamp(min=1e-8)
    probs = arr / total
    entropy = float((-(probs * probs.clamp(min=1e-8).log()).sum()).item())
    topk = max(1, min(10, arr.numel()))
    topk_mass = float(torch.topk(probs, k=topk).values.sum().item())
    return {
        "min": float(arr.min().item()),
        "max": float(arr.max().item()),
        "mean": float(arr.mean().item()),
        "entropy": entropy,
        "top10_mass": topk_mass,
    }


def build_decoded_offsets(tokenizer, token_ids: Sequence[int]) -> Tuple[str, List[Tuple[int, int]]]:
    decoded_prefix = ""
    offsets: List[Tuple[int, int]] = []
    for idx in range(len(token_ids)):
        next_prefix = tokenizer.decode(
            token_ids[: idx + 1],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        offsets.append((len(decoded_prefix), len(next_prefix)))
        decoded_prefix = next_prefix
    return decoded_prefix, offsets


def find_token_span(offsets: Sequence[Tuple[int, int]], char_start: int, char_end: int) -> Tuple[Optional[int], Optional[int]]:
    tok_start = None
    tok_end = None
    for idx, (start, end) in enumerate(offsets):
        if start < char_end and end > char_start:
            if tok_start is None:
                tok_start = idx
            tok_end = idx + 1
    return tok_start, tok_end


def extract_noun_info_from_token_ids(
    token_ids: Sequence[int],
    tokenizer,
    nlp,
    max_caption_tokens: int,
) -> Dict[str, Any]:
    token_ids = list(token_ids[:max_caption_tokens])
    caption_text, offsets = build_decoded_offsets(tokenizer, token_ids)
    doc = nlp(caption_text)

    noun_chunks: List[Dict[str, Any]] = []
    for chunk in doc.noun_chunks:
        if not any(token.pos_ in {"NOUN", "PROPN"} for token in chunk):
            continue
        tok_start, tok_end = find_token_span(offsets, chunk.start_char, chunk.end_char)
        head_start, head_end = find_token_span(
            offsets,
            chunk.root.idx,
            chunk.root.idx + len(chunk.root.text),
        )
        if tok_start is None or tok_end is None or head_start is None or head_end is None:
            continue
        noun_chunks.append(
            {
                "text": chunk.text,
                "token_start": int(tok_start),
                "token_end": int(tok_end),
                "char_start": int(chunk.start_char),
                "char_end": int(chunk.end_char),
                "head_text": chunk.root.text,
                "head_token_start": int(head_start),
                "head_token_end": int(head_end),
                "head_char_start": int(chunk.root.idx),
                "head_char_end": int(chunk.root.idx + len(chunk.root.text)),
            }
        )

    selected = noun_chunks[0] if noun_chunks else None
    selection_strategy = "noun_chunk"

    if selected is None:
        first_head = None
        for token in doc:
            if token.pos_ in {"NOUN", "PROPN"} and not token.is_space and not token.is_punct:
                first_head = token
                break
        if first_head is not None:
            subtree = list(first_head.subtree)
            if subtree:
                phrase_start_char = subtree[0].idx
                phrase_end_char = subtree[-1].idx + len(subtree[-1].text)
            else:
                phrase_start_char = first_head.idx
                phrase_end_char = first_head.idx + len(first_head.text)
            tok_start, tok_end = find_token_span(offsets, phrase_start_char, phrase_end_char)
            head_start, head_end = find_token_span(
                offsets,
                first_head.idx,
                first_head.idx + len(first_head.text),
            )
            if tok_start is not None and tok_end is not None and head_start is not None and head_end is not None:
                selected = {
                    "text": caption_text[phrase_start_char:phrase_end_char],
                    "token_start": int(tok_start),
                    "token_end": int(tok_end),
                    "char_start": int(phrase_start_char),
                    "char_end": int(phrase_end_char),
                    "head_text": first_head.text,
                    "head_token_start": int(head_start),
                    "head_token_end": int(head_end),
                    "head_char_start": int(first_head.idx),
                    "head_char_end": int(first_head.idx + len(first_head.text)),
                }
                selection_strategy = "head_subtree"

    if selected is None:
        selection_strategy = "none"

    return {
        "caption": normalize_caption(caption_text),
        "token_ids": token_ids,
        "noun_chunks": noun_chunks,
        "selected": selected,
        "selection_strategy": selection_strategy,
    }


def normalize_label(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return " ".join(text.split())


def build_coco_index(instances_json: str) -> Dict[str, Any]:
    with open(instances_json, "r", encoding="utf-8") as f:
        payload = json.load(f)

    images_by_id = {int(item["id"]): item for item in payload.get("images", [])}
    anns_by_image_id: Dict[int, List[Dict[str, Any]]] = {}
    for ann in payload.get("annotations", []):
        anns_by_image_id.setdefault(int(ann["image_id"]), []).append(ann)

    categories_by_id = {int(item["id"]): item for item in payload.get("categories", [])}
    exact_name_to_ids: Dict[str, List[int]] = {}
    last_token_to_ids: Dict[str, List[int]] = {}
    for category_id, category in categories_by_id.items():
        name_norm = normalize_label(category["name"])
        exact_name_to_ids.setdefault(name_norm, []).append(category_id)
        tokens = name_norm.split()
        if tokens:
            last_token_to_ids.setdefault(tokens[-1], []).append(category_id)

    return {
        "images_by_id": images_by_id,
        "anns_by_image_id": anns_by_image_id,
        "categories_by_id": categories_by_id,
        "exact_name_to_ids": exact_name_to_ids,
        "last_token_to_ids": last_token_to_ids,
    }


def match_coco_category_ids(
    head_text: Optional[str],
    phrase_text: Optional[str],
    coco_index: Dict[str, Any],
) -> List[int]:
    exact_name_to_ids = coco_index["exact_name_to_ids"]
    last_token_to_ids = coco_index["last_token_to_ids"]

    head_norm = normalize_label(head_text or "")
    phrase_norm = normalize_label(phrase_text or "")

    if phrase_norm in exact_name_to_ids:
        return sorted(set(exact_name_to_ids[phrase_norm]))
    if head_norm in exact_name_to_ids:
        return sorted(set(exact_name_to_ids[head_norm]))

    alias_norm = normalize_label(SAFE_CATEGORY_ALIASES.get(head_norm, ""))
    if alias_norm and alias_norm in exact_name_to_ids:
        return sorted(set(exact_name_to_ids[alias_norm]))

    if head_norm in last_token_to_ids and len(last_token_to_ids[head_norm]) == 1:
        return list(last_token_to_ids[head_norm])

    phrase_tokens = phrase_norm.split()
    if phrase_tokens:
        last_token = phrase_tokens[-1]
        if last_token in last_token_to_ids and len(last_token_to_ids[last_token]) == 1:
            return list(last_token_to_ids[last_token])

    return []


def select_object_chunk_for_image(
    noun_chunks: Sequence[Dict[str, Any]],
    image_id: int,
    coco_index: Dict[str, Any],
) -> Tuple[Optional[Dict[str, Any]], str, List[int], List[Dict[str, Any]]]:
    if noun_chunks:
        chunk = noun_chunks[0]
        category_ids = match_coco_category_ids(
            head_text=chunk.get("head_text"),
            phrase_text=chunk.get("text"),
            coco_index=coco_index,
        )
        matched_annotations = [
            ann
            for ann in coco_index["anns_by_image_id"].get(image_id, [])
            if int(ann["category_id"]) in category_ids
        ]
        strategy = (
            "first_noun_chunk_coco_matched"
            if category_ids and matched_annotations
            else "first_noun_chunk_unmatched"
        )
        return chunk, strategy, category_ids, matched_annotations

    return None, "none", [], []


def build_patch_mask_from_bboxes(width: int, height: int, boxes: Sequence[Sequence[float]], grid_side: int = 16) -> torch.Tensor:
    mask = torch.zeros(grid_side * grid_side, dtype=torch.bool)
    if width <= 0 or height <= 0 or not boxes:
        return mask
    for row in range(grid_side):
        for col in range(grid_side):
            x_center = (col + 0.5) / grid_side * width
            y_center = (row + 0.5) / grid_side * height
            inside = False
            for x, y, w, h in boxes:
                if x_center >= x and x_center <= x + w and y_center >= y and y_center <= y + h:
                    inside = True
                    break
            if inside:
                mask[row * grid_side + col] = True
    return mask


def draw_boxes_on_processed_image(source: torch.Tensor, boxes: Sequence[Sequence[float]], image_width: int, image_height: int) -> Image.Image:
    image = tensor_to_pil(source)
    draw = ImageDraw.Draw(image)
    if image_width <= 0 or image_height <= 0:
        return image
    scale_x = image.width / float(image_width)
    scale_y = image.height / float(image_height)
    for x, y, w, h in boxes:
        draw.rectangle(
            (x * scale_x, y * scale_y, (x + w) * scale_x, (y + h) * scale_y),
            outline=(40, 255, 80),
            width=2,
        )
    return image


def evaluate_trace_map_against_boxes(
    trace_map: torch.Tensor,
    image_info: Dict[str, Any],
    boxes: Sequence[Sequence[float]],
) -> Dict[str, Any]:
    if not boxes:
        return {"valid": False}

    width = int(image_info.get("width", 0))
    height = int(image_info.get("height", 0))
    patch_mask = build_patch_mask_from_bboxes(width, height, boxes, grid_side=16)
    if not bool(patch_mask.any()):
        return {"valid": False}

    arr = trace_map.detach().float().clamp(min=0.0)
    probs = arr / arr.sum().clamp(min=1e-8)
    inside_mass = float(probs[patch_mask].sum().item())
    argmax_in_box = bool(patch_mask[int(torch.argmax(probs).item())].item())
    topk = max(1, int(round(arr.numel() * 0.10)))
    topk_idx = torch.topk(probs, k=topk).indices
    topk_in_box = float(patch_mask[topk_idx].float().mean().item())
    return {
        "valid": True,
        "inside_mass": inside_mass,
        "argmax_in_box": argmax_in_box,
        "top10_patch_in_box_rate": topk_in_box,
        "num_boxes": int(len(boxes)),
    }


def aggregate_step_maps(step_maps: Sequence[torch.Tensor], token_start: int, token_end: int) -> Tuple[torch.Tensor, bool]:
    if token_start < 0 or token_end <= token_start or token_end > len(step_maps):
        if step_maps:
            zero_map = torch.zeros_like(step_maps[0], dtype=torch.float32)
        else:
            zero_map = torch.zeros(256, dtype=torch.float32)
        return zero_map, False
    stacked = torch.stack([step_maps[idx].float() for idx in range(token_start, token_end)], dim=0)
    trace = stacked.mean(dim=0)
    trace = trace / trace.sum().clamp(min=1e-8)
    return trace, True


def build_ranking_entry(record: Dict[str, Any], metric_key: str, sample_root: str) -> Dict[str, Any]:
    metric = record.get(metric_key, {}) if isinstance(record.get(metric_key), dict) else {}
    image_id = str(record.get("image_id", ""))
    sample_dir = os.path.join(sample_root, image_id)
    return {
        "image_id": image_id,
        "image": record.get("image"),
        "caption": record.get("caption"),
        "first_noun_phrase": record.get("first_noun_phrase"),
        "noun_head": record.get("noun_head"),
        "selection_strategy": record.get("selection_strategy"),
        "matched_coco_category_names": record.get("matched_coco_category_names", []),
        "inside_mass": metric.get("inside_mass"),
        "argmax_in_box": metric.get("argmax_in_box"),
        "top10_patch_in_box_rate": metric.get("top10_patch_in_box_rate"),
        "record_path": os.path.abspath(os.path.join(sample_dir, "record.json")),
        "trace_overlay_path": os.path.abspath(
            os.path.join(sample_dir, "head_trace_overlay.png" if metric_key == "head_bbox_eval" else "phrase_trace_overlay.png")
        ),
        "trace_map_path": os.path.abspath(
            os.path.join(sample_dir, "head_trace_map.png" if metric_key == "head_bbox_eval" else "phrase_trace_map.png")
        ),
        "bbox_overlay_path": os.path.abspath(os.path.join(sample_dir, "bbox_overlay.png")),
    }


def write_debug_rankings(output_dir: str, metadata_records: Sequence[Dict[str, Any]], top_k: int) -> Dict[str, str]:
    analysis_dir = os.path.join(output_dir, "analysis")
    ensure_output_dir(analysis_dir)
    sample_root = os.path.join(output_dir, "samples")
    written_paths: Dict[str, str] = {}
    top_k = max(1, int(top_k))

    for metric_key, prefix in (("head_bbox_eval", "head"), ("phrase_bbox_eval", "phrase")):
        ranked = [
            build_ranking_entry(record, metric_key, sample_root)
            for record in metadata_records
            if isinstance(record.get(metric_key), dict) and record[metric_key].get("valid")
        ]
        ranked.sort(key=lambda item: float(item["inside_mass"]), reverse=True)
        best = ranked[:top_k]
        worst = list(reversed(ranked[-top_k:])) if ranked else []

        best_path = os.path.join(analysis_dir, f"{prefix}_best_{top_k}.json")
        worst_path = os.path.join(analysis_dir, f"{prefix}_worst_{top_k}.json")
        save_json(best_path, best)
        save_json(worst_path, worst)
        written_paths[f"{prefix}_best"] = os.path.abspath(best_path)
        written_paths[f"{prefix}_worst"] = os.path.abspath(worst_path)

    guide_path = os.path.join(analysis_dir, "README.txt")
    guide_lines = [
        "How to review Stage A generation-trace diagnostics",
        "",
        "1. Open head_best_*.json and head_worst_*.json first.",
        "2. For each entry, inspect trace_overlay_path against bbox_overlay_path.",
        "3. Prefer head maps over phrase maps when deciding whether the prior is usable.",
        "",
        "Files in this folder are generated automatically from metadata.jsonl.",
    ]
    Path(guide_path).write_text("\n".join(guide_lines) + "\n", encoding="utf-8")
    written_paths["guide"] = os.path.abspath(guide_path)
    return written_paths


def maybe_force_eager_attention(model) -> None:
    targets = [
        getattr(model, "config", None),
        getattr(model, "model", None),
        getattr(getattr(model, "model", None), "config", None),
        getattr(model, "get_model", lambda: None)(),
        getattr(getattr(model, "get_model", lambda: None)(), "config", None),
    ]
    for target in targets:
        if target is None:
            continue
        if hasattr(target, "_attn_implementation"):
            target._attn_implementation = "eager"
        if hasattr(target, "config") and hasattr(target.config, "_attn_implementation"):
            target.config._attn_implementation = "eager"


def collect_step_attention_map(attentions, image_start: int, image_end: int, last_n_layers: int) -> torch.Tensor:
    if attentions is None:
        raise RuntimeError("No attentions returned. Try --force-eager-attention.")
    valid_layers = [layer for layer in attentions if layer is not None and torch.is_tensor(layer)]
    if not valid_layers:
        raise RuntimeError("Attention tuple was empty. Try --force-eager-attention.")
    use_layers = valid_layers[-max(1, int(last_n_layers)) :]
    layer_maps: List[torch.Tensor] = []
    for layer_attn in use_layers:
        if layer_attn.ndim != 4:
            continue
        # [B, heads, q_len, k_len] -> use last query row as the context used to predict the next token.
        img_slice = layer_attn[0, :, -1, image_start:image_end].float()
        if img_slice.numel() == 0:
            continue
        layer_maps.append(img_slice.mean(dim=0))
    if not layer_maps:
        raise RuntimeError("Could not extract image attention slice from attentions.")
    attn_map = torch.stack(layer_maps, dim=0).mean(dim=0)
    attn_map = attn_map.clamp(min=0.0)
    attn_map = attn_map / attn_map.sum().clamp(min=1e-8)
    return attn_map.detach().cpu().float()


def generate_caption_with_trace(
    model,
    tokenizer,
    prompt_input_ids: torch.Tensor,
    image_tensor_single: torch.Tensor,
    eos_token_id: int,
    max_new_tokens: int,
    start_image_token_id: int,
    end_image_token_id: int,
    last_n_layers: int,
    use_kv_cache: bool,
) -> Dict[str, Any]:
    if image_tensor_single.ndim != 4:
        raise ValueError(f"Expected image tensor [1,C,H,W], got {tuple(image_tensor_single.shape)}")
    image_batch = image_tensor_single.unsqueeze(0)

    with torch.inference_mode():
        (
            _input_ids,
            _position_ids,
            _attention_mask,
            _past_key_values,
            prefix_inputs_embeds,
            _labels,
            _selected_features,
            _input_embed_mask,
            _attention_bias,
            extra_mm,
        ) = model.prepare_inputs_labels_for_multimodal(
            prompt_input_ids,
            None,
            None,
            None,
            None,
            images=image_batch,
            image_embeds=None,
        )

        if prefix_inputs_embeds is None:
            raise RuntimeError("prepare_inputs_labels_for_multimodal returned no inputs_embeds.")

        image_positions = (prompt_input_ids[0] == IMAGE_TOKEN_INDEX).nonzero(as_tuple=False).flatten()
        if image_positions.numel() != 1:
            raise RuntimeError(f"Expected exactly one image placeholder, got {int(image_positions.numel())}")
        image_start = int(image_positions[0].item())
        image_end = image_start + int(getattr(model, "num_image_tokens", 256))

        inputs_embeds = prefix_inputs_embeds
        current_inputs_embeds = prefix_inputs_embeds
        generated_ids: List[int] = []
        step_maps: List[torch.Tensor] = []
        stop_reason = "max_new_tokens"
        stop_ids = {int(eos_token_id), int(start_image_token_id), int(end_image_token_id)}
        past_key_values = None

        for _step in range(max_new_tokens):
            outputs = model.forward(
                input_ids=None,
                attention_mask=None,
                position_ids=None,
                past_key_values=past_key_values,
                inputs_embeds=current_inputs_embeds if use_kv_cache else inputs_embeds,
                use_cache=use_kv_cache,
                return_dict=True,
                decoding=False,
                guidance_level=1.0,
                output_attentions=True,
                output_hidden_states=False,
            )

            next_token_logits = outputs.logits[:, -1, :]
            next_token = int(torch.argmax(next_token_logits, dim=-1).item())

            if next_token in stop_ids:
                stop_reason = f"stop_token:{next_token}"
                break

            step_map = collect_step_attention_map(
                attentions=outputs.attentions,
                image_start=image_start,
                image_end=image_end,
                last_n_layers=last_n_layers,
            )
            step_maps.append(step_map)
            generated_ids.append(next_token)

            next_token_tensor = torch.tensor([[next_token]], device=model.device, dtype=torch.long)
            next_token_embed = model.get_model().embed_tokens(next_token_tensor)
            next_token_embed = next_token_embed.to(device=inputs_embeds.device, dtype=inputs_embeds.dtype)
            if use_kv_cache:
                past_key_values = outputs.past_key_values
                current_inputs_embeds = next_token_embed
            else:
                inputs_embeds = torch.cat((inputs_embeds, next_token_embed), dim=1)
                current_inputs_embeds = inputs_embeds

        caption = normalize_caption(
            tokenizer.decode(
                generated_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
        )
        token_texts = tokenizer.convert_ids_to_tokens(generated_ids)

    return {
        "caption": caption,
        "generated_ids": generated_ids,
        "token_texts": token_texts,
        "step_maps": step_maps,
        "stop_reason": stop_reason,
        "image_token_span": [image_start, image_end],
        "cache_mode": "kv_cache" if use_kv_cache else "no_cache",
    }


def generate_caption_only(
    model,
    tokenizer,
    prompt_input_ids: torch.Tensor,
    image_tensor_single: torch.Tensor,
    eos_token_id: int,
    max_new_tokens: int,
    start_image_token_id: int,
    end_image_token_id: int,
    use_kv_cache: bool,
) -> Dict[str, Any]:
    if image_tensor_single.ndim != 4:
        raise ValueError(f"Expected image tensor [1,C,H,W], got {tuple(image_tensor_single.shape)}")
    image_batch = image_tensor_single.unsqueeze(0)

    with torch.inference_mode():
        (
            _input_ids,
            _position_ids,
            _attention_mask,
            _past_key_values,
            prefix_inputs_embeds,
            _labels,
            _selected_features,
            _input_embed_mask,
            _attention_bias,
            _extra_mm,
        ) = model.prepare_inputs_labels_for_multimodal(
            prompt_input_ids,
            None,
            None,
            None,
            None,
            images=image_batch,
            image_embeds=None,
        )
        if prefix_inputs_embeds is None:
            raise RuntimeError("prepare_inputs_labels_for_multimodal returned no inputs_embeds.")

        image_positions = (prompt_input_ids[0] == IMAGE_TOKEN_INDEX).nonzero(as_tuple=False).flatten()
        if image_positions.numel() != 1:
            raise RuntimeError(f"Expected exactly one image placeholder, got {int(image_positions.numel())}")
        image_start = int(image_positions[0].item())
        image_end = image_start + int(getattr(model, "num_image_tokens", 256))
        prefix_len = int(prefix_inputs_embeds.shape[1])

        inputs_embeds = prefix_inputs_embeds.to(device=model.device, dtype=torch.float32)
        current_inputs_embeds = inputs_embeds
        generated_ids: List[int] = []
        stop_reason = "max_new_tokens"
        stop_ids = {int(eos_token_id), int(start_image_token_id), int(end_image_token_id)}
        past_key_values = None

        for _step in range(max_new_tokens):
            total_context_len = (prefix_len + len(generated_ids)) if use_kv_cache else inputs_embeds.shape[1]
            step_attention_mask = torch.ones(
                (1, total_context_len),
                device=model.device,
                dtype=torch.bool,
            )
            outputs = model.forward(
                input_ids=None,
                attention_mask=step_attention_mask,
                position_ids=None,
                past_key_values=past_key_values,
                inputs_embeds=current_inputs_embeds if use_kv_cache else inputs_embeds,
                use_cache=use_kv_cache,
                return_dict=True,
                decoding=False,
                guidance_level=1.0,
                output_attentions=False,
                output_hidden_states=False,
            )
            next_token = int(torch.argmax(outputs.logits[:, -1, :], dim=-1).item())
            if next_token in stop_ids:
                stop_reason = f"stop_token:{next_token}"
                break
            generated_ids.append(next_token)
            next_token_tensor = torch.tensor([[next_token]], device=model.device, dtype=torch.long)
            next_token_embed = model.get_model().embed_tokens(next_token_tensor)
            next_token_embed = next_token_embed.to(device=inputs_embeds.device, dtype=inputs_embeds.dtype)
            if use_kv_cache:
                past_key_values = outputs.past_key_values
                current_inputs_embeds = next_token_embed
            else:
                inputs_embeds = torch.cat((inputs_embeds, next_token_embed), dim=1)
                current_inputs_embeds = inputs_embeds

        caption = normalize_caption(
            tokenizer.decode(
                generated_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
        )
        token_texts = tokenizer.convert_ids_to_tokens(generated_ids)

    return {
        "caption": caption,
        "generated_ids": generated_ids,
        "token_texts": token_texts,
        "prefix_inputs_embeds": inputs_embeds[:, :prefix_len, :].detach(),
        "image_token_span": [image_start, image_end],
        "prefix_len": prefix_len,
        "stop_reason": stop_reason,
        "cache_mode": "kv_cache" if use_kv_cache else "no_cache",
    }


def teacher_forced_extract_step_maps(
    model,
    prefix_inputs_embeds: torch.Tensor,
    generated_ids: Sequence[int],
    image_start: int,
    image_end: int,
    last_n_layers: int,
) -> List[torch.Tensor]:
    if not generated_ids:
        return []
    prefix_len = int(prefix_inputs_embeds.shape[1])

    with torch.inference_mode():
        gen_ids_tensor = torch.tensor([list(generated_ids)], device=model.device, dtype=torch.long)
        gen_embeds = model.get_model().embed_tokens(gen_ids_tensor)
        gen_embeds = gen_embeds.to(device=prefix_inputs_embeds.device, dtype=prefix_inputs_embeds.dtype)
        full_embeds = torch.cat((prefix_inputs_embeds, gen_embeds), dim=1)
        full_embeds = full_embeds.to(device=model.device, dtype=torch.float32)
        attention_mask = torch.ones((1, full_embeds.shape[1]), device=model.device, dtype=torch.bool)

        outputs = model.forward(
            input_ids=None,
            attention_mask=attention_mask,
            position_ids=None,
            past_key_values=None,
            inputs_embeds=full_embeds,
            use_cache=False,
            return_dict=True,
            decoding=False,
            guidance_level=1.0,
            output_attentions=True,
            output_hidden_states=False,
        )
        attentions = outputs.attentions
        if attentions is None:
            raise RuntimeError("No attentions returned. Try --force-eager-attention.")
        valid_layers = [layer for layer in attentions if layer is not None and torch.is_tensor(layer)]
        if not valid_layers:
            raise RuntimeError("Attention tuple was empty. Try --force-eager-attention.")
        use_layers = valid_layers[-max(1, int(last_n_layers)) :]

        step_maps: List[torch.Tensor] = []
        for k in range(len(generated_ids)):
            query_pos = prefix_len + k - 1  # query row whose logit predicts generated_ids[k]
            layer_maps: List[torch.Tensor] = []
            for layer_attn in use_layers:
                if layer_attn.ndim != 4:
                    continue
                if query_pos < 0 or query_pos >= layer_attn.shape[2]:
                    continue
                img_slice = layer_attn[0, :, query_pos, image_start:image_end].float()
                if img_slice.numel() == 0:
                    continue
                layer_maps.append(img_slice.mean(dim=0))
            if not layer_maps:
                raise RuntimeError("Could not extract image attention slice from teacher-forced attentions.")
            attn_map = torch.stack(layer_maps, dim=0).mean(dim=0)
            attn_map = attn_map.clamp(min=0.0)
            attn_map = attn_map / attn_map.sum().clamp(min=1e-8)
            step_maps.append(attn_map.detach().cpu().float())

    return step_maps


class MapShardWriter:
    def __init__(self, output_dir: str, shard_size: int) -> None:
        self.output_dir = output_dir
        self.shard_size = int(shard_size)
        self.shard_idx = 0
        self.pending_records: List[Dict[str, Any]] = []
        self.pending_image_ids: List[str] = []
        self.pending_phrase_maps: List[torch.Tensor] = []
        self.pending_head_maps: List[torch.Tensor] = []
        self.shard_paths: List[str] = []
        ensure_output_dir(self.output_dir)

    def append(self, record: Dict[str, Any], image_id: str, phrase_map: torch.Tensor, head_map: torch.Tensor) -> List[Dict[str, Any]]:
        self.pending_records.append(record)
        self.pending_image_ids.append(str(image_id))
        self.pending_phrase_maps.append(phrase_map.detach().cpu().to(torch.float16))
        self.pending_head_maps.append(head_map.detach().cpu().to(torch.float16))
        if len(self.pending_records) >= self.shard_size:
            return self.flush()
        return []

    def flush(self) -> List[Dict[str, Any]]:
        if not self.pending_records:
            return []
        shard_name = f"cache_{self.shard_idx:05d}.pt"
        shard_path = os.path.join(self.output_dir, shard_name)
        payload = {
            "image_ids": list(self.pending_image_ids),
            "phrase_maps": torch.stack(self.pending_phrase_maps, dim=0),
            "head_maps": torch.stack(self.pending_head_maps, dim=0),
        }
        torch.save(payload, shard_path)
        self.shard_paths.append(os.path.abspath(shard_path))

        finalized: List[Dict[str, Any]] = []
        for local_idx, record in enumerate(self.pending_records):
            record["map_shard"] = os.path.abspath(shard_path)
            record["map_index"] = int(local_idx)
            finalized.append(record)

        self.pending_records = []
        self.pending_image_ids = []
        self.pending_phrase_maps = []
        self.pending_head_maps = []
        self.shard_idx += 1
        return finalized


def main() -> None:
    args = parse_args()
    use_kv_cache = not args.disable_kv_cache
    ensure_output_dir(args.output_dir)
    ensure_output_dir(os.path.join(args.output_dir, "shards"))
    if args.save_debug_images:
        ensure_output_dir(os.path.join(args.output_dir, "samples"))
    set_seed(args.seed)

    image_paths = list_image_files(args.image_dir)
    if args.max_samples is not None:
        image_paths = image_paths[: args.max_samples]
    global_requested_samples = len(image_paths)
    if global_requested_samples == 0:
        raise SystemExit(f"No images found in: {args.image_dir}")
    image_paths = take_shard(image_paths, args.num_shards, args.shard_index)

    metadata_path = os.path.join(args.output_dir, "metadata.jsonl")
    summary_path = os.path.join(args.output_dir, "summary.json")
    if not image_paths:
        Path(metadata_path).write_text("", encoding="utf-8")
        empty_summary = {
            "image_dir": os.path.abspath(args.image_dir),
            "output_dir": os.path.abspath(args.output_dir),
            "requested_samples": 0,
            "global_requested_samples": global_requested_samples,
            "num_shards": args.num_shards,
            "shard_index": args.shard_index,
            "notes": ["This shard had no assigned samples after sharding."],
        }
        save_json(summary_path, empty_summary)
        print(json.dumps(empty_summary, indent=2, ensure_ascii=False))
        return

    print(f"Loading spaCy model: {args.spacy_model}")
    nlp = spacy.load(args.spacy_model)
    print(f"Loading COCO instances: {args.coco_instances_json}")
    coco_index = build_coco_index(args.coco_instances_json)

    dtype = resolve_dtype(args.dtype, args.device)
    tokenizer, model, image_processor, _context_len = load_scale_rae_model(
        model_path=args.model_path,
        device=args.device,
        dtype=dtype,
    )
    if args.force_eager_attention:
        maybe_force_eager_attention(model)
        print("Forced eager attention for trace extraction.")

    start_id, end_id, eos_id = prepare_special_token_ids(tokenizer)
    caption_prompt_built = build_prompt(args.caption_prompt, model_config=model.config, with_image=True, num_frames=1)
    prompt_input_ids = tokenize_prompt(caption_prompt_built, tokenizer, device=model.device)

    debug_saved = 0
    trace_valid_count = 0
    error_count = 0
    phrase_eval_valid_count = 0
    head_eval_valid_count = 0
    phrase_inside_mass_total = 0.0
    head_inside_mass_total = 0.0
    phrase_argmax_in_box_count = 0
    head_argmax_in_box_count = 0
    phrase_top10_total = 0.0
    head_top10_total = 0.0

    shard_writer = MapShardWriter(os.path.join(args.output_dir, "shards"), shard_size=args.map_shard_size)

    with open(metadata_path, "w", encoding="utf-8") as metadata_file, tqdm(
        total=len(image_paths),
        desc="StageA Trace",
        unit="img",
        dynamic_ncols=True,
    ) as pbar:
        for idx, image_path in enumerate(image_paths, start=1):
            image_id = Path(image_path).stem
            record: Dict[str, Any] = {
                "image": os.path.abspath(image_path),
                "image_id": image_id,
                "caption_prompt": args.caption_prompt,
                "caption_prompt_built": caption_prompt_built,
                "num_shards": int(args.num_shards),
                "shard_index": int(args.shard_index),
                "trace_source": (
                    "teacher_forced_attention"
                    if args.attention_source == "teacher_forced"
                    else "generation_time_attention"
                ),
                "trace_last_n_layers": int(args.trace_last_n_layers),
            }
            phrase_map = torch.zeros(256, dtype=torch.float32)
            head_map = torch.zeros(256, dtype=torch.float32)

            try:
                image = load_image_rgb(image_path)
                image_tensor, original_size = preprocess_single_image(
                    image,
                    image_processor,
                    device=model.device,
                    dtype=model.dtype,
                )
                source_vis = denormalize_image_tensor(image_tensor, image_processor)[0].detach().cpu()

                if args.attention_source == "teacher_forced":
                    gen_info = generate_caption_only(
                        model=model,
                        tokenizer=tokenizer,
                        prompt_input_ids=prompt_input_ids,
                        image_tensor_single=image_tensor,
                        eos_token_id=eos_id,
                        max_new_tokens=args.caption_max_new_tokens,
                        start_image_token_id=start_id,
                        end_image_token_id=end_id,
                        use_kv_cache=use_kv_cache,
                    )
                    step_maps = teacher_forced_extract_step_maps(
                        model=model,
                        prefix_inputs_embeds=gen_info["prefix_inputs_embeds"],
                        generated_ids=gen_info["generated_ids"],
                        image_start=gen_info["image_token_span"][0],
                        image_end=gen_info["image_token_span"][1],
                        last_n_layers=args.trace_last_n_layers,
                    )
                    trace_info = {
                        "caption": gen_info["caption"],
                        "generated_ids": gen_info["generated_ids"],
                        "token_texts": gen_info["token_texts"],
                        "step_maps": step_maps,
                        "stop_reason": gen_info["stop_reason"],
                        "image_token_span": gen_info["image_token_span"],
                        "cache_mode": gen_info["cache_mode"] + "+teacher_forced",
                    }
                else:
                    trace_info = generate_caption_with_trace(
                        model=model,
                        tokenizer=tokenizer,
                        prompt_input_ids=prompt_input_ids,
                        image_tensor_single=image_tensor,
                        eos_token_id=eos_id,
                        max_new_tokens=args.caption_max_new_tokens,
                        start_image_token_id=start_id,
                        end_image_token_id=end_id,
                        last_n_layers=args.trace_last_n_layers,
                        use_kv_cache=use_kv_cache,
                    )
                record["caption"] = trace_info["caption"]
                record["generated_token_ids"] = trace_info["generated_ids"]
                record["generated_tokens"] = trace_info["token_texts"]
                record["stop_reason"] = trace_info["stop_reason"]
                record["cache_mode"] = trace_info["cache_mode"]

                noun_info = extract_noun_info_from_token_ids(
                    token_ids=trace_info["generated_ids"],
                    tokenizer=tokenizer,
                    nlp=nlp,
                    max_caption_tokens=args.max_caption_tokens,
                )
                record["caption"] = noun_info["caption"]
                record["token_ids"] = noun_info["token_ids"]
                record["noun_chunks"] = noun_info["noun_chunks"]
                image_numeric_id = int(image_id)
                selected, selection_strategy, category_ids, matched_annotations = select_object_chunk_for_image(
                    noun_chunks=noun_info["noun_chunks"],
                    image_id=image_numeric_id,
                    coco_index=coco_index,
                )
                if selected is None:
                    selected = noun_info["selected"]
                    selection_strategy = noun_info["selection_strategy"]
                    category_ids = match_coco_category_ids(
                        head_text=selected.get("head_text") if selected else None,
                        phrase_text=selected.get("text") if selected else None,
                        coco_index=coco_index,
                    )
                    matched_annotations = [
                        ann
                        for ann in coco_index["anns_by_image_id"].get(image_numeric_id, [])
                        if int(ann["category_id"]) in category_ids
                    ]
                record["selection_strategy"] = selection_strategy

                if selected is not None:
                    record["first_noun_phrase"] = selected["text"]
                    record["phrase_token_start"] = selected["token_start"]
                    record["phrase_token_end"] = selected["token_end"]
                    record["noun_head"] = selected["head_text"]
                    record["head_token_start"] = selected["head_token_start"]
                    record["head_token_end"] = selected["head_token_end"]
                else:
                    record["first_noun_phrase"] = None
                    record["phrase_token_start"] = -1
                    record["phrase_token_end"] = -1
                    record["noun_head"] = None
                    record["head_token_start"] = -1
                    record["head_token_end"] = -1

                phrase_map, phrase_valid = aggregate_step_maps(
                    trace_info["step_maps"],
                    record["phrase_token_start"],
                    record["phrase_token_end"],
                )
                head_map, head_valid = aggregate_step_maps(
                    trace_info["step_maps"],
                    record["head_token_start"],
                    record["head_token_end"],
                )
                record["trace_valid"] = bool(phrase_valid or head_valid)
                if record["trace_valid"]:
                    trace_valid_count += 1
                record["phrase_map_stats"] = map_stats(phrase_map)
                record["head_map_stats"] = map_stats(head_map)

                image_info = coco_index["images_by_id"].get(image_numeric_id, {})
                matched_boxes = [ann["bbox"] for ann in matched_annotations]
                matched_category_names = [
                    coco_index["categories_by_id"][int(cat_id)]["name"]
                    for cat_id in category_ids
                    if int(cat_id) in coco_index["categories_by_id"]
                ]
                record["matched_coco_category_ids"] = category_ids
                record["matched_coco_category_names"] = matched_category_names

                phrase_eval = evaluate_trace_map_against_boxes(phrase_map, image_info, matched_boxes)
                head_eval = evaluate_trace_map_against_boxes(head_map, image_info, matched_boxes)
                record["phrase_bbox_eval"] = phrase_eval
                record["head_bbox_eval"] = head_eval

                if phrase_eval.get("valid"):
                    phrase_eval_valid_count += 1
                    phrase_inside_mass_total += float(phrase_eval["inside_mass"])
                    phrase_top10_total += float(phrase_eval["top10_patch_in_box_rate"])
                    phrase_argmax_in_box_count += int(bool(phrase_eval["argmax_in_box"]))
                if head_eval.get("valid"):
                    head_eval_valid_count += 1
                    head_inside_mass_total += float(head_eval["inside_mass"])
                    head_top10_total += float(head_eval["top10_patch_in_box_rate"])
                    head_argmax_in_box_count += int(bool(head_eval["argmax_in_box"]))

                if args.save_debug_images and debug_saved < args.debug_limit:
                    sample_dir = os.path.join(args.output_dir, "samples", image_id)
                    ensure_output_dir(sample_dir)
                    source_path = os.path.join(sample_dir, "input_processed.png")
                    phrase_map_path = os.path.join(sample_dir, "phrase_trace_map.png")
                    head_map_path = os.path.join(sample_dir, "head_trace_map.png")
                    phrase_overlay_path = os.path.join(sample_dir, "phrase_trace_overlay.png")
                    head_overlay_path = os.path.join(sample_dir, "head_trace_overlay.png")
                    bbox_overlay_path = os.path.join(sample_dir, "bbox_overlay.png")
                    tensor_to_pil(source_vis).save(source_path)
                    make_attention_heatmap(phrase_map).save(phrase_map_path)
                    make_attention_heatmap(head_map).save(head_map_path)
                    tensor_to_pil(make_attention_overlay(source_vis, phrase_map, color=(1.0, 0.15, 0.15))).save(phrase_overlay_path)
                    tensor_to_pil(make_attention_overlay(source_vis, head_map, color=(0.15, 0.65, 1.0))).save(head_overlay_path)
                    draw_boxes_on_processed_image(source_vis, matched_boxes, int(original_size[0]), int(original_size[1])).save(bbox_overlay_path)
                    record["debug_paths"] = {
                        "input_processed": os.path.abspath(source_path),
                        "phrase_trace_map": os.path.abspath(phrase_map_path),
                        "head_trace_map": os.path.abspath(head_map_path),
                        "phrase_trace_overlay": os.path.abspath(phrase_overlay_path),
                        "head_trace_overlay": os.path.abspath(head_overlay_path),
                        "bbox_overlay": os.path.abspath(bbox_overlay_path),
                    }
                    save_json(os.path.join(sample_dir, "record.json"), record)
                    debug_saved += 1

            except Exception as exc:
                record["error"] = f"{type(exc).__name__}: {exc}"
                error_count += 1
                record.setdefault("trace_valid", False)
                record.setdefault("phrase_map_stats", map_stats(phrase_map))
                record.setdefault("head_map_stats", map_stats(head_map))

            finalized = shard_writer.append(record, image_id, phrase_map, head_map)
            for record_out in finalized:
                metadata_file.write(json.dumps(record_out, ensure_ascii=False) + "\n")
            pbar.update(1)
            pbar.set_postfix(
                valid=trace_valid_count,
                errors=error_count,
                phrase_eval=phrase_eval_valid_count,
                head_eval=head_eval_valid_count,
                cache="on" if use_kv_cache else "off",
            )

        finalized = shard_writer.flush()
        for record_out in finalized:
            metadata_file.write(json.dumps(record_out, ensure_ascii=False) + "\n")

    metadata_records = []
    with open(metadata_path, "r", encoding="utf-8") as metadata_file:
        for line in metadata_file:
            line = line.strip()
            if line:
                metadata_records.append(json.loads(line))
    analysis_paths = write_debug_rankings(args.output_dir, metadata_records, top_k=args.ranking_top_k)

    summary: Dict[str, Any] = {
        "image_dir": os.path.abspath(args.image_dir),
        "output_dir": os.path.abspath(args.output_dir),
        "coco_instances_json": os.path.abspath(args.coco_instances_json),
        "model_path": args.model_path,
        "caption_prompt": args.caption_prompt,
        "caption_prompt_built": caption_prompt_built,
        "spacy_model": args.spacy_model,
        "device": str(model.device),
        "dtype": str(model.dtype),
        "caption_max_new_tokens": args.caption_max_new_tokens,
        "max_caption_tokens": args.max_caption_tokens,
        "trace_last_n_layers": int(args.trace_last_n_layers),
        "use_kv_cache": bool(use_kv_cache),
        "force_eager_attention": bool(args.force_eager_attention),
        "attention_source": args.attention_source,
        "seed": args.seed,
        "num_shards": int(args.num_shards),
        "shard_index": int(args.shard_index),
        "requested_samples": int(len(image_paths)),
        "global_requested_samples": int(global_requested_samples),
        "trace_valid_count": int(trace_valid_count),
        "trace_invalid_count": int(len(image_paths) - trace_valid_count),
        "error_count": int(error_count),
        "phrase_bbox_eval_count": int(phrase_eval_valid_count),
        "head_bbox_eval_count": int(head_eval_valid_count),
        "phrase_inside_mass_mean": (phrase_inside_mass_total / phrase_eval_valid_count) if phrase_eval_valid_count else None,
        "head_inside_mass_mean": (head_inside_mass_total / head_eval_valid_count) if head_eval_valid_count else None,
        "phrase_argmax_in_box_rate": (phrase_argmax_in_box_count / phrase_eval_valid_count) if phrase_eval_valid_count else None,
        "head_argmax_in_box_rate": (head_argmax_in_box_count / head_eval_valid_count) if head_eval_valid_count else None,
        "phrase_top10_patch_in_box_rate_mean": (phrase_top10_total / phrase_eval_valid_count) if phrase_eval_valid_count else None,
        "head_top10_patch_in_box_rate_mean": (head_top10_total / head_eval_valid_count) if head_eval_valid_count else None,
        "debug_saved_count": int(debug_saved),
        "metadata_jsonl": os.path.abspath(metadata_path),
        "map_shards": list(shard_writer.shard_paths),
        "analysis_paths": analysis_paths,
        "notes": [
            (
                "Caption traces are aggregated from a single teacher-forced forward on the full sequence (re-read)."
                if args.attention_source == "teacher_forced"
                else "Caption traces are aggregated from generation-time attentions, not teacher-forced re-reads."
            ),
            "The step map for a generated token uses the query row at its predicting position over image patches.",
            "Debug heatmaps and overlays use percentile-normalized visualization, so bright regions are for inspection only and not raw probabilities.",
            "Weak evaluation uses COCO instance boxes matched from the noun head / phrase to category names with simple heuristics.",
            "BBox metrics are approximate and operate on the 16x16 image-token grid.",
            "This diagnostic is intended to sanity-check whether the first-object prior is usable before full-cache generation.",
        ],
    }

    save_json(summary_path, summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
