#!/usr/bin/env python
"""
Build Stage A cache for CaptionSlot:

image -> concise predicted caption -> first-object noun phrase / head ->
teacher-forced prior maps (phrase_map / head_map)

Outputs:
- metadata.jsonl: small per-image metadata
- shards/cache_XXXXX.pt: sharded phrase/head prior tensors
- summary.json: run summary
- samples/<image_id>/*: optional debug visualizations
"""

import argparse
import json
import math
import os
import random
import sys
import types
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import spacy
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parents[1]
INFERENCE_ROOT = REPO_ROOT / "inference"
for path in (str(REPO_ROOT), str(INFERENCE_ROOT)):
    if path not in sys.path:
        sys.path.append(path)

# Some scale_rae imports initialize ezcolorlog, which tries to import IPython.
# In this environment that can cascade into a sqlite/libstdc++ mismatch, so we
# provide a minimal stub before importing the model stack.
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Stage A predicted-caption prior cache.")
    parser.add_argument("--image-dir", required=True, help="Directory containing images.")
    parser.add_argument("--output-dir", required=True, help="Output directory for Stage A cache.")
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--caption-prompt", default=DEFAULT_CAPTION_PROMPT)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=["fp16", "bf16", "fp32"], default="bf16")
    parser.add_argument("--caption-max-new-tokens", type=int, default=64)
    parser.add_argument("--max-caption-tokens", type=int, default=64)
    parser.add_argument("--attention-temperature", type=float, default=1.0)
    parser.add_argument("--normalize-attention-tokens", action="store_true")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--map-shard-size", type=int, default=1000)
    parser.add_argument("--spacy-model", default="en_core_web_sm")
    parser.add_argument("--save-debug-images", action="store_true")
    parser.add_argument("--debug-limit", type=int, default=50)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
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


def gaussian_window(window_size: int, sigma: float, channels: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    coords = torch.arange(window_size, device=device, dtype=dtype) - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    window_2d = torch.outer(g, g)
    return window_2d.view(1, 1, window_size, window_size).expand(channels, 1, window_size, window_size).contiguous()


def tensor_to_pil(image_tensor: torch.Tensor) -> Image.Image:
    array = image_tensor.detach().float().cpu().clamp(0, 1).permute(1, 2, 0).numpy()
    array = np.clip(array * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(array)


def make_attention_overlay(source: torch.Tensor, attn: torch.Tensor, color: Tuple[float, float, float]) -> torch.Tensor:
    c, height, width = source.shape
    n_patches = int(attn.numel())
    grid_side = int(n_patches ** 0.5)
    if grid_side * grid_side != n_patches:
        return source.clone()
    attn_2d = attn.view(grid_side, grid_side).unsqueeze(0).unsqueeze(0)
    attn_map = F.interpolate(attn_2d, size=(height, width), mode="bilinear", align_corners=False)[0, 0]
    attn_map = attn_map.clamp(0.0, 1.0)
    color_t = torch.tensor(color, dtype=torch.float32).view(3, 1, 1)
    overlay = source.float() * (1.0 - 0.6 * attn_map.unsqueeze(0)) + color_t * 0.6 * attn_map.unsqueeze(0)
    return overlay.clamp(0.0, 1.0)


def make_attention_heatmap(attn: torch.Tensor) -> Image.Image:
    n_patches = int(attn.numel())
    grid_side = int(n_patches ** 0.5)
    if grid_side * grid_side != n_patches:
        raise ValueError(f"Expected square patch count, got {n_patches}")
    attn_2d = attn.view(grid_side, grid_side).detach().float().cpu().numpy()
    attn_2d = np.clip(attn_2d * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(attn_2d, mode="L").resize((224, 224), resample=Image.BILINEAR)


def save_json(path: str, payload: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def find_token_span(offsets: Sequence[Tuple[int, int]], char_start: int, char_end: int) -> Tuple[Optional[int], Optional[int]]:
    tok_start = None
    tok_end = None
    for idx, (start, end) in enumerate(offsets):
        if start == 0 and end == 0 and idx > 0:
            continue
        if start < char_end and end > char_start:
            if tok_start is None:
                tok_start = idx
            tok_end = idx + 1
    return tok_start, tok_end


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


def extract_noun_info(
    caption: str,
    tokenizer,
    nlp,
    max_caption_tokens: int,
) -> Dict[str, Any]:
    encoding = tokenizer(
        caption,
        add_special_tokens=False,
        truncation=True,
        max_length=max_caption_tokens,
    )
    token_ids = list(encoding.input_ids)
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
        if tok_start is None or tok_end is None or tok_end > len(token_ids):
            continue
        if head_start is None or head_end is None or head_end > len(token_ids):
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
            if (
                tok_start is not None
                and tok_end is not None
                and head_start is not None
                and head_end is not None
                and tok_end <= len(token_ids)
                and head_end <= len(token_ids)
            ):
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
        "caption": caption_text,
        "token_ids": token_ids,
        "noun_chunks": noun_chunks,
        "selected": selected,
        "selection_strategy": selection_strategy,
    }


def generate_caption_single(
    model,
    tokenizer,
    prompt_input_ids_single: torch.Tensor,
    image_tensor_single: torch.Tensor,
    caption_gen_kwargs: Dict[str, Any],
) -> str:
    with torch.inference_mode():
        output_ids, _ = model.generate(
            prompt_input_ids_single,
            images=image_tensor_single,
            **caption_gen_kwargs,
        )
    return normalize_caption(
        tokenizer.decode(
            output_ids[0],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
    )


def compute_token_map(
    caption_hidden: torch.Tensor,
    image_hidden: torch.Tensor,
    token_start: int,
    token_end: int,
    normalize_tokens: bool,
    temperature: float,
) -> torch.Tensor:
    if token_start < 0 or token_end <= token_start or token_end > caption_hidden.shape[1]:
        return image_hidden.new_zeros((image_hidden.shape[0], image_hidden.shape[1]), dtype=torch.float32)
    query = caption_hidden[:, token_start:token_end, :].mean(dim=1).float()
    image_hidden = image_hidden.float()
    dim = image_hidden.shape[-1]
    if normalize_tokens:
        query = F.layer_norm(query, (dim,))
        image_hidden = F.layer_norm(image_hidden, (dim,))
    temp = max(float(temperature), 1e-6)
    logits = torch.einsum("bd,bnd->bn", query, image_hidden) / (math.sqrt(dim) * temp)
    return torch.sigmoid(logits)


def compute_prior_maps_batched(
    model,
    tokenizer,
    prompt_input_ids: torch.Tensor,
    image_tensor_batch: torch.Tensor,
    caption_token_ids_list: Sequence[Sequence[int]],
    selected_list: Sequence[Optional[Dict[str, Any]]],
    normalize_tokens: bool,
    temperature: float,
) -> List[Dict[str, Any]]:
    """Compute prior maps per-sample to avoid attention mask padding dtype issues with SDPA."""
    if image_tensor_batch.ndim != 5:
        raise ValueError(f"Expected image tensor batch [B,1,C,H,W], got {tuple(image_tensor_batch.shape)}")
    if int(prompt_input_ids.shape[0]) != int(image_tensor_batch.shape[0]):
        raise ValueError("prompt_input_ids batch size must match image batch size.")
    if len(caption_token_ids_list) != int(image_tensor_batch.shape[0]):
        raise ValueError("caption_token_ids_list length must match image batch size.")
    if len(selected_list) != int(image_tensor_batch.shape[0]):
        raise ValueError("selected_list length must match image batch size.")

    batch_size = int(image_tensor_batch.shape[0])
    num_image_tokens = int(getattr(model, "num_image_tokens", 256))
    image_positions = (prompt_input_ids[0] == IMAGE_TOKEN_INDEX).nonzero(as_tuple=False).flatten()
    if image_positions.numel() != 1:
        raise RuntimeError(f"Expected exactly one image placeholder, got {int(image_positions.numel())}")
    image_start = int(image_positions[0].item())
    image_end = image_start + num_image_tokens

    results: List[Dict[str, Any]] = []
    with torch.inference_mode():
        for row_idx in range(batch_size):
            selected = selected_list[row_idx]
            token_ids = caption_token_ids_list[row_idx]

            single_prompt = prompt_input_ids[row_idx : row_idx + 1]
            single_image = image_tensor_batch[row_idx : row_idx + 1]

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
                single_prompt, None, None, None, None,
                images=single_image, image_embeds=None,
            )

            if prefix_inputs_embeds is None:
                raise RuntimeError("prepare_inputs_labels_for_multimodal returned no inputs_embeds.")

            if token_ids and selected is not None:
                caption_tensor = torch.tensor(list(token_ids), device=model.device, dtype=torch.long).unsqueeze(0)
                caption_embeds = model.get_model().embed_tokens(caption_tensor)
                caption_embeds = caption_embeds.to(device=prefix_inputs_embeds.device, dtype=prefix_inputs_embeds.dtype)
                inputs_embeds = torch.cat([prefix_inputs_embeds, caption_embeds], dim=1)
            else:
                inputs_embeds = prefix_inputs_embeds

            outputs = model.forward(
                input_ids=None,
                attention_mask=None,
                position_ids=None,
                past_key_values=None,
                inputs_embeds=inputs_embeds,
                use_cache=False,
                return_dict=True,
                decoding=False,
                answer_token_mask=None,
                guidance_level=None,
            )

            hidden = outputs.hidden_states
            if not torch.is_tensor(hidden):
                raise RuntimeError("Expected tensor hidden_states from model.forward.")

            row_image_hidden = hidden[:, image_start:image_end, :]

            if selected is None or not token_ids:
                zero_map = row_image_hidden.new_zeros((1, row_image_hidden.shape[1]), dtype=torch.float32)
                results.append({
                    "phrase_map": zero_map,
                    "head_map": zero_map.clone(),
                    "prior_valid": False,
                    "selection_strategy": "none",
                })
                continue

            caption_start = int(prefix_inputs_embeds.shape[1])
            row_caption_hidden = hidden[:, caption_start:caption_start + len(token_ids), :]

            phrase_map = compute_token_map(
                caption_hidden=row_caption_hidden,
                image_hidden=row_image_hidden,
                token_start=int(selected["token_start"]),
                token_end=int(selected["token_end"]),
                normalize_tokens=normalize_tokens,
                temperature=temperature,
            )
            head_map = compute_token_map(
                caption_hidden=row_caption_hidden,
                image_hidden=row_image_hidden,
                token_start=int(selected["head_token_start"]),
                token_end=int(selected["head_token_end"]),
                normalize_tokens=normalize_tokens,
                temperature=temperature,
            )
            results.append({
                "phrase_map": phrase_map,
                "head_map": head_map,
                "prior_valid": True,
                "selection_strategy": "selected",
            })

    return results


def compute_prior_map_single(
    model,
    tokenizer,
    prompt_input_ids_single: torch.Tensor,
    image_tensor_single: torch.Tensor,
    caption_token_ids: Sequence[int],
    selected: Optional[Dict[str, Any]],
    normalize_tokens: bool,
    temperature: float,
) -> Dict[str, Any]:
    results = compute_prior_maps_batched(
        model=model,
        tokenizer=tokenizer,
        prompt_input_ids=prompt_input_ids_single,
        image_tensor_batch=image_tensor_single,
        caption_token_ids_list=[caption_token_ids],
        selected_list=[selected],
        normalize_tokens=normalize_tokens,
        temperature=temperature,
    )
    return results[0]


def denormalize_image_tensor(image_tensor: torch.Tensor, image_processor) -> torch.Tensor:
    mean = torch.tensor(image_processor[0].image_mean, device=image_tensor.device, dtype=image_tensor.dtype).view(1, 3, 1, 1)
    std = torch.tensor(image_processor[0].image_std, device=image_tensor.device, dtype=image_tensor.dtype).view(1, 3, 1, 1)
    return (image_tensor * std + mean).clamp(0, 1)


def map_stats(attn: torch.Tensor) -> Dict[str, float]:
    arr = attn.detach().float()
    probs = arr / arr.sum().clamp(min=1e-8)
    entropy = float((-(probs * probs.clamp(min=1e-8).log()).sum()).item())
    return {
        "min": float(arr.min().item()),
        "max": float(arr.max().item()),
        "mean": float(arr.mean().item()),
        "entropy": entropy,
    }


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
        empty_summary: Dict[str, Any] = {
            "image_dir": os.path.abspath(args.image_dir),
            "output_dir": os.path.abspath(args.output_dir),
            "model_path": args.model_path,
            "caption_prompt": args.caption_prompt,
            "spacy_model": args.spacy_model,
            "device": args.device,
            "dtype": args.dtype,
            "caption_batch_size": args.caption_batch_size,
            "caption_max_new_tokens": args.caption_max_new_tokens,
            "max_caption_tokens": args.max_caption_tokens,
            "attention_temperature": args.attention_temperature,
            "normalize_attention_tokens": bool(args.normalize_attention_tokens),
            "seed": args.seed,
            "num_shards": int(args.num_shards),
            "shard_index": int(args.shard_index),
            "requested_samples": 0,
            "global_requested_samples": int(global_requested_samples),
            "prior_valid_count": 0,
            "prior_invalid_count": 0,
            "error_count": 0,
            "fallback_or_missing_count": 0,
            "debug_saved_count": 0,
            "metadata_jsonl": os.path.abspath(metadata_path),
            "map_shards": [],
            "notes": [
                "This shard had no assigned samples after applying num_shards/shard_index.",
            ],
        }
        save_json(summary_path, empty_summary)
        print(json.dumps(empty_summary, indent=2, ensure_ascii=False))
        return

    print(f"Loading spaCy model: {args.spacy_model}")
    nlp = spacy.load(args.spacy_model)

    dtype = resolve_dtype(args.dtype, args.device)
    tokenizer, model, image_processor, _context_len = load_scale_rae_model(
        model_path=args.model_path,
        device=args.device,
        dtype=dtype,
    )
    start_id, end_id, eos_id = prepare_special_token_ids(tokenizer)

    caption_prompt_built = build_prompt(args.caption_prompt, model_config=model.config, with_image=True, num_frames=1)
    prompt_input_ids = tokenize_prompt(caption_prompt_built, tokenizer, device=model.device)

    caption_gen_kwargs = dict(
        output_image=False,
        do_sample=False,
        temperature=0.0,
        use_customize_greedy=True,
        top_p=None,
        num_beams=1,
        max_new_tokens=args.caption_max_new_tokens,
        use_cache=True,
        start_image_token_id=start_id,
        end_image_token_id=end_id,
        eos_token_id=eos_id,
        guidance_level=1.0,
    )

    debug_saved = 0
    total_prior_valid = 0
    total_fallback = 0
    total_errors = 0

    shard_writer = MapShardWriter(os.path.join(args.output_dir, "shards"), shard_size=args.map_shard_size)

    with open(metadata_path, "w", encoding="utf-8") as metadata_file, tqdm(
        total=len(image_paths),
        desc="StageA Cache",
        unit="img",
        dynamic_ncols=True,
    ) as pbar:
        for image_path in image_paths:
            image_id = Path(image_path).stem
            record: Dict[str, Any] = {
                "image": os.path.abspath(image_path),
                "image_id": image_id,
                "caption_prompt": args.caption_prompt,
                "caption_prompt_built": caption_prompt_built,
                "prior_valid": False,
                "num_shards": int(args.num_shards),
                "shard_index": int(args.shard_index),
            }

            try:
                # --- 1. Load & preprocess image ---
                image = load_image_rgb(image_path)
                image_tensor, _ = preprocess_single_image(
                    image, image_processor,
                    device=model.device, dtype=model.dtype,
                )

                # --- 2. Generate caption (single, with KV cache) ---
                caption = generate_caption_single(
                    model, tokenizer, prompt_input_ids, image_tensor,
                    caption_gen_kwargs,
                )
                record["caption"] = caption

                # --- 3. Extract noun phrase ---
                noun_info = extract_noun_info(
                    caption=caption, tokenizer=tokenizer,
                    nlp=nlp, max_caption_tokens=args.max_caption_tokens,
                )
                record["caption"] = noun_info["caption"]
                record["token_ids"] = noun_info["token_ids"]
                record["noun_chunks"] = noun_info["noun_chunks"]
                record["selection_strategy"] = noun_info["selection_strategy"]

                selected = noun_info["selected"]
                if selected is not None:
                    record["first_noun_phrase"] = selected["text"]
                    record["phrase_token_start"] = selected["token_start"]
                    record["phrase_token_end"] = selected["token_end"]
                    record["noun_head"] = selected["head_text"]
                    record["head_token_start"] = selected["head_token_start"]
                    record["head_token_end"] = selected["head_token_end"]
                else:
                    total_fallback += 1
                    record["first_noun_phrase"] = None
                    record["phrase_token_start"] = -1
                    record["phrase_token_end"] = -1
                    record["noun_head"] = None
                    record["head_token_start"] = -1
                    record["head_token_end"] = -1

                # --- 4. Compute prior maps (single forward, no padding) ---
                prior_results = compute_prior_maps_batched(
                    model=model, tokenizer=tokenizer,
                    prompt_input_ids=prompt_input_ids,
                    image_tensor_batch=image_tensor.unsqueeze(0) if image_tensor.ndim == 4 else image_tensor,
                    caption_token_ids_list=[noun_info["token_ids"]],
                    selected_list=[selected],
                    normalize_tokens=args.normalize_attention_tokens,
                    temperature=args.attention_temperature,
                )
                prior_info = prior_results[0]
                phrase_map = prior_info["phrase_map"][0].detach().cpu().float()
                head_map = prior_info["head_map"][0].detach().cpu().float()
                record["prior_valid"] = bool(prior_info["prior_valid"])
                if record["prior_valid"]:
                    total_prior_valid += 1

            except Exception as exc:
                record["error"] = f"{type(exc).__name__}: {exc}"
                total_errors += 1
                record.setdefault("selection_strategy", "error")
                record.setdefault("first_noun_phrase", None)
                record.setdefault("noun_head", None)
                phrase_map = torch.zeros(256, dtype=torch.float32)
                head_map = torch.zeros(256, dtype=torch.float32)

            record["phrase_map_stats"] = map_stats(phrase_map)
            record["head_map_stats"] = map_stats(head_map)

            if args.save_debug_images and debug_saved < args.debug_limit and record.get("prior_valid"):
                sample_dir = os.path.join(args.output_dir, "samples", image_id)
                ensure_output_dir(sample_dir)
                source = denormalize_image_tensor(image_tensor, image_processor)[0].detach().cpu()
                source_path = os.path.join(sample_dir, "input_processed.png")
                phrase_heatmap_path = os.path.join(sample_dir, "phrase_map.png")
                head_heatmap_path = os.path.join(sample_dir, "head_map.png")
                phrase_overlay_path = os.path.join(sample_dir, "phrase_overlay.png")
                head_overlay_path = os.path.join(sample_dir, "head_overlay.png")
                tensor_to_pil(source).save(source_path)
                make_attention_heatmap(phrase_map).save(phrase_heatmap_path)
                make_attention_heatmap(head_map).save(head_heatmap_path)
                tensor_to_pil(make_attention_overlay(source, phrase_map, color=(1.0, 0.15, 0.15))).save(phrase_overlay_path)
                tensor_to_pil(make_attention_overlay(source, head_map, color=(0.15, 0.65, 1.0))).save(head_overlay_path)
                record["debug_paths"] = {
                    "input_processed": os.path.abspath(source_path),
                    "phrase_map": os.path.abspath(phrase_heatmap_path),
                    "head_map": os.path.abspath(head_heatmap_path),
                    "phrase_overlay": os.path.abspath(phrase_overlay_path),
                    "head_overlay": os.path.abspath(head_overlay_path),
                }
                save_json(os.path.join(sample_dir, "record.json"), record)
                debug_saved += 1

            finalized = shard_writer.append(record, image_id, phrase_map, head_map)
            for record_out in finalized:
                metadata_file.write(json.dumps(record_out, ensure_ascii=False) + "\n")
            pbar.update(1)
            pbar.set_postfix(valid=total_prior_valid, errors=total_errors)

        finalized = shard_writer.flush()
        for record_out in finalized:
            metadata_file.write(json.dumps(record_out, ensure_ascii=False) + "\n")

    summary: Dict[str, Any] = {
        "image_dir": os.path.abspath(args.image_dir),
        "output_dir": os.path.abspath(args.output_dir),
        "model_path": args.model_path,
        "caption_prompt": args.caption_prompt,
        "caption_prompt_built": caption_prompt_built,
        "spacy_model": args.spacy_model,
        "device": str(model.device),
        "dtype": str(model.dtype),
        "caption_max_new_tokens": args.caption_max_new_tokens,
        "max_caption_tokens": args.max_caption_tokens,
        "attention_temperature": args.attention_temperature,
        "normalize_attention_tokens": bool(args.normalize_attention_tokens),
        "seed": args.seed,
        "num_shards": int(args.num_shards),
        "shard_index": int(args.shard_index),
        "requested_samples": len(image_paths),
        "global_requested_samples": int(global_requested_samples),
        "prior_valid_count": int(total_prior_valid),
        "prior_invalid_count": int(len(image_paths) - total_prior_valid),
        "error_count": int(total_errors),
        "fallback_or_missing_count": int(total_fallback),
        "debug_saved_count": int(debug_saved),
        "metadata_jsonl": os.path.abspath(metadata_path),
        "map_shards": list(shard_writer.shard_paths),
        "notes": [
            "Captions are generated from the image using the concise caption prompt.",
            "The first object prior defaults to the first valid noun chunk; if none exists, it falls back to the first noun/proper-noun subtree.",
            "phrase_map and head_map are computed by a teacher-forced forward over image + fixed caption, then dot-producting caption hidden states against image patch hidden states.",
            "Each image is processed individually: caption generation uses model.generate with KV cache, prior-map extraction uses a single teacher-forced forward pass.",
            "phrase_map is intended as the primary slot prior; head_map is stored for diagnostics and future ablations.",
            "Prior tensors are stored in sharded .pt files to avoid very large JSON artifacts.",
        ],
    }

    save_json(summary_path, summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
