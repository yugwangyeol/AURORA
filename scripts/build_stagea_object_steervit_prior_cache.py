#!/usr/bin/env python
"""
Build a Stage A object-centric SteerViT prior cache for AURORA CaptionSlot.

Pipeline:
image batch
  -> Scale-RAE object-wise refexp caption generation
  -> line parsing + cleaning + object text selection
  -> serialize object texts into a trainer-friendly caption/token span format
  -> SteerViT attention map extraction per object text
  -> save captionslot-compatible annotations + sharded prior tensors

The current AURORA CaptionSlot trainer expects:
- captionslot_annotations.json with `caption`, `token_ids`, and `noun_chunks`
- map shards with `head_maps` / `head_valid_mask` shaped `(max_slots, 256)`

So this builder keeps those field names for compatibility, while the
"noun_chunks" are actually object-text spans and the "head_maps" are SteerViT
object priors.
"""

import argparse
import json
import math
import os
import random
import re
import sys
import types
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
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

from utils.load_model import load_scale_rae_model  # type: ignore
from scale_rae.constants import (  # type: ignore
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
    IMAGE_PLACEHOLDER,
)
from scale_rae.conversation import conv_templates  # type: ignore
from scale_rae.mm_utils import tokenizer_image_token  # type: ignore


DEFAULT_MODEL_PATH = "/home/jovyan/data/Scale-RAE-Qwen1.5B_DiT2.4B"
DEFAULT_STEERVIT_SRC = "/home/jovyan/SteerViT/src"
DEFAULT_STEERVIT_CHECKPOINT = "steervit_dinov2_base.pth"
DEFAULT_OBJECT_PROMPT = """List the visible objects in the image, one object per line.
After the object lines, add one background line.
Output only lines in this format:
[OBJ1] detailed referring expression
[OBJ2] detailed referring expression
...
[BACKGROUND] background description
Rules:
- Each OBJ line should describe one distinct object as specifically as possible.
- Mention category, appearance, action, approximate location, or nearby context when useful.
- Do not impose a fixed maximum number of objects, but only include objects you can clearly see.
- Do not write an introduction, summary, or extra explanation.
- Write the descriptions in English.
"""

ALLOWED_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
BRACKETED_LINE_RE = re.compile(r"^\[([^\]]+)\]\s*(.*)$")
REQUIRED_OBJECT_FIELDS = ("refexp", "name", "location", "attributes", "action")
GENERIC_JUNK_TEXTS = {
    "",
    "<empty>",
    "empty",
    "background",
    "scene",
    "image",
    "photo",
    "object",
    "objects",
    "thing",
    "things",
    "item",
    "items",
    "stuff",
    "detailed referring expression",
    "ref",
    "obj",
    "obj1",
    "obj2",
    "obj3",
}


def ensure_output_dir(output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)


def prepare_special_token_ids(tokenizer) -> Tuple[int, int, int]:
    start_image_token_id = tokenizer.convert_tokens_to_ids("<im_start>")
    end_image_token_id = tokenizer.convert_tokens_to_ids("<im_end>")
    eos_token_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    return start_image_token_id, end_image_token_id, eos_token_id


def _maybe_add_image_tokens_to_prompt(qs: str, num_frames: int, use_im_se: bool) -> str:
    image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
    if IMAGE_PLACEHOLDER in qs:
        if use_im_se:
            qs = qs.replace(IMAGE_PLACEHOLDER, image_token_se * num_frames)
        else:
            qs = qs.replace(IMAGE_PLACEHOLDER, DEFAULT_IMAGE_TOKEN * num_frames)
    else:
        if use_im_se:
            qs = image_token_se * num_frames + "\n" + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN * num_frames + "\n" + qs
    return qs


def build_prompt(prompt_text: str, model_config, with_image: bool, num_frames: int = 1) -> str:
    qs = prompt_text
    if with_image:
        qs = _maybe_add_image_tokens_to_prompt(
            qs,
            num_frames=num_frames,
            use_im_se=bool(model_config.mm_use_im_start_end),
        )
    conv = conv_templates["qwen_2"].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    return conv.get_prompt()


def tokenize_prompt(prompt: str, tokenizer, device: torch.device) -> torch.Tensor:
    input_ids = tokenizer_image_token(
        prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
    ).unsqueeze(0)
    return input_ids.to(device)


def load_image_rgb(image_path: str) -> Image.Image:
    return Image.open(image_path).convert("RGB")


def preprocess_single_image(
    image: Image.Image,
    image_processor,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> Tuple[torch.Tensor, Tuple[int, int]]:
    image_tensor = image_processor[0].preprocess(image, return_tensors="pt")["pixel_values"][0]
    image_tensor = image_tensor.unsqueeze(0)
    image_tensor = image_tensor.to(device, dtype=dtype)
    return image_tensor, image.size


class _PrefetchImageDataset(torch.utils.data.Dataset):
    """Worker-side PIL decode + Scale-RAE preprocess + SteerViT transform."""

    def __init__(self, image_paths: Sequence[str], image_processor, steervit_transform) -> None:
        self.image_paths = list(image_paths)
        self.image_processor = image_processor
        self.steervit_transform = steervit_transform

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        path = self.image_paths[idx]
        image_id = Path(path).stem
        try:
            pil = Image.open(path).convert("RGB")
            caption_pixel = self.image_processor[0].preprocess(
                pil, return_tensors="pt"
            )["pixel_values"][0].contiguous()
            steervit_pixel = self.steervit_transform(pil).contiguous()
            return {
                "ok": True,
                "image_path": path,
                "image_id": image_id,
                "caption_pixel": caption_pixel,
                "steervit_pixel": steervit_pixel,
            }
        except Exception as exc:  # noqa: BLE001
            return {
                "ok": False,
                "image_path": path,
                "image_id": image_id,
                "error": f"{type(exc).__name__}: {exc}",
            }


def _identity_collate(batch: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return batch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build an AURORA object-wise SteerViT prior cache.")
    parser.add_argument("--image-dir", default=None, help="Directory containing images.")
    parser.add_argument(
        "--image-list-json",
        default=None,
        help="Optional JSON containing image paths. Supports a list or {'image_paths': [...]} payload.",
    )
    parser.add_argument("--output-dir", required=True, help="Output directory.")
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=["fp16", "bf16", "fp32"], default="fp32")
    parser.add_argument("--caption-batch-size", type=int, default=8)
    parser.add_argument("--caption-max-new-tokens", type=int, default=128)
    parser.add_argument("--max-caption-tokens", type=int, default=192)
    parser.add_argument("--max-slots", type=int, default=15)
    parser.add_argument("--grid-side", type=int, default=16)
    parser.add_argument("--steervit-src", default=DEFAULT_STEERVIT_SRC)
    parser.add_argument("--steervit-checkpoint", default=DEFAULT_STEERVIT_CHECKPOINT)
    parser.add_argument("--steervit-map-type", choices=("attention", "heatmap"), default="attention")
    parser.add_argument("--steervit-head-pooling", choices=("mean", "max", "min", "median"), default="mean")
    parser.add_argument("--steervit-gate-factor", type=float, default=1.0)
    parser.add_argument("--steervit-batch-size", type=int, default=64)
    parser.add_argument("--map-shard-size", type=int, default=1000)
    parser.add_argument("--prompt", default=DEFAULT_OBJECT_PROMPT)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--save-debug-images", action="store_true")
    parser.add_argument("--debug-limit", type=int, default=25)
    parser.add_argument("--disable-kv-cache", action="store_true")
    parser.add_argument("--tf32", action="store_true")
    parser.add_argument("--loader-num-workers", type=int, default=8)
    parser.add_argument("--fsync-every-n-batches", type=int, default=20)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def set_fast_math(tf32: bool) -> None:
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = bool(tf32)
        torch.backends.cudnn.allow_tf32 = bool(tf32)


def resolve_dtype(dtype_name: str, device_name: str) -> torch.dtype:
    if dtype_name == "fp16":
        dtype = torch.float16
    elif dtype_name == "bf16":
        dtype = torch.bfloat16
    else:
        dtype = torch.float32
    if str(device_name).startswith("cpu") and dtype != torch.float32:
        return torch.float32
    return dtype


def save_json(path: str, payload: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def list_image_files(image_dir: str) -> List[str]:
    return [
        str(path)
        for path in sorted(Path(image_dir).iterdir())
        if path.is_file() and path.suffix.lower() in ALLOWED_IMAGE_SUFFIXES
    ]


def take_shard(items: Sequence[Any], num_shards: int, shard_index: int) -> List[Any]:
    if num_shards < 1:
        raise ValueError("--num-shards must be at least 1.")
    if shard_index < 0 or shard_index >= num_shards:
        raise ValueError("--shard-index must be in [0, num_shards).")
    return [item for idx, item in enumerate(items) if idx % num_shards == shard_index]


def load_image_paths_from_json(json_path: str) -> List[str]:
    with open(json_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    if isinstance(payload, list):
        image_paths = payload
    elif isinstance(payload, dict):
        image_paths = payload.get("image_paths") or payload.get("images") or []
        if image_paths and isinstance(image_paths[0], dict):
            image_paths = [row["image_path"] for row in image_paths if "image_path" in row]
    else:
        raise TypeError(f"Unsupported image list JSON format: {type(payload)!r}")

    return [str(Path(item).expanduser().resolve()) for item in image_paths if item]


def resolve_requested_image_paths(args: argparse.Namespace) -> Tuple[List[str], str]:
    if args.image_list_json and args.image_dir:
        raise ValueError("Use either --image-dir or --image-list-json, not both.")
    if args.image_list_json:
        source = str(Path(args.image_list_json).expanduser().resolve())
        return load_image_paths_from_json(source), source
    if args.image_dir:
        source = str(Path(args.image_dir).expanduser().resolve())
        return list_image_files(source), source
    raise ValueError("One of --image-dir or --image-list-json must be provided.")


def normalize_caption(text: str) -> str:
    text = " ".join(str(text or "").strip().split())
    return text if text else "<empty>"


def normalize_generation_text(text: str) -> str:
    lines: List[str] = []
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    for raw_line in normalized.split("\n"):
        line = " ".join(raw_line.strip().split())
        if line:
            lines.append(line)
    return "\n".join(lines) if lines else "<empty>"


def strip_optional_bullet_prefix(line: str) -> str:
    for prefix in ("- ", "* ", "• "):
        if line.startswith(prefix):
            return line[len(prefix) :].strip()
    return line


def parse_object_payload(payload: str) -> Tuple[Dict[str, str], Dict[str, str], List[str]]:
    fields: Dict[str, str] = {}
    unknown_fields: Dict[str, str] = {}
    unparsed_segments: List[str] = []
    for raw_segment in payload.split("|"):
        segment = raw_segment.strip()
        if not segment:
            continue
        if ":" not in segment:
            unparsed_segments.append(segment)
            continue
        key, value = segment.split(":", 1)
        key_norm = re.sub(r"\s+", "_", key.strip().lower())
        value_norm = value.strip()
        if key_norm in REQUIRED_OBJECT_FIELDS:
            fields[key_norm] = value_norm
        else:
            unknown_fields[key_norm] = value_norm
    return fields, unknown_fields, unparsed_segments


def parse_objectwise_output(text: str) -> Dict[str, Any]:
    raw_lines = [line for line in text.split("\n") if line.strip()]
    normalized_lines = [strip_optional_bullet_prefix(line.strip()) for line in raw_lines]

    objects: List[Dict[str, Any]] = []
    background: Optional[str] = None
    extra_lines: List[str] = []

    for line in normalized_lines:
        bracket_match = BRACKETED_LINE_RE.match(line)
        if bracket_match is None:
            extra_lines.append(line)
            continue

        label = bracket_match.group(1).strip()
        payload = bracket_match.group(2).lstrip(": ").strip()
        label_lower = label.lower()
        if label_lower in {"background", "bg"}:
            background = payload or None
            continue

        index_match = re.search(r"(\d+)", label)
        obj_index = int(index_match.group(1)) if index_match is not None else len(objects) + 1
        parsed_fields, unknown_fields, unparsed_segments = parse_object_payload(payload)
        parse_mode = "structured" if parsed_fields else "refexp_only"
        record: Dict[str, Any] = {
            "obj_id": f"OBJ{obj_index}",
            "obj_index": obj_index,
            "raw_label": label,
            "raw_line": line,
            "parse_mode": parse_mode,
            "unknown_fields": unknown_fields,
            "unparsed_segments": unparsed_segments,
        }
        if parse_mode == "refexp_only":
            record["refexp"] = payload
            record["name"] = ""
            record["location"] = ""
            record["attributes"] = ""
            record["action"] = ""
        else:
            for field_name in REQUIRED_OBJECT_FIELDS:
                record[field_name] = parsed_fields.get(field_name, "")
        objects.append(record)

    objects.sort(key=lambda item: item["obj_index"])
    return {
        "raw_text": text,
        "raw_lines": raw_lines,
        "normalized_lines": normalized_lines,
        "objects": objects,
        "background": background,
        "extra_lines": extra_lines,
        "num_objects": len(objects),
        "has_background": background is not None,
    }


def move_vision_towers_to_device(model, device: torch.device, dtype: torch.dtype) -> None:
    towers = []
    try:
        towers = list(model.get_vision_tower_aux_list() or [])
    except Exception:
        towers = []
    for tower in towers:
        if tower is None:
            continue
        tower.to(device=device)
        if dtype != torch.float32:
            tower.to(dtype=dtype)
    projector = getattr(model.get_model(), "mm_projector_auxes", None)
    if projector is not None:
        try:
            projector.to(device=device)
            if dtype != torch.float32:
                projector.to(dtype=dtype)
        except Exception:
            pass


def get_runtime_dtypes(model) -> Dict[str, str]:
    info: Dict[str, str] = {}
    try:
        info["model_dtype_property"] = str(model.dtype)
    except Exception:
        info["model_dtype_property"] = "unknown"
    try:
        info["backbone_dtype"] = str(next(model.get_model().parameters()).dtype)
    except Exception:
        info["backbone_dtype"] = "unknown"
    try:
        info["lm_head_dtype"] = str(next(model.lm_head.parameters()).dtype)
    except Exception:
        info["lm_head_dtype"] = "unknown"
    try:
        vision_towers = model.get_vision_tower_aux_list()
        if vision_towers:
            tower = getattr(vision_towers[0], "vision_tower", None)
            if tower is not None:
                info["vision_tower_dtype"] = str(next(tower.parameters()).dtype)
            else:
                info["vision_tower_dtype"] = str(vision_towers[0].dtype)
        else:
            info["vision_tower_dtype"] = "none"
    except Exception:
        info["vision_tower_dtype"] = "unknown"
    return info


def get_runtime_devices(model) -> Dict[str, str]:
    info: Dict[str, str] = {}
    try:
        info["model_device_property"] = str(model.device)
    except Exception:
        info["model_device_property"] = "unknown"
    try:
        info["backbone_device"] = str(next(model.get_model().parameters()).device)
    except Exception:
        info["backbone_device"] = "unknown"
    try:
        info["lm_head_device"] = str(next(model.lm_head.parameters()).device)
    except Exception:
        info["lm_head_device"] = "unknown"
    try:
        vision_towers = model.get_vision_tower_aux_list()
        if vision_towers:
            tower = getattr(vision_towers[0], "vision_tower", None)
            if tower is not None:
                info["vision_tower_device"] = str(next(tower.parameters()).device)
            else:
                info["vision_tower_device"] = "unknown"
        else:
            info["vision_tower_device"] = "none"
    except Exception:
        info["vision_tower_device"] = "unknown"
    try:
        projector = getattr(model.get_model(), "mm_projector_auxes", None)
        if projector is not None:
            info["mm_projector_device"] = str(next(projector.parameters()).device)
        else:
            info["mm_projector_device"] = "none"
    except Exception:
        info["mm_projector_device"] = "unknown"
    return info


def build_caption_batch(
    model,
    prompt_input_ids: torch.Tensor,
    images_tensor: torch.Tensor,
) -> Tuple[torch.Tensor, Any, int]:
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
        images=images_tensor,
        image_embeds=None,
    )
    if prefix_inputs_embeds is None:
        raise RuntimeError("prepare_inputs_labels_for_multimodal returned no inputs_embeds.")
    prefix_len = int(prefix_inputs_embeds.shape[1])
    return prefix_inputs_embeds, extra_mm, prefix_len


def generate_captions_batched_kv(
    model,
    tokenizer,
    prompt_input_ids: torch.Tensor,
    images_tensor: torch.Tensor,
    eos_token_id: int,
    max_new_tokens: int,
    start_image_token_id: int,
    end_image_token_id: int,
    use_kv_cache: bool,
) -> Dict[str, Any]:
    prefix_inputs_embeds, extra_mm, prefix_len = build_caption_batch(
        model=model,
        prompt_input_ids=prompt_input_ids,
        images_tensor=images_tensor,
    )
    device = model.device
    inputs_embeds = prefix_inputs_embeds.to(device=device)
    batch_size = int(inputs_embeds.shape[0])

    stop_ids_tensor = torch.tensor(
        [int(eos_token_id), int(start_image_token_id), int(end_image_token_id)],
        dtype=torch.long,
        device=device,
    )

    # GPU-side generation buffers: no per-step .tolist() sync
    token_buffer = torch.full(
        (batch_size, max_new_tokens), int(eos_token_id), dtype=torch.long, device=device
    )
    write_mask_buffer = torch.zeros(
        (batch_size, max_new_tokens), dtype=torch.bool, device=device
    )
    finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
    stop_token_buffer = torch.full(
        (batch_size,), -1, dtype=torch.long, device=device
    )

    full_attention_mask = torch.ones(
        (batch_size, prefix_len + max_new_tokens), device=device, dtype=torch.bool
    )

    past_key_values = None
    current_inputs_embeds = inputs_embeds
    current_context_len = prefix_len
    last_step = max_new_tokens
    early_check_every = 16

    with torch.inference_mode():
        for step in range(max_new_tokens):
            attention_mask = full_attention_mask[:, :current_context_len]
            outputs = model.forward(
                input_ids=None,
                attention_mask=attention_mask,
                position_ids=None,
                past_key_values=past_key_values,
                inputs_embeds=current_inputs_embeds if use_kv_cache else inputs_embeds,
                use_cache=use_kv_cache,
                return_dict=True,
                decoding=False,
                guidance_level=1.0,
                output_attentions=False,
                output_hidden_states=False,
                answer_token_mask=extra_mm,
            )
            next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1)
            is_stop = (next_token.unsqueeze(1) == stop_ids_tensor.unsqueeze(0)).any(dim=1)
            active = ~finished
            newly_stopped = active & is_stop
            write_mask = active & ~is_stop

            token_buffer[:, step] = torch.where(write_mask, next_token, token_buffer[:, step])
            write_mask_buffer[:, step] = write_mask

            not_yet_recorded = stop_token_buffer < 0
            stop_token_buffer = torch.where(
                newly_stopped & not_yet_recorded, next_token, stop_token_buffer
            )
            finished = finished | is_stop

            if (step + 1) % early_check_every == 0 or (step + 1) == max_new_tokens:
                if bool(finished.all()):
                    last_step = step + 1
                    break

            next_token_masked = next_token.masked_fill(finished, int(eos_token_id))
            next_token_embed = model.get_model().embed_tokens(next_token_masked.unsqueeze(1))
            next_token_embed = next_token_embed.to(device=inputs_embeds.device, dtype=inputs_embeds.dtype)

            if use_kv_cache:
                past_key_values = outputs.past_key_values
                current_inputs_embeds = next_token_embed
                current_context_len += 1
            else:
                inputs_embeds = torch.cat((inputs_embeds, next_token_embed), dim=1)
                current_inputs_embeds = inputs_embeds
                current_context_len = int(inputs_embeds.shape[1])

    # Single CPU sync at end.
    token_cpu = token_buffer[:, :last_step].cpu().tolist()
    mask_cpu = write_mask_buffer[:, :last_step].cpu().tolist()
    stop_tok_cpu = stop_token_buffer.cpu().tolist()

    generated_ids: List[List[int]] = []
    stop_reasons: List[str] = []
    for row in range(batch_size):
        row_tokens = [tid for tid, m in zip(token_cpu[row], mask_cpu[row]) if m]
        generated_ids.append(row_tokens)
        if stop_tok_cpu[row] >= 0:
            stop_reasons.append(f"stop_token:{int(stop_tok_cpu[row])}")
        else:
            stop_reasons.append("max_new_tokens")

    captions = [
        normalize_generation_text(
            tokenizer.decode(
                token_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
        )
        for token_ids in generated_ids
    ]
    return {
        "captions": captions,
        "generated_ids": generated_ids,
        "stop_reasons": stop_reasons,
        "cache_mode": "kv_cache" if use_kv_cache else "no_cache",
    }


def build_candidate_text(object_row: Dict[str, Any]) -> str:
    parse_mode = str(object_row.get("parse_mode") or "").strip().lower()
    if parse_mode == "refexp_only":
        return str(object_row.get("refexp") or "")

    parts: List[str] = []
    for key in ("refexp", "name", "attributes", "location", "action"):
        value = normalize_caption(object_row.get(key, ""))
        if value and value.lower() not in {"none", "n/a", "na", "unknown"}:
            if value not in parts:
                parts.append(value)
    return ", ".join(parts)


def clean_object_text(text: str) -> str:
    text = str(text or "")
    text = text.replace("[", " ").replace("]", " ")
    text = re.sub(r"\bobj\s*\d+\b", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text).strip(" \t\r\n,;:.")
    return normalize_caption(text)


def strip_repetitive_tail(text: str) -> str:
    words = [tok for tok in str(text or "").split() if tok]
    if len(words) < 8:
        return text

    norm_words = [re.sub(r"[^a-z0-9]", "", tok.lower()) for tok in words]

    # Cut before a long exact repetition run like "sa sa sa sa sa".
    run_token = None
    run_start = None
    run_len = 0
    for idx, tok in enumerate(norm_words):
        if tok and tok == run_token:
            run_len += 1
        else:
            run_token = tok
            run_start = idx
            run_len = 1
        if tok and run_len >= 5:
            trimmed = " ".join(words[:run_start]).strip(" \t\r\n,;:.")
            return trimmed or text

    # Also trim a very repetitive suffix dominated by one short token.
    for tail_len in (24, 16, 12, 8):
        if len(norm_words) < tail_len:
            continue
        tail = norm_words[-tail_len:]
        counts: Dict[str, int] = {}
        for tok in tail:
            if not tok:
                continue
            counts[tok] = counts.get(tok, 0) + 1
        if not counts:
            continue
        dominant, dominant_count = max(counts.items(), key=lambda kv: kv[1])
        if dominant_count >= max(5, int(0.6 * tail_len)) and len(dominant) <= 3:
            first_dom_idx = len(norm_words) - tail_len + tail.index(dominant)
            trimmed = " ".join(words[:first_dom_idx]).strip(" \t\r\n,;:.")
            return trimmed or text

    return text


def is_truncated_line(raw_line: str, text: str) -> bool:
    stripped = str(raw_line or "").strip()
    text_norm = normalize_caption(text)
    if not text_norm or len(text_norm) < 3:
        return True
    if stripped in {"[", "[OBJ", "[Obj", "[BACKGROUND"}:
        return True
    if stripped.count("[") > stripped.count("]"):
        return True
    if stripped.endswith((",", ":", ";", "|")):
        return True
    if text_norm.endswith(("[", "|", ",", ":", ";")):
        return True
    return False


def is_generic_junk(text: str) -> bool:
    lowered = normalize_caption(text).lower()
    if lowered in GENERIC_JUNK_TEXTS:
        return True
    if re.fullmatch(r"obj\d+", lowered):
        return True
    if re.fullmatch(r"[a-z]$", lowered):
        return True
    return False


def normalize_for_duplicate_match(text: str) -> str:
    lowered = normalize_caption(text).lower()
    lowered = re.sub(r"[^a-z0-9\s]", " ", lowered)
    lowered = re.sub(r"\s+", " ", lowered).strip()
    return lowered


def is_conservative_prefix_duplicate(text: str, kept_texts: Sequence[str]) -> bool:
    candidate = normalize_for_duplicate_match(text)
    if not candidate:
        return False
    candidate_tokens = candidate.split()
    if len(candidate_tokens) < 3:
        return False

    for prev in kept_texts:
        prev_norm = normalize_for_duplicate_match(prev)
        if not prev_norm or prev_norm == candidate:
            continue

        shorter, longer = (candidate, prev_norm) if len(candidate) <= len(prev_norm) else (prev_norm, candidate)
        shorter_tokens = shorter.split()
        longer_tokens = longer.split()
        if len(shorter_tokens) < 3 or len(shorter_tokens) >= len(longer_tokens):
            continue

        if shorter == " ".join(longer_tokens[: len(shorter_tokens)]):
            return True
    return False


def extract_clean_object_texts(parsed: Dict[str, Any], max_slots: int) -> Tuple[List[str], Dict[str, int]]:
    stats = {
        "raw_objects": int(parsed.get("num_objects", 0)),
        "removed_background": 1 if parsed.get("has_background") else 0,
        "removed_truncated": 0,
        "removed_generic": 0,
        "removed_duplicates": 0,
        "removed_empty": 0,
        "removed_over_max_slots": 0,
    }
    cleaned: List[str] = []
    seen = set()
    for obj in parsed.get("objects", []):
        raw_text = build_candidate_text(obj)
        text = clean_object_text(raw_text)
        text = clean_object_text(strip_repetitive_tail(text))
        raw_line = str(obj.get("raw_line") or "")
        if not text:
            stats["removed_empty"] += 1
            continue
        if is_truncated_line(raw_line, text):
            stats["removed_truncated"] += 1
            continue
        if is_generic_junk(text):
            stats["removed_generic"] += 1
            continue
        lowered = text.lower()
        if lowered in seen:
            stats["removed_duplicates"] += 1
            continue
        if is_conservative_prefix_duplicate(text, cleaned):
            stats["removed_duplicates"] += 1
            continue
        seen.add(lowered)
        cleaned.append(text)

    if len(cleaned) > max_slots:
        stats["removed_over_max_slots"] = len(cleaned) - max_slots
        cleaned = cleaned[:max_slots]
    stats["clean_objects"] = len(cleaned)
    return cleaned, stats


def build_caption_from_object_texts(
    tokenizer,
    object_texts: Sequence[str],
    max_caption_tokens: int,
    max_slots: int,
) -> Dict[str, Any]:
    sep_ids = tokenizer(". ", add_special_tokens=False).input_ids
    caption_ids: List[int] = []
    spans: List[Dict[str, Any]] = []
    kept_texts: List[str] = []
    dropped_for_budget = 0

    for object_text in list(object_texts)[:max_slots]:
        object_ids = tokenizer(object_text, add_special_tokens=False).input_ids
        if not object_ids:
            continue

        prefix_ids = sep_ids if caption_ids else []
        remaining = max_caption_tokens - (len(caption_ids) + len(prefix_ids))
        if remaining <= 0:
            dropped_for_budget += 1
            continue
        if len(object_ids) > remaining:
            object_ids = object_ids[:remaining]
            if not object_ids:
                dropped_for_budget += 1
                continue

        if prefix_ids:
            caption_ids.extend(prefix_ids)
        token_start = len(caption_ids)
        caption_ids.extend(object_ids)
        token_end = len(caption_ids)
        text_kept = normalize_caption(
            tokenizer.decode(
                object_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
        )
        head_text = text_kept.split()[-1] if text_kept and text_kept != "<empty>" else text_kept
        spans.append(
            {
                "text": text_kept,
                "token_start": int(token_start),
                "token_end": int(token_end),
                "head_text": head_text,
                "head_token_start": int(token_start),
                "head_token_end": int(token_end),
            }
        )
        kept_texts.append(text_kept)

    caption_text = normalize_caption(
        tokenizer.decode(
            caption_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
    )
    return {
        "caption": caption_text,
        "token_ids": caption_ids,
        "noun_chunks": spans,
        "object_texts": kept_texts,
        "dropped_for_token_budget": dropped_for_budget,
    }


def safe_square_side(num_tokens: int) -> Tuple[int, bool]:
    side = int(math.sqrt(num_tokens))
    return side, side * side == num_tokens


def _install_timm_attention_extract_compat() -> None:
    import timm.utils

    if hasattr(timm.utils, "AttentionExtract"):
        return

    class AttentionExtractCompat:
        def __init__(self, model=None, mode: str = "eval", method: str = "hook", **kwargs):
            self.model = model
            self.mode = mode
            self.method = method
            self.hooks = None

    timm.utils.AttentionExtract = AttentionExtractCompat


class SteerViTExtractor:
    def __init__(
        self,
        steervit_src: str,
        checkpoint_name: str,
        device: torch.device,
        dtype: torch.dtype,
        map_type: str,
        head_pooling: str,
        gate_factor: float,
        remove_sinks: bool = True,
    ):
        _install_timm_attention_extract_compat()
        sys.path.insert(0, steervit_src)
        from steervit import SteerViT
        from steervit.utils import suppress_attention_sinks_multi

        self.device = device
        self.dtype = dtype
        self.map_type = map_type
        self.head_pooling = head_pooling
        self.remove_sinks = remove_sinks
        self._suppress_attention_sinks_multi = suppress_attention_sinks_multi
        self.model = SteerViT.from_pretrained(checkpoint_name, device=str(device))
        self.model = self.model.to(device=device)
        if dtype != torch.float32:
            self.model.text_model = self.model.text_model.to(dtype=dtype)
            self.model.vision_model = self.model.vision_model.to(dtype=dtype)
            self.model.connector = self.model.connector.to(dtype=dtype)
            self.model.lin_seg_head = self.model.lin_seg_head.to(dtype=dtype)
        self.model.eval()
        self.model.set_gate_factor(gate_factor)
        self.transform = self.model.get_transforms()
        self.model_image_size = tuple(self.model.image_size)

    def get_runtime_info(self) -> Dict[str, str]:
        info = {
            "steervit_device": str(self.device),
            "steervit_requested_dtype": str(self.dtype),
            "steervit_map_type": str(self.map_type),
        }
        try:
            info["steervit_text_model_device"] = str(next(self.model.text_model.parameters()).device)
            info["steervit_text_model_dtype"] = str(next(self.model.text_model.parameters()).dtype)
        except Exception:
            info["steervit_text_model_device"] = "unknown"
            info["steervit_text_model_dtype"] = "unknown"
        try:
            info["steervit_vision_model_device"] = str(next(self.model.vision_model.parameters()).device)
            info["steervit_vision_model_dtype"] = str(next(self.model.vision_model.parameters()).dtype)
        except Exception:
            info["steervit_vision_model_device"] = "unknown"
            info["steervit_vision_model_dtype"] = "unknown"
        try:
            info["steervit_connector_device"] = str(next(self.model.connector.parameters()).device)
            info["steervit_connector_dtype"] = str(next(self.model.connector.parameters()).dtype)
        except Exception:
            info["steervit_connector_device"] = "unknown"
            info["steervit_connector_dtype"] = "unknown"
        try:
            info["steervit_seg_head_device"] = str(next(self.model.lin_seg_head.parameters()).device)
            info["steervit_seg_head_dtype"] = str(next(self.model.lin_seg_head.parameters()).dtype)
        except Exception:
            info["steervit_seg_head_device"] = "unknown"
            info["steervit_seg_head_dtype"] = "unknown"
        return info

    def _pool_heads(self, cls_attention: torch.Tensor) -> torch.Tensor:
        if self.head_pooling == "mean":
            return cls_attention.mean(dim=1)
        if self.head_pooling == "max":
            return cls_attention.max(dim=1).values
        if self.head_pooling == "min":
            return cls_attention.min(dim=1).values
        if self.head_pooling == "median":
            return cls_attention.median(dim=1).values
        raise ValueError(f"Unsupported head pooling: {self.head_pooling}")

    @torch.inference_mode()
    def forward_batch(self, pixel_values: torch.Tensor, texts: Sequence[str]) -> torch.Tensor:
        pixel_values = pixel_values.to(self.device, non_blocking=True)
        if self.dtype != torch.float32:
            pixel_values = pixel_values.to(self.dtype)

        if self.map_type == "heatmap":
            heatmaps = self.model.get_heatmaps(pixel_values, texts=list(texts)).squeeze(1)
            return heatmaps.float()

        _ = self.model(pixel_values, list(texts))
        last_block = self.model.vision_model.trunk.blocks[-1]
        attention_map = getattr(last_block.attn, "attn_map", None)
        if attention_map is None:
            raise RuntimeError("SteerViT last-layer attention map is unavailable.")

        num_prefix_tokens = int(self.model.vision_model.trunk.num_prefix_tokens)
        cls_attention = attention_map[:, :, 0, num_prefix_tokens:].float()
        num_tokens = int(cls_attention.shape[-1])
        side, is_square = safe_square_side(num_tokens)
        if not is_square:
            raise RuntimeError(f"Unexpected SteerViT patch token count: {num_tokens}")
        heatmaps = self._pool_heads(cls_attention).view(cls_attention.shape[0], side, side)

        if self.remove_sinks:
            heatmaps, _ = self._suppress_attention_sinks_multi(heatmaps)

        bmin = heatmaps.amin(dim=(-1, -2), keepdim=True)
        bmax = heatmaps.amax(dim=(-1, -2), keepdim=True)
        heatmaps = (heatmaps - bmin) / (bmax - bmin + 1e-6)
        heatmaps = F.interpolate(
            heatmaps.unsqueeze(1),
            size=self.model_image_size,
            mode="bilinear",
            align_corners=False,
        ).squeeze(1)
        return heatmaps.float()


def heatmaps_to_patch_vectors(heatmaps: torch.Tensor, grid_side: int) -> torch.Tensor:
    if heatmaps.ndim != 3:
        raise ValueError(f"Expected heatmaps with shape [B, H, W], got {tuple(heatmaps.shape)}")
    downsized = F.interpolate(
        heatmaps.unsqueeze(1).float(),
        size=(grid_side, grid_side),
        mode="bilinear",
        align_corners=False,
    ).squeeze(1)
    bmin = downsized.amin(dim=(-1, -2), keepdim=True)
    bmax = downsized.amax(dim=(-1, -2), keepdim=True)
    downsized = torch.where(
        (bmax - bmin) > 1e-8,
        (downsized - bmin) / (bmax - bmin + 1e-6),
        torch.zeros_like(downsized),
    )
    return downsized.flatten(start_dim=1)


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


class SlotMapShardWriter:
    def __init__(self, output_dir: str, shard_size: int, start_shard_idx: int = 0) -> None:
        self.output_dir = output_dir
        self.shard_size = int(shard_size)
        self.shard_idx = int(start_shard_idx)
        self.pending_records: List[Dict[str, Any]] = []
        self.pending_image_ids: List[str] = []
        self.pending_phrase_maps: List[torch.Tensor] = []
        self.pending_head_maps: List[torch.Tensor] = []
        self.pending_phrase_valid_mask: List[torch.Tensor] = []
        self.pending_head_valid_mask: List[torch.Tensor] = []
        self.shard_paths: List[str] = []
        ensure_output_dir(self.output_dir)

    def append(
        self,
        record: Dict[str, Any],
        image_id: str,
        phrase_maps: torch.Tensor,
        head_maps: torch.Tensor,
        phrase_valid_mask: torch.Tensor,
        head_valid_mask: torch.Tensor,
    ) -> List[Dict[str, Any]]:
        self.pending_records.append(record)
        self.pending_image_ids.append(str(image_id))
        self.pending_phrase_maps.append(phrase_maps.detach().cpu().to(torch.float16))
        self.pending_head_maps.append(head_maps.detach().cpu().to(torch.float16))
        self.pending_phrase_valid_mask.append(phrase_valid_mask.detach().cpu().to(torch.bool))
        self.pending_head_valid_mask.append(head_valid_mask.detach().cpu().to(torch.bool))
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
            "phrase_valid_mask": torch.stack(self.pending_phrase_valid_mask, dim=0),
            "head_valid_mask": torch.stack(self.pending_head_valid_mask, dim=0),
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
        self.pending_phrase_valid_mask = []
        self.pending_head_valid_mask = []
        self.shard_idx += 1
        return finalized


def build_annotation_entry(record: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "caption": record.get("caption", ""),
        "token_ids": record.get("token_ids", []),
        "noun_chunks": record.get("noun_chunks", []),
        "selection_strategy": record.get("selection_strategy", "object_refexp"),
        "map_shard": record.get("map_shard"),
        "map_index": record.get("map_index"),
        "n_cached_chunks": int(record.get("n_cached_chunks", 0)),
        "phrase_valid_mask": record.get("phrase_valid_mask", []),
        "head_valid_mask": record.get("head_valid_mask", []),
        "object_texts": record.get("object_texts", []),
        "clean_num_objects": int(record.get("clean_num_objects", 0)),
        "raw_num_objects": int(record.get("raw_num_objects", 0)),
        "dropped_for_token_budget": int(record.get("dropped_for_token_budget", 0)),
        "object_cleaning": record.get("object_cleaning", {}),
    }


def finalize_records(
    finalized_records: Sequence[Dict[str, Any]],
    metadata_file,
    annotations: Dict[str, Dict[str, Any]],
) -> None:
    for record_out in finalized_records:
        metadata_file.write(json.dumps(record_out, ensure_ascii=False) + "\n")
        annotations[str(record_out["image_id"])] = build_annotation_entry(record_out)


def load_resume_state(
    metadata_path: str,
    annotations_path: str,
    shards_dir: str,
) -> Dict[str, Any]:
    completed_ids: set = set()
    existing_annotations: Dict[str, Dict[str, Any]] = {}
    historical_records: List[Dict[str, Any]] = []

    if os.path.isfile(metadata_path):
        with open(metadata_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                img_id = str(rec.get("image_id", ""))
                if not img_id:
                    continue
                completed_ids.add(img_id)
                historical_records.append(rec)

    if os.path.isfile(annotations_path):
        try:
            with open(annotations_path, "r", encoding="utf-8") as f:
                existing_annotations = json.load(f)
        except (json.JSONDecodeError, OSError):
            existing_annotations = {}

    if not existing_annotations and historical_records:
        rebuilt_annotations: Dict[str, Dict[str, Any]] = {}
        for rec in historical_records:
            image_id = str(rec.get("image_id", ""))
            if not image_id:
                continue
            rebuilt_annotations[image_id] = build_annotation_entry(rec)
        existing_annotations = rebuilt_annotations

    existing_shard_paths: List[str] = []
    next_shard_idx = 0
    if os.path.isdir(shards_dir):
        shard_files = sorted(
            p for p in os.listdir(shards_dir)
            if p.startswith("cache_") and p.endswith(".pt")
        )
        existing_shard_paths = [os.path.abspath(os.path.join(shards_dir, p)) for p in shard_files]
        next_shard_idx = len(shard_files)

    return {
        "completed_ids": completed_ids,
        "existing_annotations": existing_annotations,
        "historical_records": historical_records,
        "existing_shard_paths": existing_shard_paths,
        "next_shard_idx": next_shard_idx,
    }


def aggregate_historical_counts(records: Sequence[Dict[str, Any]]) -> Dict[str, int]:
    counts = {
        "total_errors": 0,
        "images_with_any_object": 0,
        "images_with_any_head_prior": 0,
        "total_cached_objects": 0,
        "total_valid_head_maps": 0,
        "total_removed_duplicates": 0,
        "total_removed_truncated": 0,
        "total_removed_generic": 0,
    }
    for rec in records:
        if rec.get("error"):
            counts["total_errors"] += 1
        n_cached = int(rec.get("n_cached_chunks", 0))
        if n_cached > 0:
            counts["images_with_any_object"] += 1
            counts["total_cached_objects"] += n_cached
        head_mask = rec.get("head_valid_mask") or []
        if any(bool(x) for x in head_mask):
            counts["images_with_any_head_prior"] += 1
        counts["total_valid_head_maps"] += sum(1 for x in head_mask if x)
        cleaning = rec.get("object_cleaning") or {}
        counts["total_removed_duplicates"] += int(cleaning.get("removed_duplicates", 0))
        counts["total_removed_truncated"] += int(cleaning.get("removed_truncated", 0))
        counts["total_removed_generic"] += int(cleaning.get("removed_generic", 0))
    return counts


def main() -> None:
    args = parse_args()
    if args.grid_side != 16:
        raise ValueError(
            f"grid_side={args.grid_side} is not compatible with the current AURORA CaptionSlot trainer. "
            "The current trainer expects 16x16 = 256 patch priors."
        )

    use_kv_cache = not args.disable_kv_cache
    ensure_output_dir(args.output_dir)
    ensure_output_dir(os.path.join(args.output_dir, "shards"))
    if args.save_debug_images:
        ensure_output_dir(os.path.join(args.output_dir, "samples"))
    set_seed(args.seed)
    set_fast_math(args.tf32)

    image_paths, image_source = resolve_requested_image_paths(args)
    if args.max_samples is not None:
        image_paths = image_paths[: args.max_samples]
    global_requested_samples = len(image_paths)
    if global_requested_samples == 0:
        raise SystemExit(f"No images found for source: {image_source}")
    image_paths_all_for_shard = take_shard(image_paths, args.num_shards, args.shard_index)

    metadata_path = os.path.join(args.output_dir, "metadata.jsonl")
    annotations_path = os.path.join(args.output_dir, "captionslot_annotations.json")
    summary_path = os.path.join(args.output_dir, "summary.json")
    shards_dir = os.path.join(args.output_dir, "shards")

    resume_state = load_resume_state(metadata_path, annotations_path, shards_dir)
    hist_counts_merged = aggregate_historical_counts(resume_state["historical_records"])
    completed_ids = resume_state["completed_ids"]
    image_paths = [p for p in image_paths_all_for_shard if Path(p).stem not in completed_ids]
    resumed_sample_count = len(image_paths_all_for_shard) - len(image_paths)

    if not image_paths_all_for_shard:
        if not os.path.isfile(metadata_path):
            Path(metadata_path).write_text("", encoding="utf-8")
        if not os.path.isfile(annotations_path):
            save_json(annotations_path, {})
        empty_summary = {
            "image_source": os.path.abspath(image_source),
            "output_dir": os.path.abspath(args.output_dir),
            "requested_samples": 0,
            "global_requested_samples": int(global_requested_samples),
            "annotation_entry_count": 0,
            "images_with_any_object": 0,
            "images_with_any_head_prior": 0,
            "total_cached_objects": 0,
            "total_valid_head_maps": 0,
            "error_count": 0,
            "metadata_jsonl": os.path.abspath(metadata_path),
            "annotations_json": os.path.abspath(annotations_path),
            "map_shards": [],
            "notes": ["This shard had no assigned samples after sharding."],
        }
        save_json(summary_path, empty_summary)
        print(json.dumps(empty_summary, indent=2, ensure_ascii=False))
        return

    if not image_paths:
        save_json(annotations_path, resume_state["existing_annotations"])
        resume_summary = {
            "image_source": os.path.abspath(image_source),
            "output_dir": os.path.abspath(args.output_dir),
            "requested_samples": int(len(image_paths_all_for_shard)),
            "global_requested_samples": int(global_requested_samples),
            "resumed_sample_count": int(resumed_sample_count),
            "annotation_entry_count": int(len(resume_state["existing_annotations"])),
            "images_with_any_object": int(hist_counts_merged["images_with_any_object"]),
            "images_with_any_head_prior": int(hist_counts_merged["images_with_any_head_prior"]),
            "total_cached_objects": int(hist_counts_merged["total_cached_objects"]),
            "total_valid_head_maps": int(hist_counts_merged["total_valid_head_maps"]),
            "error_count": int(hist_counts_merged["total_errors"]),
            "metadata_jsonl": os.path.abspath(metadata_path),
            "annotations_json": os.path.abspath(annotations_path),
            "map_shards": list(resume_state["existing_shard_paths"]),
            "notes": ["Resume complete: no new samples to process in this shard."],
        }
        save_json(summary_path, resume_summary)
        print(json.dumps(resume_summary, indent=2, ensure_ascii=False))
        return

    dtype = resolve_dtype(args.dtype, args.device)
    tokenizer, model, image_processor, _context_len = load_scale_rae_model(
        model_path=args.model_path,
        device=args.device,
        dtype=dtype,
    )
    model_device = torch.device(args.device if torch.cuda.is_available() or not str(args.device).startswith("cuda") else "cpu")
    move_vision_towers_to_device(model, model_device, dtype)
    runtime_dtypes = get_runtime_dtypes(model)
    runtime_devices = get_runtime_devices(model)
    runtime_backbone_dtype = next(model.get_model().parameters()).dtype

    start_id, end_id, eos_id = prepare_special_token_ids(tokenizer)
    prompt_built = build_prompt(args.prompt, model_config=model.config, with_image=True, num_frames=1)
    prompt_input_ids = tokenize_prompt(prompt_built, tokenizer, device=model.device)
    extractor = SteerViTExtractor(
        steervit_src=args.steervit_src,
        checkpoint_name=args.steervit_checkpoint,
        device=torch.device(args.device),
        dtype=dtype,
        map_type=args.steervit_map_type,
        head_pooling=args.steervit_head_pooling,
        gate_factor=args.steervit_gate_factor,
    )
    steervit_runtime = extractor.get_runtime_info()
    runtime_report = {
        "scale_rae": {
            "device": runtime_devices,
            "dtype": runtime_dtypes,
        },
        "steervit": steervit_runtime,
    }
    print("Runtime placement:")
    print(json.dumps(runtime_report, indent=2, ensure_ascii=False))

    total_errors = 0
    images_with_any_object = 0
    images_with_any_head_prior = 0
    total_cached_objects = 0
    total_valid_head_maps = 0
    total_removed_duplicates = 0
    total_removed_truncated = 0
    total_removed_generic = 0
    caption_batch_fallback_count = 0
    debug_saved = 0

    annotations: Dict[str, Dict[str, Any]] = dict(resume_state["existing_annotations"])
    shard_writer = SlotMapShardWriter(
        os.path.join(args.output_dir, "shards"),
        shard_size=args.map_shard_size,
        start_shard_idx=int(resume_state["next_shard_idx"]),
    )
    shard_writer.shard_paths.extend(resume_state["existing_shard_paths"])

    prefetch_dataset = _PrefetchImageDataset(
        image_paths=image_paths,
        image_processor=image_processor,
        steervit_transform=extractor.transform,
    )
    loader_num_workers = max(0, int(args.loader_num_workers))
    prefetch_loader = torch.utils.data.DataLoader(
        prefetch_dataset,
        batch_size=int(args.caption_batch_size),
        shuffle=False,
        num_workers=loader_num_workers,
        collate_fn=_identity_collate,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=loader_num_workers > 0,
        prefetch_factor=4 if loader_num_workers > 0 else None,
    )

    fsync_every = max(1, int(args.fsync_every_n_batches))
    empty_head_maps = torch.zeros(
        (args.max_slots, args.grid_side * args.grid_side), dtype=torch.float32
    )
    empty_phrase_maps = torch.zeros(
        (args.max_slots, args.grid_side * args.grid_side), dtype=torch.float32
    )
    empty_valid_mask = torch.zeros(args.max_slots, dtype=torch.bool)

    def _register_error_record(record: Dict[str, Any]) -> List[Dict[str, Any]]:
        record["caption"] = ""
        record["token_ids"] = []
        record["noun_chunks"] = []
        record["object_texts"] = []
        record["n_cached_chunks"] = 0
        record["clean_num_objects"] = 0
        record["raw_num_objects"] = 0
        record["object_cleaning"] = {}
        record["phrase_valid_mask"] = [False] * int(args.max_slots)
        record["head_valid_mask"] = [False] * int(args.max_slots)
        return shard_writer.append(
            record=record,
            image_id=record["image_id"],
            phrase_maps=empty_phrase_maps.clone(),
            head_maps=empty_head_maps.clone(),
            phrase_valid_mask=empty_valid_mask.clone(),
            head_valid_mask=empty_valid_mask.clone(),
        )

    metadata_open_mode = "a" if os.path.isfile(metadata_path) else "w"
    with open(metadata_path, metadata_open_mode, encoding="utf-8") as metadata_file, tqdm(
        total=len(image_paths),
        desc="StageA Object SteerViT Cache",
        unit="img",
        dynamic_ncols=True,
    ) as pbar:
        for batch_idx, batch in enumerate(prefetch_loader, start=1):
            batch_items: List[Dict[str, Any]] = []
            for entry in batch:
                record: Dict[str, Any] = {
                    "image": os.path.abspath(entry["image_path"]),
                    "image_id": entry["image_id"],
                    "caption_prompt": args.prompt,
                    "caption_prompt_built": prompt_built,
                    "num_shards": int(args.num_shards),
                    "shard_index": int(args.shard_index),
                    "prior_source": "steervit_attention" if args.steervit_map_type == "attention" else "steervit_heatmap",
                    "grid_side": int(args.grid_side),
                    "max_slots": int(args.max_slots),
                    "selection_strategy": "object_refexp",
                }
                if not entry.get("ok", False):
                    total_errors += 1
                    record["error"] = entry.get("error", "load_failed")
                    finalized = _register_error_record(record)
                    finalize_records(finalized, metadata_file, annotations)
                    pbar.update(1)
                    continue
                batch_items.append(
                    {
                        "image_path": entry["image_path"],
                        "image_id": entry["image_id"],
                        "record": record,
                        "caption_pixel": entry["caption_pixel"],
                        "steervit_pixel": entry["steervit_pixel"],
                    }
                )

            if not batch_items:
                continue

            images_tensor = torch.stack(
                [item["caption_pixel"] for item in batch_items], dim=0
            ).to(device=model.device, dtype=runtime_backbone_dtype, non_blocking=True)
            batch_input_ids = prompt_input_ids.expand(len(batch_items), -1).contiguous()

            try:
                caption_batch = generate_captions_batched_kv(
                    model=model,
                    tokenizer=tokenizer,
                    prompt_input_ids=batch_input_ids,
                    images_tensor=images_tensor,
                    eos_token_id=eos_id,
                    max_new_tokens=args.caption_max_new_tokens,
                    start_image_token_id=start_id,
                    end_image_token_id=end_id,
                    use_kv_cache=use_kv_cache,
                )
                cache_mode = caption_batch["cache_mode"]
                for row_idx, item in enumerate(batch_items):
                    item["caption_info"] = {
                        "caption": caption_batch["captions"][row_idx],
                        "generated_ids": caption_batch["generated_ids"][row_idx],
                        "stop_reason": caption_batch["stop_reasons"][row_idx],
                        "cache_mode": cache_mode,
                    }
            except Exception as batch_exc:
                caption_batch_fallback_count += 1
                print(f"[caption-fallback] batch {batch_idx} -> single: {type(batch_exc).__name__}: {batch_exc}")
                recovered_items: List[Dict[str, Any]] = []
                for item in batch_items:
                    try:
                        single_pixels = item["caption_pixel"].unsqueeze(0).to(
                            device=model.device, dtype=runtime_backbone_dtype, non_blocking=True
                        )
                        single_result = generate_captions_batched_kv(
                            model=model,
                            tokenizer=tokenizer,
                            prompt_input_ids=prompt_input_ids,
                            images_tensor=single_pixels,
                            eos_token_id=eos_id,
                            max_new_tokens=args.caption_max_new_tokens,
                            start_image_token_id=start_id,
                            end_image_token_id=end_id,
                            use_kv_cache=use_kv_cache,
                        )
                        item["caption_info"] = {
                            "caption": single_result["captions"][0],
                            "generated_ids": single_result["generated_ids"][0],
                            "stop_reason": single_result["stop_reasons"][0],
                            "cache_mode": "fallback_single",
                        }
                        recovered_items.append(item)
                    except Exception as exc:
                        total_errors += 1
                        rec = item["record"]
                        rec["error"] = f"{type(exc).__name__}: {exc}"
                        finalized = _register_error_record(rec)
                        finalize_records(finalized, metadata_file, annotations)
                        pbar.update(1)
                batch_items = recovered_items

            if not batch_items:
                continue

            # Parse captions + build per-image object_texts.
            for item in batch_items:
                record = item["record"]
                caption_info = item["caption_info"]
                parsed = parse_objectwise_output(caption_info["caption"])
                clean_object_texts, cleaning_stats = extract_clean_object_texts(parsed, max_slots=args.max_slots)
                caption_serialized = build_caption_from_object_texts(
                    tokenizer=tokenizer,
                    object_texts=clean_object_texts,
                    max_caption_tokens=args.max_caption_tokens,
                    max_slots=args.max_slots,
                )
                noun_chunks = caption_serialized["noun_chunks"]
                object_texts = caption_serialized["object_texts"]

                record["raw_generation"] = caption_info["caption"]
                record["caption"] = caption_serialized["caption"]
                record["token_ids"] = caption_serialized["token_ids"]
                record["noun_chunks"] = noun_chunks
                record["object_texts"] = object_texts
                item["object_texts"] = object_texts
                record["n_cached_chunks"] = int(len(noun_chunks))
                record["clean_num_objects"] = int(len(object_texts))
                record["raw_num_objects"] = int(parsed.get("num_objects", 0))
                record["dropped_for_token_budget"] = int(caption_serialized["dropped_for_token_budget"])
                record["object_cleaning"] = cleaning_stats
                record["stop_reason"] = caption_info["stop_reason"]
                record["cache_mode"] = caption_info["cache_mode"] + "+steervit"
                if object_texts:
                    images_with_any_object += 1
                    total_cached_objects += len(object_texts)
                total_removed_duplicates += int(cleaning_stats.get("removed_duplicates", 0))
                total_removed_truncated += int(cleaning_stats.get("removed_truncated", 0))
                total_removed_generic += int(cleaning_stats.get("removed_generic", 0))

                item["head_maps"] = torch.zeros(
                    (args.max_slots, args.grid_side * args.grid_side), dtype=torch.float32
                )
                item["phrase_maps"] = torch.zeros(
                    (args.max_slots, args.grid_side * args.grid_side), dtype=torch.float32
                )
                item["head_valid_mask"] = torch.zeros(args.max_slots, dtype=torch.bool)
                item["phrase_valid_mask"] = torch.zeros(args.max_slots, dtype=torch.bool)

            # Build flat (image_row, slot, text) tasks and batch SteerViT across them.
            steervit_pixels_gpu = torch.stack(
                [item["steervit_pixel"] for item in batch_items], dim=0
            ).to(extractor.device, non_blocking=True)

            flat_texts: List[str] = []
            flat_row_idx: List[int] = []
            flat_slot_idx: List[int] = []
            for row_idx, item in enumerate(batch_items):
                for slot_idx, text in enumerate(item["object_texts"][: args.max_slots]):
                    flat_texts.append(text)
                    flat_row_idx.append(row_idx)
                    flat_slot_idx.append(slot_idx)

            if flat_texts:
                steervit_bs = max(1, int(args.steervit_batch_size))
                for chunk_start in range(0, len(flat_texts), steervit_bs):
                    chunk_end = min(chunk_start + steervit_bs, len(flat_texts))
                    chunk_texts = flat_texts[chunk_start:chunk_end]
                    chunk_rows = flat_row_idx[chunk_start:chunk_end]
                    chunk_slots = flat_slot_idx[chunk_start:chunk_end]
                    idx_tensor = torch.tensor(chunk_rows, dtype=torch.long, device=extractor.device)
                    pixel_chunk = steervit_pixels_gpu.index_select(0, idx_tensor)
                    try:
                        chunk_heatmaps = extractor.forward_batch(
                            pixel_values=pixel_chunk, texts=chunk_texts
                        )
                        chunk_vectors = heatmaps_to_patch_vectors(
                            chunk_heatmaps, grid_side=args.grid_side
                        ).detach().cpu()
                        # Pre-compute non-empty vectors in one shot.
                        nonempty = (chunk_vectors.sum(dim=-1) > 0.0).tolist()
                        for t_idx, (row_idx, slot_idx) in enumerate(zip(chunk_rows, chunk_slots)):
                            if not nonempty[t_idx]:
                                continue
                            v = chunk_vectors[t_idx]
                            item = batch_items[row_idx]
                            item["head_maps"][slot_idx] = v
                            item["phrase_maps"][slot_idx] = v
                            item["head_valid_mask"][slot_idx] = True
                            item["phrase_valid_mask"][slot_idx] = True
                    except Exception as exc:
                        total_errors += 1
                        err = f"{type(exc).__name__}: {exc}"
                        for row_idx in set(chunk_rows):
                            batch_items[row_idx]["record"]["error"] = err

            # Finalize per-image.
            for item in batch_items:
                record = item["record"]
                head_valid_mask = item["head_valid_mask"]
                phrase_valid_mask = item["phrase_valid_mask"]
                record["phrase_valid_mask"] = [bool(x) for x in phrase_valid_mask.tolist()]
                record["head_valid_mask"] = [bool(x) for x in head_valid_mask.tolist()]
                any_head_prior = bool(head_valid_mask.any().item())
                record["any_head_prior"] = any_head_prior
                if any_head_prior:
                    images_with_any_head_prior += 1
                    total_valid_head_maps += int(head_valid_mask.sum().item())

                if args.save_debug_images and debug_saved < args.debug_limit and record["object_texts"]:
                    sample_dir = os.path.join(args.output_dir, "samples", record["image_id"])
                    ensure_output_dir(sample_dir)
                    input_path = os.path.join(sample_dir, "input_processed.png")
                    head_map_path = os.path.join(sample_dir, "slot0_head_map.png")
                    head_overlay_path = os.path.join(sample_dir, "slot0_head_overlay.png")
                    try:
                        pil_debug = Image.open(item["image_path"]).convert("RGB").resize((224, 224))
                        pil_debug.save(input_path)
                        make_attention_heatmap(item["head_maps"][0]).save(head_map_path)
                        source_tensor = torch.from_numpy(
                            np.array(pil_debug).astype(np.float32) / 255.0
                        ).permute(2, 0, 1)
                        tensor_to_pil(
                            make_attention_overlay(source_tensor, item["head_maps"][0], color=(0.15, 0.65, 1.0))
                        ).save(head_overlay_path)
                        record["debug_paths"] = {
                            "input_processed": os.path.abspath(input_path),
                            "slot0_head_map": os.path.abspath(head_map_path),
                            "slot0_head_overlay": os.path.abspath(head_overlay_path),
                        }
                        save_json(os.path.join(sample_dir, "record.json"), record)
                    except Exception:
                        pass
                    debug_saved += 1

                finalized = shard_writer.append(
                    record=record,
                    image_id=record["image_id"],
                    phrase_maps=item["phrase_maps"],
                    head_maps=item["head_maps"],
                    phrase_valid_mask=phrase_valid_mask,
                    head_valid_mask=head_valid_mask,
                )
                finalize_records(finalized, metadata_file, annotations)
                pbar.update(1)

            if batch_idx % fsync_every == 0:
                metadata_file.flush()
                os.fsync(metadata_file.fileno())
            pbar.set_postfix(
                batch=batch_idx,
                obj_imgs=images_with_any_object,
                head_valid=images_with_any_head_prior,
                errors=total_errors,
                caption_bs=args.caption_batch_size,
                steervit_bs=args.steervit_batch_size,
            )

        finalized = shard_writer.flush()
        finalize_records(finalized, metadata_file, annotations)
        metadata_file.flush()
        os.fsync(metadata_file.fileno())

    ordered_annotations = {
        image_id: annotations[image_id]
        for image_id in sorted(annotations.keys(), key=lambda key: int(key))
    }
    save_json(annotations_path, ordered_annotations)

    summary = {
        "image_source": os.path.abspath(image_source),
        "output_dir": os.path.abspath(args.output_dir),
        "model_path": args.model_path,
        "prompt": args.prompt,
        "prompt_built": prompt_built,
        "steervit_src": os.path.abspath(args.steervit_src),
        "steervit_checkpoint": args.steervit_checkpoint,
        "steervit_map_type": args.steervit_map_type,
        "grid_side": int(args.grid_side),
        "grid_tokens": int(args.grid_side * args.grid_side),
        "device": str(model.device),
        "model_device_property": runtime_devices["model_device_property"],
        "backbone_device": runtime_devices["backbone_device"],
        "lm_head_device": runtime_devices["lm_head_device"],
        "vision_tower_device": runtime_devices["vision_tower_device"],
        "mm_projector_device": runtime_devices["mm_projector_device"],
        "requested_dtype": str(dtype),
        "dtype": runtime_dtypes["model_dtype_property"],
        "backbone_dtype": runtime_dtypes["backbone_dtype"],
        "lm_head_dtype": runtime_dtypes["lm_head_dtype"],
        "vision_tower_dtype": runtime_dtypes["vision_tower_dtype"],
        "steervit_runtime": steervit_runtime,
        "caption_batch_size": int(args.caption_batch_size),
        "steervit_batch_size": int(args.steervit_batch_size),
        "caption_max_new_tokens": int(args.caption_max_new_tokens),
        "max_caption_tokens": int(args.max_caption_tokens),
        "max_slots": int(args.max_slots),
        "use_kv_cache": bool(use_kv_cache),
        "tf32": bool(args.tf32),
        "seed": int(args.seed),
        "num_shards": int(args.num_shards),
        "shard_index": int(args.shard_index),
        "requested_samples": int(len(image_paths_all_for_shard)),
        "newly_processed_samples": int(len(image_paths)),
        "resumed_sample_count": int(resumed_sample_count),
        "global_requested_samples": int(global_requested_samples),
        "annotation_entry_count": int(len(ordered_annotations)),
        "images_with_any_object": int(images_with_any_object + hist_counts_merged["images_with_any_object"]),
        "images_with_any_head_prior": int(images_with_any_head_prior + hist_counts_merged["images_with_any_head_prior"]),
        "total_cached_objects": int(total_cached_objects + hist_counts_merged["total_cached_objects"]),
        "total_valid_head_maps": int(total_valid_head_maps + hist_counts_merged["total_valid_head_maps"]),
        "total_removed_duplicates": int(total_removed_duplicates + hist_counts_merged["total_removed_duplicates"]),
        "total_removed_truncated": int(total_removed_truncated + hist_counts_merged["total_removed_truncated"]),
        "total_removed_generic": int(total_removed_generic + hist_counts_merged["total_removed_generic"]),
        "error_count": int(total_errors + hist_counts_merged["total_errors"]),
        "caption_batch_fallback_count": int(caption_batch_fallback_count),
        "debug_saved_count": int(debug_saved),
        "metadata_jsonl": os.path.abspath(metadata_path),
        "annotations_json": os.path.abspath(annotations_path),
        "map_shards": list(shard_writer.shard_paths),
        "notes": [
            "This cache is captionslot_annotations.json-compatible with the current CaptionSlotDataset.",
            "noun_chunks are object-text spans serialized from cleaned object-wise refexp captions.",
            "head_maps are SteerViT priors flattened onto the current 16x16 = 256 patch grid.",
            "phrase_maps are stored as aliases of head_maps for compatibility/debugging.",
            "If AURORA later moves to 32x32 patch priors, these maps must be regenerated or reprojected.",
        ],
    }
    save_json(summary_path, summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
