#!/usr/bin/env python
"""
Build a Stage A training cache for CaptionSlot on COCO train2017.

Pipeline:
image batch
  -> batched caption generation (KV-cache)
  -> batched spaCy noun-chunk parsing
  -> batched teacher-forced attention extraction
  -> aggregate per-chunk phrase/head maps
  -> write trainer-friendly annotations + sharded prior tensors

This builder is intentionally lighter than the diagnostic scripts:
- no COCO bbox evaluation
- no ranking analysis
- optional debug image saving only
"""

import argparse
import json
import os
import sys
import types
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import spacy
import torch
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

from build_stagea_generation_trace_diagnostic import (  # type: ignore
    DEFAULT_CAPTION_PROMPT,
    DEFAULT_MODEL_PATH,
    aggregate_step_maps,
    build_decoded_offsets,
    denormalize_image_tensor,
    find_token_span,
    list_image_files,
    make_attention_heatmap,
    make_attention_overlay,
    maybe_force_eager_attention,
    normalize_caption,
    resolve_dtype,
    save_json,
    set_seed,
    take_shard,
    tensor_to_pil,
)
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a fast Stage A CaptionSlot training cache.")
    parser.add_argument("--image-dir", required=True, help="Directory containing images.")
    parser.add_argument("--output-dir", required=True, help="Output directory.")
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--caption-prompt", default=DEFAULT_CAPTION_PROMPT)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=["fp16", "bf16", "fp32"], default="bf16")
    parser.add_argument("--caption-max-new-tokens", type=int, default=32)
    parser.add_argument("--max-caption-tokens", type=int, default=64)
    parser.add_argument("--trace-last-n-layers", type=int, default=4)
    parser.add_argument("--caption-batch-size", type=int, default=8)
    parser.add_argument("--trace-batch-size", type=int, default=4)
    parser.add_argument("--spacy-batch-size", type=int, default=128)
    parser.add_argument("--max-slots", type=int, default=10)
    parser.add_argument("--map-shard-size", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--spacy-model", default="en_core_web_sm")
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--save-debug-images", action="store_true")
    parser.add_argument("--debug-limit", type=int, default=25)
    parser.add_argument("--force-eager-attention", action="store_true")
    parser.add_argument("--disable-kv-cache", action="store_true")
    return parser.parse_args()


def batched(items: Sequence[Any], batch_size: int) -> Iterable[Sequence[Any]]:
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def extract_noun_info_from_token_ids_with_doc(
    token_ids: Sequence[int],
    tokenizer,
    doc,
    max_caption_tokens: int,
) -> Dict[str, Any]:
    token_ids = list(token_ids[:max_caption_tokens])
    caption_text, offsets = build_decoded_offsets(tokenizer, token_ids)

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

    if not noun_chunks and selected is not None:
        noun_chunks = [selected]

    return {
        "caption": normalize_caption(caption_text),
        "token_ids": token_ids,
        "noun_chunks": noun_chunks,
        "selected": selected,
        "selection_strategy": selection_strategy,
    }


def build_caption_batch(
    model,
    prompt_input_ids: torch.Tensor,
    images_tensor: torch.Tensor,
) -> Tuple[torch.Tensor, Any, int, int, int]:
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
    image_positions = (prompt_input_ids[0] == IMAGE_TOKEN_INDEX).nonzero(as_tuple=False).flatten()
    if image_positions.numel() != 1:
        raise RuntimeError(f"Expected exactly one image placeholder, got {int(image_positions.numel())}")
    image_start = int(image_positions[0].item())
    image_end = image_start + int(getattr(model, "num_image_tokens", 256))
    prefix_len = int(prefix_inputs_embeds.shape[1])
    return prefix_inputs_embeds, extra_mm, prefix_len, image_start, image_end


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
    prefix_inputs_embeds, extra_mm, prefix_len, image_start, image_end = build_caption_batch(
        model=model,
        prompt_input_ids=prompt_input_ids,
        images_tensor=images_tensor,
    )
    inputs_embeds = prefix_inputs_embeds.to(device=model.device)
    batch_size = int(inputs_embeds.shape[0])
    stop_ids = {int(eos_token_id), int(start_image_token_id), int(end_image_token_id)}

    generated_ids: List[List[int]] = [[] for _ in range(batch_size)]
    finished = torch.zeros(batch_size, dtype=torch.bool, device=model.device)
    stop_reasons = ["max_new_tokens"] * batch_size
    past_key_values = None
    current_inputs_embeds = inputs_embeds
    current_context_len = prefix_len

    with torch.inference_mode():
        for _step in range(max_new_tokens):
            attention_mask = torch.ones(
                (batch_size, current_context_len),
                device=model.device,
                dtype=torch.bool,
            )
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
            stop_mask = finished.clone()
            for token_id in stop_ids:
                stop_mask |= next_token.eq(token_id)

            for row_idx, token_id in enumerate(next_token.tolist()):
                if finished[row_idx]:
                    continue
                if token_id in stop_ids:
                    stop_reasons[row_idx] = f"stop_token:{token_id}"
                else:
                    generated_ids[row_idx].append(int(token_id))

            finished |= stop_mask
            if finished.all():
                break

            next_token = next_token.masked_fill(finished, int(eos_token_id))
            next_token_embed = model.get_model().embed_tokens(next_token.unsqueeze(1))
            next_token_embed = next_token_embed.to(device=inputs_embeds.device, dtype=inputs_embeds.dtype)

            if use_kv_cache:
                past_key_values = outputs.past_key_values
                current_inputs_embeds = next_token_embed
                current_context_len += 1
            else:
                inputs_embeds = torch.cat((inputs_embeds, next_token_embed), dim=1)
                current_inputs_embeds = inputs_embeds
                current_context_len = int(inputs_embeds.shape[1])

    captions = [
        normalize_caption(
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
        "token_texts": [tokenizer.convert_ids_to_tokens(token_ids) for token_ids in generated_ids],
        "prefix_inputs_embeds": inputs_embeds[:, :prefix_len, :].detach(),
        "prefix_len": prefix_len,
        "image_token_span": [image_start, image_end],
        "stop_reasons": stop_reasons,
        "cache_mode": "kv_cache" if use_kv_cache else "no_cache",
    }


def teacher_forced_extract_step_maps_batched(
    model,
    prefix_inputs_embeds: torch.Tensor,
    generated_ids_list: Sequence[Sequence[int]],
    image_start: int,
    image_end: int,
    last_n_layers: int,
    pad_token_id: int,
) -> List[List[torch.Tensor]]:
    batch_size = len(generated_ids_list)
    if batch_size == 0:
        return []
    prefix_len = int(prefix_inputs_embeds.shape[1])
    lengths = [len(ids) for ids in generated_ids_list]
    max_gen_len = max(lengths, default=0)
    if max_gen_len == 0:
        return [[] for _ in range(batch_size)]

    gen_ids_tensor = torch.full(
        (batch_size, max_gen_len),
        int(pad_token_id),
        device=model.device,
        dtype=torch.long,
    )
    attention_mask = torch.zeros(
        (batch_size, prefix_len + max_gen_len),
        device=model.device,
        dtype=torch.bool,
    )
    attention_mask[:, :prefix_len] = True

    for row_idx, token_ids in enumerate(generated_ids_list):
        if token_ids:
            ids_tensor = torch.tensor(list(token_ids), device=model.device, dtype=torch.long)
            gen_ids_tensor[row_idx, : len(token_ids)] = ids_tensor
            attention_mask[row_idx, prefix_len : prefix_len + len(token_ids)] = True

    with torch.inference_mode():
        gen_embeds = model.get_model().embed_tokens(gen_ids_tensor)
        gen_embeds = gen_embeds.to(device=prefix_inputs_embeds.device, dtype=prefix_inputs_embeds.dtype)
        full_embeds = torch.cat((prefix_inputs_embeds, gen_embeds), dim=1)
        full_embeds = full_embeds.to(device=model.device)

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

    all_step_maps: List[List[torch.Tensor]] = []
    for row_idx, token_ids in enumerate(generated_ids_list):
        row_step_maps: List[torch.Tensor] = []
        for token_idx in range(len(token_ids)):
            query_pos = prefix_len + token_idx - 1
            layer_maps: List[torch.Tensor] = []
            for layer_attn in use_layers:
                if layer_attn.ndim != 4:
                    continue
                if query_pos < 0 or query_pos >= layer_attn.shape[2]:
                    continue
                img_slice = layer_attn[row_idx, :, query_pos, image_start:image_end].float()
                if img_slice.numel() == 0:
                    continue
                layer_maps.append(img_slice.mean(dim=0))
            if not layer_maps:
                raise RuntimeError("Could not extract image attention slice from teacher-forced attentions.")
            attn_map = torch.stack(layer_maps, dim=0).mean(dim=0)
            attn_map = attn_map.clamp(min=0.0)
            attn_map = attn_map / attn_map.sum().clamp(min=1e-8)
            row_step_maps.append(attn_map.detach().cpu().float())
        all_step_maps.append(row_step_maps)

    return all_step_maps


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
        "selection_strategy": record.get("selection_strategy", "none"),
        "map_shard": record.get("map_shard"),
        "map_index": record.get("map_index"),
        "n_cached_chunks": int(record.get("n_cached_chunks", 0)),
        "phrase_valid_mask": record.get("phrase_valid_mask", []),
        "head_valid_mask": record.get("head_valid_mask", []),
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
        "images_with_any_noun_chunk": 0,
        "images_with_any_head_prior": 0,
        "images_with_any_phrase_prior": 0,
        "total_cached_noun_chunks": 0,
        "total_valid_head_maps": 0,
        "total_valid_phrase_maps": 0,
    }
    for rec in records:
        if rec.get("error"):
            counts["total_errors"] += 1
        noun_chunks = rec.get("noun_chunks") or []
        if noun_chunks:
            counts["images_with_any_noun_chunk"] += 1
            counts["total_cached_noun_chunks"] += len(noun_chunks)
        phrase_mask = rec.get("phrase_valid_mask") or []
        head_mask = rec.get("head_valid_mask") or []
        if any(bool(x) for x in phrase_mask):
            counts["images_with_any_phrase_prior"] += 1
        if any(bool(x) for x in head_mask):
            counts["images_with_any_head_prior"] += 1
        counts["total_valid_phrase_maps"] += sum(1 for x in phrase_mask if x)
        counts["total_valid_head_maps"] += sum(1 for x in head_mask if x)
    return counts


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
            info["vision_tower_dtype"] = str(vision_towers[0].dtype)
        else:
            info["vision_tower_dtype"] = "none"
    except Exception:
        info["vision_tower_dtype"] = "unknown"

    return info


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
    if resumed_sample_count > 0:
        print(
            f"Resume detected: {resumed_sample_count} already-completed images in this shard will be skipped. "
            f"Remaining: {len(image_paths)}"
        )

    if not image_paths_all_for_shard:
        if not os.path.isfile(metadata_path):
            Path(metadata_path).write_text("", encoding="utf-8")
        if not os.path.isfile(annotations_path):
            save_json(annotations_path, {})
        empty_summary = {
            "image_dir": os.path.abspath(args.image_dir),
            "output_dir": os.path.abspath(args.output_dir),
            "requested_samples": 0,
            "global_requested_samples": int(global_requested_samples),
            "annotation_entry_count": 0,
            "images_with_any_noun_chunk": 0,
            "images_with_any_head_prior": 0,
            "images_with_any_phrase_prior": 0,
            "total_cached_noun_chunks": 0,
            "total_valid_head_maps": 0,
            "total_valid_phrase_maps": 0,
            "error_count": 0,
            "caption_batch_fallback_count": 0,
            "trace_batch_fallback_count": 0,
            "metadata_jsonl": os.path.abspath(metadata_path),
            "annotations_json": os.path.abspath(annotations_path),
            "map_shards": [],
            "notes": ["This shard had no assigned samples after sharding."],
        }
        save_json(summary_path, empty_summary)
        print(json.dumps(empty_summary, indent=2, ensure_ascii=False))
        return

    if not image_paths:
        print("All assigned samples were already completed in a previous run. Writing summary and exiting.")
        hist_counts = aggregate_historical_counts(resume_state["historical_records"])
        save_json(annotations_path, resume_state["existing_annotations"])
        resume_summary = {
            "image_dir": os.path.abspath(args.image_dir),
            "output_dir": os.path.abspath(args.output_dir),
            "requested_samples": int(len(image_paths_all_for_shard)),
            "global_requested_samples": int(global_requested_samples),
            "resumed_sample_count": int(resumed_sample_count),
            "annotation_entry_count": int(len(resume_state["existing_annotations"])),
            "images_with_any_noun_chunk": int(hist_counts["images_with_any_noun_chunk"]),
            "images_with_any_head_prior": int(hist_counts["images_with_any_head_prior"]),
            "images_with_any_phrase_prior": int(hist_counts["images_with_any_phrase_prior"]),
            "total_cached_noun_chunks": int(hist_counts["total_cached_noun_chunks"]),
            "total_valid_head_maps": int(hist_counts["total_valid_head_maps"]),
            "total_valid_phrase_maps": int(hist_counts["total_valid_phrase_maps"]),
            "error_count": int(hist_counts["total_errors"]),
            "caption_batch_fallback_count": 0,
            "trace_batch_fallback_count": 0,
            "metadata_jsonl": os.path.abspath(metadata_path),
            "annotations_json": os.path.abspath(annotations_path),
            "map_shards": list(resume_state["existing_shard_paths"]),
            "notes": ["Resume complete: no new samples to process in this shard."],
        }
        save_json(summary_path, resume_summary)
        print(json.dumps(resume_summary, indent=2, ensure_ascii=False))
        return

    print(f"Loading spaCy model: {args.spacy_model}")
    nlp = spacy.load(
        args.spacy_model,
        disable=["ner", "textcat"],
    )

    dtype = resolve_dtype(args.dtype, args.device)
    tokenizer, model, image_processor, _context_len = load_scale_rae_model(
        model_path=args.model_path,
        device=args.device,
        dtype=dtype,
    )
    if args.force_eager_attention:
        maybe_force_eager_attention(model)
        print("Forced eager attention for teacher-forced cache extraction.")

    start_id, end_id, eos_id = prepare_special_token_ids(tokenizer)
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else eos_id
    caption_prompt_built = build_prompt(args.caption_prompt, model_config=model.config, with_image=True, num_frames=1)
    prompt_input_ids = tokenize_prompt(caption_prompt_built, tokenizer, device=model.device)
    runtime_dtypes = get_runtime_dtypes(model)
    runtime_backbone_dtype = next(model.get_model().parameters()).dtype

    total_errors = 0
    images_with_any_noun_chunk = 0
    images_with_any_head_prior = 0
    images_with_any_phrase_prior = 0
    total_cached_noun_chunks = 0
    total_valid_head_maps = 0
    total_valid_phrase_maps = 0
    caption_batch_fallback_count = 0
    trace_batch_fallback_count = 0
    debug_saved = 0

    annotations: Dict[str, Dict[str, Any]] = dict(resume_state["existing_annotations"])
    shard_writer = SlotMapShardWriter(
        os.path.join(args.output_dir, "shards"),
        shard_size=args.map_shard_size,
        start_shard_idx=int(resume_state["next_shard_idx"]),
    )
    shard_writer.shard_paths.extend(resume_state["existing_shard_paths"])

    metadata_open_mode = "a" if os.path.isfile(metadata_path) else "w"

    with open(metadata_path, metadata_open_mode, encoding="utf-8") as metadata_file, tqdm(
        total=len(image_paths),
        desc="StageA Train Cache",
        unit="img",
        dynamic_ncols=True,
    ) as pbar:
        for batch_idx, batch_paths in enumerate(batched(image_paths, args.caption_batch_size), start=1):
            batch_items: List[Dict[str, Any]] = []
            for image_path in batch_paths:
                record: Dict[str, Any] = {
                    "image": os.path.abspath(image_path),
                    "image_id": Path(image_path).stem,
                    "caption_prompt": args.caption_prompt,
                    "caption_prompt_built": caption_prompt_built,
                    "num_shards": int(args.num_shards),
                    "shard_index": int(args.shard_index),
                    "trace_source": "teacher_forced_attention",
                    "trace_last_n_layers": int(args.trace_last_n_layers),
                    "max_slots": int(args.max_slots),
                    "selection_strategy": "none",
                }
                try:
                    image = load_image_rgb(image_path)
                    image_tensor, _ = preprocess_single_image(
                        image,
                        image_processor,
                        device=model.device,
                        dtype=runtime_backbone_dtype,
                    )
                    item = {
                        "image_path": image_path,
                        "image_id": Path(image_path).stem,
                        "record": record,
                        "image_tensor": image_tensor,
                    }
                    if args.save_debug_images and debug_saved < args.debug_limit:
                        item["source_vis"] = denormalize_image_tensor(image_tensor, image_processor)[0].detach().cpu()
                    batch_items.append(item)
                except Exception as exc:
                    total_errors += 1
                    record["error"] = f"{type(exc).__name__}: {exc}"
                    record["caption"] = ""
                    record["token_ids"] = []
                    record["noun_chunks"] = []
                    record["n_cached_chunks"] = 0
                    record["phrase_valid_mask"] = [False] * int(args.max_slots)
                    record["head_valid_mask"] = [False] * int(args.max_slots)
                    finalized = shard_writer.append(
                        record=record,
                        image_id=record["image_id"],
                        phrase_maps=torch.zeros((args.max_slots, 256), dtype=torch.float32),
                        head_maps=torch.zeros((args.max_slots, 256), dtype=torch.float32),
                        phrase_valid_mask=torch.zeros(args.max_slots, dtype=torch.bool),
                        head_valid_mask=torch.zeros(args.max_slots, dtype=torch.bool),
                    )
                    finalize_records(finalized, metadata_file, annotations)
                    pbar.update(1)

            if not batch_items:
                pbar.set_postfix(
                    batch=batch_idx,
                    head_valid=images_with_any_head_prior,
                    errors=total_errors,
                    caption_bs=args.caption_batch_size,
                    trace_bs=args.trace_batch_size,
                )
                continue

            images_tensor = torch.stack([item["image_tensor"] for item in batch_items], dim=0)
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
            except Exception:
                caption_batch_fallback_count += 1
                caption_batch = {
                    "captions": [],
                    "generated_ids": [],
                    "token_texts": [],
                    "prefix_inputs_embeds": [],
                    "image_token_span": None,
                    "stop_reasons": [],
                    "cache_mode": "fallback_single",
                }
                recovered_items: List[Dict[str, Any]] = []
                for item in batch_items:
                    try:
                        single_input_ids = prompt_input_ids
                        single_images = item["image_tensor"]
                        single_result = generate_captions_batched_kv(
                            model=model,
                            tokenizer=tokenizer,
                            prompt_input_ids=single_input_ids,
                            images_tensor=single_images,
                            eos_token_id=eos_id,
                            max_new_tokens=args.caption_max_new_tokens,
                            start_image_token_id=start_id,
                            end_image_token_id=end_id,
                            use_kv_cache=use_kv_cache,
                        )
                        recovered = dict(item)
                        recovered["caption_info"] = {
                            "caption": single_result["captions"][0],
                            "generated_ids": single_result["generated_ids"][0],
                            "token_texts": single_result["token_texts"][0],
                            "prefix_inputs_embeds": single_result["prefix_inputs_embeds"][0:1],
                            "stop_reason": single_result["stop_reasons"][0],
                            "image_token_span": single_result["image_token_span"],
                            "cache_mode": single_result["cache_mode"],
                        }
                        recovered_items.append(recovered)
                    except Exception as exc:
                        total_errors += 1
                        record = item["record"]
                        record["error"] = f"{type(exc).__name__}: {exc}"
                        record["caption"] = ""
                        record["token_ids"] = []
                        record["noun_chunks"] = []
                        record["n_cached_chunks"] = 0
                        record["phrase_valid_mask"] = [False] * int(args.max_slots)
                        record["head_valid_mask"] = [False] * int(args.max_slots)
                        finalized = shard_writer.append(
                            record=record,
                            image_id=record["image_id"],
                            phrase_maps=torch.zeros((args.max_slots, 256), dtype=torch.float32),
                            head_maps=torch.zeros((args.max_slots, 256), dtype=torch.float32),
                            phrase_valid_mask=torch.zeros(args.max_slots, dtype=torch.bool),
                            head_valid_mask=torch.zeros(args.max_slots, dtype=torch.bool),
                        )
                        finalize_records(finalized, metadata_file, annotations)
                        pbar.update(1)
                batch_items = recovered_items
            else:
                for row_idx, item in enumerate(batch_items):
                    item["caption_info"] = {
                        "caption": caption_batch["captions"][row_idx],
                        "generated_ids": caption_batch["generated_ids"][row_idx],
                        "token_texts": caption_batch["token_texts"][row_idx],
                        "prefix_inputs_embeds": caption_batch["prefix_inputs_embeds"][row_idx : row_idx + 1],
                        "stop_reason": caption_batch["stop_reasons"][row_idx],
                        "image_token_span": caption_batch["image_token_span"],
                        "cache_mode": caption_batch["cache_mode"],
                    }

            if not batch_items:
                pbar.set_postfix(
                    batch=batch_idx,
                    head_valid=images_with_any_head_prior,
                    errors=total_errors,
                    caption_bs=args.caption_batch_size,
                    trace_bs=args.trace_batch_size,
                )
                continue

            docs = list(
                nlp.pipe(
                    [item["caption_info"]["caption"] for item in batch_items],
                    batch_size=min(args.spacy_batch_size, max(1, len(batch_items))),
                )
            )
            for item, doc in zip(batch_items, docs):
                caption_info = item["caption_info"]
                noun_info = extract_noun_info_from_token_ids_with_doc(
                    token_ids=caption_info["generated_ids"],
                    tokenizer=tokenizer,
                    doc=doc,
                    max_caption_tokens=args.max_caption_tokens,
                )
                noun_chunks = noun_info["noun_chunks"][: args.max_slots]
                item["noun_info"] = noun_info
                item["noun_chunks"] = noun_chunks

                record = item["record"]
                record["caption"] = noun_info["caption"]
                record["token_ids"] = noun_info["token_ids"]
                record["noun_chunks"] = noun_chunks
                record["selection_strategy"] = noun_info["selection_strategy"]
                record["stop_reason"] = caption_info["stop_reason"]
                record["cache_mode"] = caption_info["cache_mode"] + "+teacher_forced"
                record["n_cached_chunks"] = int(len(noun_chunks))
                if noun_chunks:
                    images_with_any_noun_chunk += 1
                    total_cached_noun_chunks += len(noun_chunks)
                    record["first_noun_phrase"] = noun_chunks[0]["text"]
                    record["noun_head"] = noun_chunks[0]["head_text"]
                else:
                    record["first_noun_phrase"] = None
                    record["noun_head"] = None

            for trace_items in batched(batch_items, args.trace_batch_size):
                prefix_batch = torch.cat([item["caption_info"]["prefix_inputs_embeds"] for item in trace_items], dim=0)
                generated_ids_list = [item["caption_info"]["generated_ids"] for item in trace_items]
                image_span = trace_items[0]["caption_info"]["image_token_span"]
                try:
                    step_maps_batch = teacher_forced_extract_step_maps_batched(
                        model=model,
                        prefix_inputs_embeds=prefix_batch,
                        generated_ids_list=generated_ids_list,
                        image_start=int(image_span[0]),
                        image_end=int(image_span[1]),
                        last_n_layers=args.trace_last_n_layers,
                        pad_token_id=pad_token_id,
                    )
                except Exception:
                    trace_batch_fallback_count += 1
                    step_maps_batch = []
                    for item in trace_items:
                        try:
                            row_maps = teacher_forced_extract_step_maps_batched(
                                model=model,
                                prefix_inputs_embeds=item["caption_info"]["prefix_inputs_embeds"],
                                generated_ids_list=[item["caption_info"]["generated_ids"]],
                                image_start=int(item["caption_info"]["image_token_span"][0]),
                                image_end=int(item["caption_info"]["image_token_span"][1]),
                                last_n_layers=args.trace_last_n_layers,
                                pad_token_id=pad_token_id,
                            )[0]
                            step_maps_batch.append(row_maps)
                        except Exception as exc:
                            total_errors += 1
                            item["record"]["error"] = f"{type(exc).__name__}: {exc}"
                            step_maps_batch.append([])

                for item, step_maps in zip(trace_items, step_maps_batch):
                    record = item["record"]
                    noun_chunks = item["noun_chunks"]
                    phrase_maps = torch.zeros((args.max_slots, 256), dtype=torch.float32)
                    head_maps = torch.zeros((args.max_slots, 256), dtype=torch.float32)
                    phrase_valid_mask = torch.zeros(args.max_slots, dtype=torch.bool)
                    head_valid_mask = torch.zeros(args.max_slots, dtype=torch.bool)

                    for slot_idx, chunk in enumerate(noun_chunks):
                        phrase_map, phrase_valid = aggregate_step_maps(
                            step_maps,
                            int(chunk["token_start"]),
                            int(chunk["token_end"]),
                        )
                        head_map, head_valid = aggregate_step_maps(
                            step_maps,
                            int(chunk["head_token_start"]),
                            int(chunk["head_token_end"]),
                        )
                        if phrase_valid:
                            phrase_maps[slot_idx] = phrase_map
                            phrase_valid_mask[slot_idx] = True
                        if head_valid:
                            head_maps[slot_idx] = head_map
                            head_valid_mask[slot_idx] = True

                    record["phrase_valid_mask"] = [bool(x) for x in phrase_valid_mask.tolist()]
                    record["head_valid_mask"] = [bool(x) for x in head_valid_mask.tolist()]
                    record["any_phrase_prior"] = bool(phrase_valid_mask.any().item())
                    record["any_head_prior"] = bool(head_valid_mask.any().item())

                    if record["any_phrase_prior"]:
                        images_with_any_phrase_prior += 1
                        total_valid_phrase_maps += int(phrase_valid_mask.sum().item())
                    if record["any_head_prior"]:
                        images_with_any_head_prior += 1
                        total_valid_head_maps += int(head_valid_mask.sum().item())

                    if args.save_debug_images and debug_saved < args.debug_limit and noun_chunks and "source_vis" in item:
                        sample_dir = os.path.join(args.output_dir, "samples", record["image_id"])
                        ensure_output_dir(sample_dir)
                        input_path = os.path.join(sample_dir, "input_processed.png")
                        first_phrase_map_path = os.path.join(sample_dir, "slot0_phrase_map.png")
                        first_head_map_path = os.path.join(sample_dir, "slot0_head_map.png")
                        first_phrase_overlay_path = os.path.join(sample_dir, "slot0_phrase_overlay.png")
                        first_head_overlay_path = os.path.join(sample_dir, "slot0_head_overlay.png")
                        tensor_to_pil(item["source_vis"]).save(input_path)
                        make_attention_heatmap(phrase_maps[0]).save(first_phrase_map_path)
                        make_attention_heatmap(head_maps[0]).save(first_head_map_path)
                        tensor_to_pil(
                            make_attention_overlay(item["source_vis"], phrase_maps[0], color=(1.0, 0.15, 0.15))
                        ).save(first_phrase_overlay_path)
                        tensor_to_pil(
                            make_attention_overlay(item["source_vis"], head_maps[0], color=(0.15, 0.65, 1.0))
                        ).save(first_head_overlay_path)
                        record["debug_paths"] = {
                            "input_processed": os.path.abspath(input_path),
                            "slot0_phrase_map": os.path.abspath(first_phrase_map_path),
                            "slot0_head_map": os.path.abspath(first_head_map_path),
                            "slot0_phrase_overlay": os.path.abspath(first_phrase_overlay_path),
                            "slot0_head_overlay": os.path.abspath(first_head_overlay_path),
                        }
                        save_json(os.path.join(sample_dir, "record.json"), record)
                        debug_saved += 1

                    finalized = shard_writer.append(
                        record=record,
                        image_id=record["image_id"],
                        phrase_maps=phrase_maps,
                        head_maps=head_maps,
                        phrase_valid_mask=phrase_valid_mask,
                        head_valid_mask=head_valid_mask,
                    )
                    finalize_records(finalized, metadata_file, annotations)
                    pbar.update(1)

            metadata_file.flush()
            os.fsync(metadata_file.fileno())

            pbar.set_postfix(
                batch=batch_idx,
                head_valid=images_with_any_head_prior,
                noun_imgs=images_with_any_noun_chunk,
                errors=total_errors,
                caption_bs=args.caption_batch_size,
                trace_bs=args.trace_batch_size,
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
        "image_dir": os.path.abspath(args.image_dir),
        "output_dir": os.path.abspath(args.output_dir),
        "model_path": args.model_path,
        "caption_prompt": args.caption_prompt,
        "caption_prompt_built": caption_prompt_built,
        "spacy_model": args.spacy_model,
        "device": str(model.device),
        "requested_dtype": str(dtype),
        "dtype": runtime_dtypes["model_dtype_property"],
        "backbone_dtype": runtime_dtypes["backbone_dtype"],
        "lm_head_dtype": runtime_dtypes["lm_head_dtype"],
        "vision_tower_dtype": runtime_dtypes["vision_tower_dtype"],
        "caption_max_new_tokens": int(args.caption_max_new_tokens),
        "max_caption_tokens": int(args.max_caption_tokens),
        "trace_last_n_layers": int(args.trace_last_n_layers),
        "caption_batch_size": int(args.caption_batch_size),
        "trace_batch_size": int(args.trace_batch_size),
        "spacy_batch_size": int(args.spacy_batch_size),
        "max_slots": int(args.max_slots),
        "use_kv_cache": bool(use_kv_cache),
        "force_eager_attention": bool(args.force_eager_attention),
        "seed": int(args.seed),
        "num_shards": int(args.num_shards),
        "shard_index": int(args.shard_index),
        "requested_samples": int(len(image_paths_all_for_shard)),
        "newly_processed_samples": int(len(image_paths)),
        "resumed_sample_count": int(resumed_sample_count),
        "global_requested_samples": int(global_requested_samples),
        "annotation_entry_count": int(len(ordered_annotations)),
        "images_with_any_noun_chunk": int(images_with_any_noun_chunk + hist_counts_merged["images_with_any_noun_chunk"]),
        "images_with_any_head_prior": int(images_with_any_head_prior + hist_counts_merged["images_with_any_head_prior"]),
        "images_with_any_phrase_prior": int(images_with_any_phrase_prior + hist_counts_merged["images_with_any_phrase_prior"]),
        "total_cached_noun_chunks": int(total_cached_noun_chunks + hist_counts_merged["total_cached_noun_chunks"]),
        "total_valid_head_maps": int(total_valid_head_maps + hist_counts_merged["total_valid_head_maps"]),
        "total_valid_phrase_maps": int(total_valid_phrase_maps + hist_counts_merged["total_valid_phrase_maps"]),
        "error_count": int(total_errors + hist_counts_merged["total_errors"]),
        "caption_batch_fallback_count": int(caption_batch_fallback_count),
        "trace_batch_fallback_count": int(trace_batch_fallback_count),
        "debug_saved_count": int(debug_saved),
        "metadata_jsonl": os.path.abspath(metadata_path),
        "annotations_json": os.path.abspath(annotations_path),
        "map_shards": list(shard_writer.shard_paths),
        "notes": [
            "This cache is optimized for Stage B training: batched caption generation plus batched teacher-forced attention extraction.",
            "All noun chunks up to max_slots are stored in mention order so Stage B can start with first-object and later expand to multi-slot.",
            "Both phrase and head maps are saved in the shard payload; head maps are expected to be the primary prior candidate.",
            "captionslot_annotations.json is compatible with the current CaptionSlotDataset and also records prior shard locations.",
            "Resume-aware run: metadata.jsonl is the source of truth; re-invocation skips image_ids already present.",
        ],
    }
    save_json(summary_path, summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
