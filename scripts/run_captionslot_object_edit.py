#!/usr/bin/env python
"""Probe object-level CaptionSlot editing by swapping donor object slot states."""

import argparse
import json
import os
import sys
import types
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import torch
from torch.utils.data import DataLoader

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[0]
INFERENCE_ROOT = REPO_ROOT / "inference"
SCALE_RAE_ROOT = Path("/home/jovyan/Scale-RAE")
SCALE_RAE_INFERENCE_ROOT = SCALE_RAE_ROOT / "inference"
for path in (str(REPO_ROOT), str(INFERENCE_ROOT), str(SCALE_RAE_ROOT), str(SCALE_RAE_INFERENCE_ROOT)):
    if path not in sys.path:
        sys.path.append(path)

if "IPython" not in sys.modules:
    ipython_stub = types.ModuleType("IPython")
    ipython_stub.get_ipython = lambda: None
    ipython_stub.version_info = (0, 0, 0, "")
    sys.modules["IPython"] = ipython_stub

from eval_caption_to_image_rfid import build_abs_diff_image, save_triptych, tensor_to_pil  # type: ignore
from build_stagea_object_steervit_prior_cache import build_caption_from_object_texts  # type: ignore
from eval_captionslot_checkpoint import (  # type: ignore
    CaptionSlotEvalCollator,
    CaptionSlotEvalDataset,
    decode_generated_images,
    denormalize_images,
    get_image_stats,
    load_caption_records,
    load_eval_decoder,
    maybe_inject_lora_from_checkpoint,
    patch_diffusion_steps,
    resolve_dtype,
)
from scale_rae.train.captionslot_trainer import (  # type: ignore
    _register_captionslot_template_token_ids,
    _register_im_start_end_token_ids,
)
from scale_rae.utils import disable_torch_init  # type: ignore
from utils.load_model import load_scale_rae_model  # type: ignore


DEFAULT_MODEL_PATH = "/home/jovyan/AURORA/checkpoints/aurora_refcoco_singlephase_maxpool_lora_continue_lowlr_fp32/best-checkpoint"
DEFAULT_IMAGE_DIR = "/home/jovyan/data/coco/val2017"
DEFAULT_CAPTIONS_JSONL = "/home/jovyan/Scale-RAE/outputs/caption_to_image_recon_concise/captions.jsonl"
DEFAULT_OUTPUT_DIR = "/home/jovyan/AURORA/outputs/captionslot_object_edit_probe"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Swap one donor object slot into a target CaptionSlot reconstruction.")
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--image-dir", default=DEFAULT_IMAGE_DIR)
    parser.add_argument("--captions-jsonl", default=DEFAULT_CAPTIONS_JSONL)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=["fp16", "bf16", "fp32"], default="fp32")
    parser.add_argument("--target-index", type=int, default=0)
    parser.add_argument("--donor-index", type=int, default=1)
    parser.add_argument("--target-image", default="")
    parser.add_argument("--donor-image", default="")
    parser.add_argument("--target-object-text", default="")
    parser.add_argument("--donor-object-text", default="")
    parser.add_argument("--target-object-index", type=int, default=0)
    parser.add_argument("--donor-object-index", type=int, default=0)
    parser.add_argument("--max-caption-tokens", type=int, default=64)
    parser.add_argument("--guidance-level", type=float, default=1.0)
    parser.add_argument("--diffusion-steps", type=int, default=10)
    parser.add_argument("--save-ablations", action="store_true")
    return parser.parse_args()


def _select_two_records(records: Sequence[Dict[str, Any]], target_index: int, donor_index: int) -> List[Dict[str, Any]]:
    valid = [r for r in records if r.get("token_ids") and r.get("noun_chunks")]
    if len(valid) < 2:
        raise SystemExit("Need at least two records with token_ids and noun_chunks.")
    if target_index < 0 or target_index >= len(valid):
        raise SystemExit(f"--target-index out of range for {len(valid)} valid records.")
    if donor_index < 0 or donor_index >= len(valid):
        raise SystemExit(f"--donor-index out of range for {len(valid)} valid records.")
    if target_index == donor_index:
        raise SystemExit("Use different --target-index and --donor-index values.")
    return [valid[target_index], valid[donor_index]]


def _make_single_object_record(
    image_path: str,
    object_text: str,
    tokenizer,
    max_caption_tokens: int,
    max_slots: int,
) -> Dict[str, Any]:
    if not image_path:
        raise SystemExit("Image path is required for a custom record.")
    if not object_text.strip():
        raise SystemExit(f"--*-object-text is required for custom image {image_path}.")
    serialized = build_caption_from_object_texts(
        tokenizer=tokenizer,
        object_texts=[object_text.strip()],
        max_caption_tokens=max_caption_tokens,
        max_slots=max_slots,
    )
    image_path = str(Path(image_path).expanduser())
    return {
        "image": image_path,
        "image_id": Path(image_path).stem,
        "file_name": Path(image_path).name,
        "caption": serialized["caption"],
        "token_ids": serialized["token_ids"],
        "noun_chunks": serialized["noun_chunks"],
        "object_texts": serialized["object_texts"],
    }


def _object_slot_mask(
    total_slots: int,
    slots_per_object: int,
    object_index: int,
    active_slots: int,
    device: torch.device,
) -> torch.Tensor:
    mask = torch.zeros((1, total_slots), dtype=torch.bool, device=device)
    start = int(object_index) * int(slots_per_object)
    end = min(start + int(slots_per_object), int(active_slots), total_slots)
    if start < 0 or start >= end:
        raise SystemExit(
            f"Object index {object_index} is invalid for active_slots={active_slots}, "
            f"slots_per_object={slots_per_object}."
        )
    mask[:, start:end] = True
    return mask


def _copy_object_slots(
    target_slots: torch.Tensor,
    donor_slots: torch.Tensor,
    target_mask: torch.Tensor,
    donor_mask: torch.Tensor,
) -> torch.Tensor:
    edited = target_slots.clone()
    target_idx = target_mask[0].nonzero(as_tuple=False).flatten()
    donor_idx = donor_mask[0].nonzero(as_tuple=False).flatten()
    n = min(int(target_idx.numel()), int(donor_idx.numel()))
    if n <= 0:
        raise SystemExit("No slot rows selected for swap.")
    edited[:, target_idx[:n]] = donor_slots[:, donor_idx[:n]]
    return edited


def _active_slot_mask(total_slots: int, active_slots: int, device: torch.device) -> torch.Tensor:
    mask = torch.zeros((1, total_slots), dtype=torch.bool, device=device)
    mask[:, : max(0, min(int(active_slots), int(total_slots)))] = True
    return mask


def _all_token_mask(x: torch.Tensor) -> torch.Tensor:
    return torch.ones(x.shape[:2], dtype=torch.bool, device=x.device)


def _object_texts_from_record(record: Dict[str, Any]) -> List[str]:
    object_texts = [str(x).strip() for x in (record.get("object_texts") or []) if str(x).strip()]
    if object_texts:
        return object_texts
    return [str(x.get("text", "")).strip() for x in (record.get("noun_chunks") or []) if str(x.get("text", "")).strip()]


def _build_caption_swapped_record(
    target_record: Dict[str, Any],
    donor_record: Dict[str, Any],
    target_object_index: int,
    donor_object_index: int,
    tokenizer,
    max_caption_tokens: int,
    max_slots: int,
) -> Dict[str, Any]:
    target_texts = _object_texts_from_record(target_record)
    donor_texts = _object_texts_from_record(donor_record)
    if target_object_index < 0 or target_object_index >= len(target_texts):
        raise SystemExit(f"Target object index {target_object_index} is invalid for {len(target_texts)} target objects.")
    if donor_object_index < 0 or donor_object_index >= len(donor_texts):
        raise SystemExit(f"Donor object index {donor_object_index} is invalid for {len(donor_texts)} donor objects.")

    edited_texts = list(target_texts)
    edited_texts[target_object_index] = donor_texts[donor_object_index]
    serialized = build_caption_from_object_texts(
        tokenizer=tokenizer,
        object_texts=edited_texts,
        max_caption_tokens=max_caption_tokens,
        max_slots=max_slots,
    )
    edited = dict(target_record)
    edited["caption"] = serialized["caption"]
    edited["token_ids"] = serialized["token_ids"]
    edited["noun_chunks"] = serialized["noun_chunks"]
    edited["object_texts"] = serialized["object_texts"]
    edited["file_name"] = Path(str(target_record.get("image") or target_record.get("file_name"))).name
    return edited


def _save_recon_set(output_dir: str, name: str, source: torch.Tensor, recon: torch.Tensor) -> None:
    out_dir = os.path.join(output_dir, "samples")
    os.makedirs(out_dir, exist_ok=True)
    input_img = tensor_to_pil(source)
    recon_img = tensor_to_pil(recon)
    diff_img = build_abs_diff_image(source, recon)
    input_img.save(os.path.join(out_dir, f"{name}_input.png"))
    recon_img.save(os.path.join(out_dir, f"{name}_recon.png"))
    diff_img.save(os.path.join(out_dir, f"{name}_abs_diff.png"))
    save_triptych(input_img, recon_img, diff_img, os.path.join(out_dir, f"{name}_triptych.png"))


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    disable_torch_init()

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    dtype = resolve_dtype(args.dtype)

    tokenizer, model, image_processors, _ = load_scale_rae_model(
        model_path=args.model_path,
        device=str(device),
        dtype=dtype,
    )
    image_processor = image_processors[0]
    target_image_processor = image_processors[1] if len(image_processors) > 1 else image_processor
    if tokenizer.pad_token is None:
        tokenizer.pad_token = "<|endoftext|>"
        tokenizer.pad_token_id = 151643
    _register_im_start_end_token_ids(model, tokenizer)
    if not hasattr(model, "captionslot_system_prefix_ids"):
        _register_captionslot_template_token_ids(model, tokenizer)
    maybe_inject_lora_from_checkpoint(model, args.model_path)
    model = model.to(device).eval()
    patch_diffusion_steps(model, args.diffusion_steps)
    decoder = load_eval_decoder(model).to(device).eval()

    max_slots = int(getattr(model.config, "captionslot_max_slots", 10))
    slots_per_object = int(getattr(model.config, "captionslot_slots_per_object", 1))
    image_feature_token_len = int(getattr(model.config, "image_feature_token_len", getattr(model, "num_image_tokens", 256)))
    if args.target_image or args.donor_image:
        if not args.target_image or not args.donor_image:
            raise SystemExit("--target-image and --donor-image must be provided together.")
        selected = [
            _make_single_object_record(
                args.target_image,
                args.target_object_text,
                tokenizer,
                args.max_caption_tokens,
                max_slots,
            ),
            _make_single_object_record(
                args.donor_image,
                args.donor_object_text,
                tokenizer,
                args.max_caption_tokens,
                max_slots,
            ),
        ]
    else:
        records = load_caption_records(args.captions_jsonl, args.image_dir, max_samples=None)
        selected = _select_two_records(records, args.target_index, args.donor_index)
    edited_record = _build_caption_swapped_record(
        target_record=selected[0],
        donor_record=selected[1],
        target_object_index=args.target_object_index,
        donor_object_index=args.donor_object_index,
        tokenizer=tokenizer,
        max_caption_tokens=args.max_caption_tokens,
        max_slots=max_slots,
    )

    dataset = CaptionSlotEvalDataset(
        selected,
        tokenizer=tokenizer,
        image_processor=image_processor,
        target_image_processor=target_image_processor,
        max_slots=max_slots,
        slots_per_object=slots_per_object,
        image_feature_token_len=image_feature_token_len,
        max_caption_tokens=args.max_caption_tokens,
    )
    edited_dataset = CaptionSlotEvalDataset(
        [edited_record],
        tokenizer=tokenizer,
        image_processor=image_processor,
        target_image_processor=target_image_processor,
        max_slots=max_slots,
        slots_per_object=slots_per_object,
        image_feature_token_len=image_feature_token_len,
        max_caption_tokens=args.max_caption_tokens,
    )
    batch = next(iter(DataLoader(dataset, batch_size=2, collate_fn=CaptionSlotEvalCollator(tokenizer.pad_token_id))))
    edited_batch = next(iter(DataLoader(edited_dataset, batch_size=1, collate_fn=CaptionSlotEvalCollator(tokenizer.pad_token_id))))
    batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
    edited_batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in edited_batch.items()}

    with torch.no_grad():
        base = model.generate_captionslot(
            images=batch["images"],
            caption_input_ids=batch["caption_input_ids"],
            caption_attention_mask=batch["caption_attention_mask"],
            noun_chunk_spans=batch["noun_chunk_spans"],
            n_slots=batch["n_slots"],
            guidance_level=args.guidance_level,
            return_generated=True,
            return_intermediates=True,
        )

        donor_slot_hidden = base["slot_hidden"][1:2]
        target_slot_hidden = base["slot_hidden"][0:1]
        target_reg_hidden = base["reg_hidden"][0:1]
        target_active_slots = int(batch["n_slots"][0].item())
        donor_active_slots = int(batch["n_slots"][1].item())
        target_mask = _object_slot_mask(
            max_slots, slots_per_object, args.target_object_index, target_active_slots, device
        )
        donor_mask = _object_slot_mask(
            max_slots, slots_per_object, args.donor_object_index, donor_active_slots, device
        )
        edited_slots = _copy_object_slots(target_slot_hidden, donor_slot_hidden, target_mask, donor_mask)
        edited_active_mask = _active_slot_mask(
            max_slots,
            int(edited_batch["n_slots"][0].item()),
            device,
        )
        target_reg_mask = _all_token_mask(target_reg_hidden)

        edited = model.generate_captionslot(
            images=edited_batch["images"],
            caption_input_ids=edited_batch["caption_input_ids"],
            caption_attention_mask=edited_batch["caption_attention_mask"],
            noun_chunk_spans=edited_batch["noun_chunk_spans"],
            n_slots=edited_batch["n_slots"],
            guidance_level=args.guidance_level,
            return_generated=True,
            slot_input_overrides=edited_slots,
            slot_input_override_mask=edited_active_mask,
            reg_input_overrides=target_reg_hidden,
            reg_input_override_mask=target_reg_mask,
            slot_hidden_overrides=edited_slots,
            slot_hidden_override_mask=edited_active_mask,
            reg_hidden_overrides=target_reg_hidden,
            reg_hidden_override_mask=target_reg_mask,
            fixed_condition_hidden_for_rae=True,
        )
        caption_swap = None
        if args.save_ablations:
            caption_swap = model.generate_captionslot(
                images=edited_batch["images"],
                caption_input_ids=edited_batch["caption_input_ids"],
                caption_attention_mask=edited_batch["caption_attention_mask"],
                noun_chunk_spans=edited_batch["noun_chunk_spans"],
                n_slots=edited_batch["n_slots"],
                guidance_level=args.guidance_level,
                return_generated=True,
            )

        recon_base = decode_generated_images(decoder, base["generated"], device)
        recon_edited = decode_generated_images(decoder, edited["generated"], device)
        recon_caption = (
            decode_generated_images(decoder, caption_swap["generated"], device)
            if caption_swap is not None
            else None
        )
        mean, std = get_image_stats(target_image_processor)
        source = denormalize_images(batch["target_images"], mean, std, to_cpu=False)
        edited_source = denormalize_images(edited_batch["target_images"], mean, std, to_cpu=False)

    _save_recon_set(args.output_dir, "target_base", source[0:1], recon_base[0:1])
    _save_recon_set(args.output_dir, "donor_base", source[1:2], recon_base[1:2])
    _save_recon_set(args.output_dir, "target_edited", edited_source[0:1], recon_edited[0:1])
    if recon_caption is not None:
        _save_recon_set(args.output_dir, "target_caption_swap", edited_source[0:1], recon_caption[0:1])

    metadata = {
        "model_path": args.model_path,
        "target": {
            "index": args.target_index,
            "image": batch["image_path"][0],
            "caption": batch["caption"][0],
            "object_index": args.target_object_index,
            "noun_chunk": dataset.entries[0]["noun_chunks"][args.target_object_index],
        },
        "donor": {
            "index": args.donor_index,
            "image": batch["image_path"][1],
            "caption": batch["caption"][1],
            "object_index": args.donor_object_index,
            "noun_chunk": dataset.entries[1]["noun_chunks"][args.donor_object_index],
        },
        "edited": {
            "caption": edited_dataset.entries[0]["caption"],
            "object_texts": edited_dataset.entries[0]["noun_chunks"],
            "swapped_target_object_index": args.target_object_index,
            "swapped_donor_object_index": args.donor_object_index,
        },
        "slots_per_object": slots_per_object,
        "target_active_slots": target_active_slots,
        "donor_active_slots": donor_active_slots,
        "edited_active_slots": int(edited_batch["n_slots"][0].item()),
        "edit_mechanism": {
            "caption_swapped": True,
            "slot_input_overridden_before_mllm_forward": True,
            "target_register_input_preserved_before_mllm_forward": True,
            "latent_query_condition_hidden_fixed_each_mllm_layer": True,
            "latent_query_direct_image_attention": False,
            "slot_hidden_overridden_for_dit_cross_attention": True,
            "target_register_hidden_preserved_for_dit_cross_attention": True,
            "dit_cross_attention_uses_new_mllm_slot_register": False,
            "rae_query_recomputed_after_slot_input_override": True,
        },
        "outputs": {
            "target_base": "samples/target_base_recon.png",
            "donor_base": "samples/donor_base_recon.png",
            "edited": "samples/target_edited_recon.png",
        },
    }
    if args.save_ablations:
        metadata["outputs"]["caption_swap_ablation"] = "samples/target_caption_swap_recon.png"
    with open(os.path.join(args.output_dir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print(f"[DONE] Saved object-edit probe to {args.output_dir}")
    print(json.dumps(metadata, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
