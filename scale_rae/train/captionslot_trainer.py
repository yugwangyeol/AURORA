"""
CaptionSlot RefCOCO-family trainer.
"""

import json
import logging
import math
import os
import pickle
import random
import time
from contextlib import nullcontext
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Set

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
import transformers
from transformers import trainer as hf_trainer
from transformers import utils as hf_utils
from transformers import Trainer
from transformers.trainer import ALL_LAYERNORM_LAYERS, get_parameter_names, is_sagemaker_mp_enabled

from scale_rae.constants import DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN
from scale_rae.model.language_model.scale_rae_qwen2 import ScaleRAEQwenForCausalLM

logger = logging.getLogger(__name__)
_TRUTHY_ENV_VALUES = {"1", "true", "yes", "y", "on"}


REFCOCO_SPECS = {
    "refcoco": {
        "ref_file": "refs(unc).p",
        "default_train_splits": {"train"},
        "default_eval_splits": {"val"},
    },
    "refcoco+": {
        "ref_file": "refs(unc).p",
        "default_train_splits": {"train"},
        "default_eval_splits": {"val"},
    },
    "refcocog": {
        "ref_file": "refs(umd).p",
        "default_train_splits": {"train"},
        "default_eval_splits": {"val"},
    },
}


def _is_captionslot_diffusion_adaln_param(name: str) -> bool:
    return "diff_head.model." in name and "adaLN_modulation.1" in name


def _is_captionslot_sparse_xattn_param(name: str) -> bool:
    return "slot_cross_attn" in name


def _is_captionslot_llm_unfreeze_param(
    name: str,
    start_layer: int,
    num_layers: int,
    attn_only: bool,
) -> bool:
    prefix = "model.layers."
    if num_layers <= 0 or not name.startswith(prefix):
        return False
    rest = name[len(prefix):]
    layer_text = rest.split(".", 1)[0]
    if not layer_text.isdigit():
        return False
    layer_idx = int(layer_text)
    if layer_idx < start_layer or layer_idx >= num_layers:
        return False
    if not attn_only:
        return True
    return ".self_attn." in name or ".input_layernorm." in name


def _get_mask_utils():
    try:
        import pycocotools.mask as mask_utils
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "pycocotools is required for RefCOCO-family mask decoding."
        ) from exc
    return mask_utils


def decode_segmentation(segmentation, height: int, width: int) -> np.ndarray:
    mask_utils = _get_mask_utils()
    if isinstance(segmentation, list):
        rles = mask_utils.frPyObjects(segmentation, height, width)
        rle = mask_utils.merge(rles)
    elif isinstance(segmentation, dict) and isinstance(segmentation.get("counts"), list):
        rle = mask_utils.frPyObjects(segmentation, height, width)
    else:
        rle = segmentation

    mask = mask_utils.decode(rle)
    if mask.ndim == 3:
        mask = np.any(mask, axis=2)
    return mask.astype(np.float32)


def _set_rng_state_for_device_safe(
    device_name: str,
    device_module,
    checkpoint_rng_state: Dict,
    is_distributed: bool,
) -> None:
    device_state_key = device_name.lower()
    if device_state_key not in checkpoint_rng_state:
        return
    try:
        if is_distributed:
            device_module.random.set_rng_state_all(checkpoint_rng_state[device_state_key])
        else:
            device_module.random.set_rng_state(checkpoint_rng_state[device_state_key])
    except Exception as exc:
        logger.error(
            "[CaptionSlot] Failed to restore %s RNG state during resume: %s",
            device_name,
            exc,
        )


def mask_to_patch_mask(mask: torch.Tensor, grid_size: int) -> torch.Tensor:
    mask = mask.float().unsqueeze(0).unsqueeze(0)
    patch = F.interpolate(mask, size=(grid_size, grid_size), mode="area")
    return patch.squeeze(0).squeeze(0).reshape(-1).clamp(0.0, 1.0)


def _split_csv(value: str) -> List[str]:
    if value is None:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def _dedupe_texts(texts: Sequence[str]) -> List[str]:
    seen = set()
    result: List[str] = []
    for text in texts:
        clean = " ".join(str(text).strip().split())
        if not clean:
            continue
        key = clean.lower()
        if key in seen:
            continue
        seen.add(key)
        result.append(clean)
    return result


def _resolve_image_path(coco_root: str, image_id: int) -> str:
    file_name = f"{int(image_id):012d}.jpg"
    for split in ("train2017", "val2017"):
        path = os.path.join(coco_root, split, file_name)
        if os.path.exists(path):
            return path
    raise FileNotFoundError(
        f"Could not find image {file_name} under {coco_root}/train2017 or {coco_root}/val2017"
    )


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None or not str(value).strip():
        return default
    return str(value).strip().lower() in _TRUTHY_ENV_VALUES


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None or not str(value).strip():
        return default
    try:
        return int(str(value).strip())
    except ValueError:
        logger.warning("[CaptionSlot] Ignoring invalid integer env %s=%r", name, value)
        return default


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    version: Optional[str] = field(default="qwen_2")
    freeze_backbone: bool = field(default=False)
    tune_mm_mlp_adapter: bool = field(default=False)
    tune_adapter_and_vision_head: bool = field(default=False)
    vision_tower_aux_list: Optional[str] = field(default=None)
    mm_vision_select_layer: Optional[int] = field(default=-1)
    pretrain_mm_mlp_adapter: Optional[str] = field(default=None)
    pretrain_adapter_and_vision_head: Optional[str] = field(default=None)
    mm_projector_type: Optional[str] = field(default="linear")
    mm_use_im_start_end: bool = field(default=False)
    mm_use_im_patch_token: bool = field(default=True)
    mm_vision_select_feature: Optional[str] = field(default="patch")
    vision_tower_aux_token_len_list: Optional[str] = field(default=None)
    vision_hidden_size: Optional[int] = field(default=1024)
    connector_only: bool = field(default=True)
    vision_loss: Optional[str] = field(default="diffusion-loss")
    vision_loss_mode: Optional[str] = field(default="query")
    vision_coef: Optional[float] = field(default=1.0)
    dit_cls: Optional[str] = field(default="DiT")
    diffusion_model_hidden_size: Optional[int] = field(default=1152)
    diffusion_model_channels: Optional[int] = field(default=1152)
    diffusion_model_depth: Optional[int] = field(default=12)
    diffusion_model_heads: Optional[int] = field(default=16)
    diffusion_model_z_channels: Optional[int] = field(default=0)
    si_token_len: int = field(default=729)
    miv_token_len: int = field(default=196)
    image_feature_token_len: Optional[int] = field(default=None)
    diffusion_target_token_len: Optional[int] = field(default=None)
    diffusion_norm_stats_path: Optional[str] = field(default=None)

    use_captionslot: bool = field(default=True)
    captionslot_max_slots: int = field(default=10)
    captionslot_slots_per_object: int = field(default=1)
    captionslot_n_register: int = field(default=8)
    captionslot_cmd_length: int = field(default=8)
    captionslot_recon_loss_weight: float = field(default=1.0)
    captionslot_mask_bce_loss_weight: float = field(default=1.0)
    captionslot_mask_tversky_loss_weight: Optional[float] = field(default=None)
    captionslot_mask_dice_loss_weight: float = field(default=1.0)  # legacy fallback
    captionslot_mask_balanced_bce: bool = field(default=False)
    captionslot_mask_tversky_alpha: float = field(default=0.5)
    captionslot_mask_tversky_beta: float = field(default=0.5)
    captionslot_object_cam_loss_weight: float = field(default=1.0)
    captionslot_register_cam_loss_weight: float = field(default=0.3)
    captionslot_cam_layers: str = field(default="-1")
    captionslot_cam_eps: float = field(default=1e-6)
    captionslot_caption_loss_weight: float = field(default=0.0)  # legacy no-op
    captionslot_diversity_loss_weight: float = field(default=0.0)  # legacy no-op
    captionslot_training_stage: int = field(default=1)
    captionslot_condition_gate_init: float = field(default=0.1)
    captionslot_train_latent_queries: bool = field(default=False)
    captionslot_unfreeze_diff_head_body: bool = field(default=False)
    captionslot_unfreeze_llm_last_n_layers: int = field(default=0)
    captionslot_unfreeze_llm_attn_only: bool = field(default=True)
    captionslot_attention_use_layer_norm: bool = field(default=True)
    captionslot_attention_temperature: float = field(default=1.0)
    captionslot_prior_bias_scale: float = field(default=0.0)  # legacy no-op
    captionslot_control_mode: str = field(default="slots")
    captionslot_add_cross_attention: bool = field(default=False)
    captionslot_cross_attention_start_block: int = field(default=8)
    captionslot_cross_attention_every_n_blocks: int = field(default=4)
    captionslot_cross_attention_include_registers: bool = field(default=True)
    captionslot_cross_attention_gate_init: float = field(default=0.05)


@dataclass
class DataArguments:
    captionslot_annotation_path: Optional[str] = field(
        default=None,
        metadata={"help": "Legacy no-op. RefCOCO samples are built directly from captionslot_coco_root."},
    )
    captionslot_image_folder: Optional[str] = field(
        default=None,
        metadata={"help": "Legacy no-op. Images are resolved from captionslot_coco_root/train2017|val2017."},
    )
    captionslot_coco_root: str = field(default="/home/jovyan/data/coco")
    captionslot_datasets: str = field(default="refcoco,refcoco+,refcocog")
    captionslot_train_splits: str = field(default="train")
    captionslot_eval_splits: str = field(default="val")
    captionslot_min_area: float = field(default=0.0)
    image_aspect_ratio: str = field(default="square")
    max_images_per_sample: int = field(default=1)
    max_caption_tokens: int = field(default=192)

    image_processor_aux_list: Optional[list] = field(default=None, repr=False, init=False)
    vision_tower_aux_token_len_list: Optional[list] = field(default=None, repr=False, init=False)
    is_multimodal: bool = field(default=False, init=False)


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    remove_unused_columns: bool = field(default=False)
    model_max_length: int = field(default=2048)
    unfreeze_mm_vision_tower: bool = field(default=False)
    diff_head_lr: Optional[float] = None
    group_by_modality_length: bool = field(default=False)
    captionslot_latent_query_lr: Optional[float] = None
    captionslot_llm_lr: Optional[float] = None
    captionslot_eval_num_images: int = field(default=128)
    # Warmup–Stable–Decay (WSD) schedule. If enabled, overrides the HF scheduler with a
    # LambdaLR profile: warmup (warmup_ratio) → stable (max LR) → cosine decay to 0
    # during the last wsd_decay_fraction of total steps.
    captionslot_use_wsd_schedule: bool = field(default=False)
    captionslot_wsd_decay_fraction: float = field(default=0.10)
    captionslot_use_cosine_min_lr_schedule: bool = field(default=False)
    captionslot_min_lr_ratio: float = field(default=0.10)

    # LoRA on LLM attention (q_proj, k_proj, v_proj, o_proj). When enabled, base LLM weights
    # remain frozen and adapters carry the LLM-side adaptation.
    captionslot_lora_enable: bool = field(default=False)
    captionslot_lora_r: int = field(default=16)
    captionslot_lora_alpha: int = field(default=32)
    captionslot_lora_dropout: float = field(default=0.05)
    captionslot_lora_target_modules: str = field(
        default="q_proj,k_proj,v_proj,o_proj",
        metadata={"help": "Comma-separated module-name suffixes to wrap with LoRA."},
    )


class RefCOCOGroupedDataset(Dataset):
    def __init__(
        self,
        coco_root: str,
        dataset_names: Sequence[str],
        split_names: Sequence[str],
        tokenizer,
        data_args: DataArguments,
        model_config,
    ) -> None:
        super().__init__()
        if not os.path.isdir(coco_root):
            raise FileNotFoundError(coco_root)
        if data_args.image_processor_aux_list is None:
            raise ValueError("image_processor_aux_list must be initialized before building the dataset.")

        self.coco_root = coco_root
        self.dataset_names = list(dataset_names)
        self.split_names: Set[str] = set(split_names)
        self.tokenizer = tokenizer
        self.data_args = data_args
        self.model_config = model_config
        self.max_slots = int(getattr(model_config, "captionslot_max_slots", 10))
        self.slots_per_object = max(int(getattr(model_config, "captionslot_slots_per_object", 1)), 1)
        self.max_unique_objects = max(self.max_slots // self.slots_per_object, 1)
        if self.slots_per_object > 1 and self.max_slots % self.slots_per_object != 0:
            logger.warning(
                "[CaptionSlot] captionslot_max_slots=%d is not divisible by captionslot_slots_per_object=%d; "
                "the final %d slot(s) will stay unused.",
                self.max_slots,
                self.slots_per_object,
                self.max_slots % self.slots_per_object,
            )
        token_count = int(
            getattr(
                model_config,
                "image_feature_token_len",
                getattr(model_config, "vision_tower_aux_token_len_list", [256])[0],
            )
        )
        self.grid_size = int(round(math.sqrt(token_count)))
        if self.grid_size * self.grid_size != token_count:
            raise ValueError(f"Expected square image token grid, got {token_count}")

        self.entries = self._build_entries()
        if not self.entries:
            raise ValueError(
                f"No valid RefCOCO-family entries found for datasets={self.dataset_names} splits={sorted(self.split_names)}"
            )

    def _build_entries(self) -> List[Dict]:
        entries: List[Dict] = []
        for dataset_name in self.dataset_names:
            if dataset_name not in REFCOCO_SPECS:
                raise ValueError(f"Unsupported RefCOCO-family dataset: {dataset_name}")
            spec = REFCOCO_SPECS[dataset_name]
            ref_path = os.path.join(self.coco_root, dataset_name, spec["ref_file"])
            instances_path = os.path.join(self.coco_root, dataset_name, "instances.json")
            if not os.path.isfile(ref_path):
                raise FileNotFoundError(ref_path)
            if not os.path.isfile(instances_path):
                raise FileNotFoundError(instances_path)

            with open(ref_path, "rb") as f:
                refs = pickle.load(f)
            with open(instances_path, "r") as f:
                instances = json.load(f)

            images_by_id = {int(image["id"]): image for image in instances.get("images", [])}
            anns_by_id = {int(ann["id"]): ann for ann in instances.get("annotations", [])}
            grouped: Dict[int, Dict] = {}

            for ref in refs:
                split = str(ref.get("split", "")).strip()
                if split not in self.split_names:
                    continue
                ann_id = int(ref["ann_id"])
                image_id = int(ref["image_id"])
                ann = anns_by_id.get(ann_id)
                image_info = images_by_id.get(image_id)
                if ann is None or image_info is None:
                    continue
                if int(ann.get("iscrowd", 0)) != 0:
                    continue
                if float(ann.get("area", 0.0)) < float(self.data_args.captionslot_min_area):
                    continue

                entry = grouped.setdefault(
                    image_id,
                    {
                        "dataset_name": dataset_name,
                        "split": split,
                        "image_id": image_id,
                        "image_info": image_info,
                        "objects": {},
                    },
                )
                obj = entry["objects"].setdefault(
                    ann_id,
                    {
                        "ann_id": ann_id,
                        "ann": ann,
                        "sentences": [],
                    },
                )
                obj["sentences"].extend(
                    sent.get("sent") or sent.get("raw") or ""
                    for sent in ref.get("sentences", [])
                )

            for image_id in sorted(grouped.keys()):
                entry = grouped[image_id]
                objects: List[Dict] = []
                for ann_id, obj in entry["objects"].items():
                    deduped_sentences = _dedupe_texts(obj["sentences"])
                    if not deduped_sentences:
                        continue
                    objects.append(
                        {
                            "ann_id": ann_id,
                            "ann": obj["ann"],
                            "sentences": deduped_sentences,
                            "area": float(obj["ann"].get("area", 0.0)),
                        }
                    )
                if not objects:
                    continue
                objects.sort(key=lambda item: item["area"], reverse=True)
                entries.append(
                    {
                        "dataset_name": entry["dataset_name"],
                        "split": entry["split"],
                        "image_id": entry["image_id"],
                        "image_info": entry["image_info"],
                        "objects": objects,
                    }
                )

        entries.sort(key=lambda item: (item["dataset_name"], item["split"], int(item["image_id"])))
        logger.info(
            "[CaptionSlot] Built %d grouped image samples from datasets=%s splits=%s",
            len(entries),
            ",".join(self.dataset_names),
            ",".join(sorted(self.split_names)),
        )
        return entries

    def __len__(self) -> int:
        return len(self.entries)

    def _sample_object_text(self, object_entry: Dict) -> str:
        return random.choice(object_entry["sentences"])

    def _build_patch_mask(
        self,
        ann: Dict,
        image_info: Dict,
        processed_hw: Sequence[int],
    ) -> torch.Tensor:
        mask_np = decode_segmentation(
            ann["segmentation"],
            height=int(image_info["height"]),
            width=int(image_info["width"]),
        )
        mask = torch.tensor(mask_np, dtype=torch.float32)
        mask = F.interpolate(
            mask.unsqueeze(0).unsqueeze(0),
            size=tuple(int(v) for v in processed_hw),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0).squeeze(0)
        return mask_to_patch_mask(mask, grid_size=self.grid_size)

    def __getitem__(self, idx: int) -> Dict:
        entry = self.entries[idx]
        image_path = _resolve_image_path(self.coco_root, int(entry["image_id"]))
        image = Image.open(image_path).convert("RGB")

        processor = self.data_args.image_processor_aux_list[0]
        target_processor = (
            self.data_args.image_processor_aux_list[1]
            if len(self.data_args.image_processor_aux_list) > 1
            else processor
        )
        image_tensor = processor.preprocess(image, return_tensors="pt")["pixel_values"][0]
        target_image_tensor = target_processor.preprocess(image, return_tensors="pt")["pixel_values"][0]
        processed_hw = image_tensor.shape[-2:]

        sep_tokens = self.tokenizer.encode(" . ", add_special_tokens=False)
        ref_spans = torch.full((self.max_slots, 2), -1, dtype=torch.long)
        gt_masks_patches = torch.zeros((self.max_slots, self.grid_size * self.grid_size), dtype=torch.float32)

        caption_token_ids: List[int] = []
        caption_object_texts: List[str] = []
        slot_object_texts: List[str] = []
        selected_unique_objects = 0
        selected_slots = 0

        for object_entry in entry["objects"][: self.max_unique_objects]:
            text = self._sample_object_text(object_entry)
            token_ids = self.tokenizer.encode(text, add_special_tokens=False)
            if not token_ids:
                continue

            prefix = sep_tokens if caption_token_ids else []
            if caption_token_ids and len(caption_token_ids) + len(prefix) + len(token_ids) > self.data_args.max_caption_tokens:
                break
            if not caption_token_ids and len(token_ids) > self.data_args.max_caption_tokens:
                token_ids = token_ids[: self.data_args.max_caption_tokens]
            if not token_ids:
                continue

            if prefix:
                caption_token_ids.extend(prefix)
            span_start = len(caption_token_ids)
            caption_token_ids.extend(token_ids)
            span_end = len(caption_token_ids)

            patch_mask = self._build_patch_mask(
                object_entry["ann"],
                entry["image_info"],
                processed_hw,
            )
            caption_object_texts.append(text)
            for _ in range(self.slots_per_object):
                if selected_slots >= self.max_slots:
                    break
                ref_spans[selected_slots, 0] = span_start
                ref_spans[selected_slots, 1] = span_end
                gt_masks_patches[selected_slots] = patch_mask
                slot_object_texts.append(text)
                selected_slots += 1
            selected_unique_objects += 1

        if selected_unique_objects == 0:
            object_entry = entry["objects"][0]
            text = self._sample_object_text(object_entry)
            token_ids = self.tokenizer.encode(text, add_special_tokens=False)[: self.data_args.max_caption_tokens]
            if not token_ids:
                token_ids = self.tokenizer.encode("object", add_special_tokens=False)
            patch_mask = self._build_patch_mask(
                object_entry["ann"],
                entry["image_info"],
                processed_hw,
            )
            slots_to_fill = min(self.slots_per_object, self.max_slots)
            for slot_idx in range(slots_to_fill):
                ref_spans[slot_idx, 0] = 0
                ref_spans[slot_idx, 1] = len(token_ids)
                gt_masks_patches[slot_idx] = patch_mask
                slot_object_texts.append(text)
            caption_token_ids = token_ids
            caption_object_texts = [text]
            selected_unique_objects = 1
            selected_slots = slots_to_fill

        caption_input_ids = torch.tensor(caption_token_ids, dtype=torch.long)
        return {
            "image": image_tensor,
            "target_image": target_image_tensor,
            "caption_input_ids": caption_input_ids,
            "ref_spans": ref_spans,
            "n_slots": torch.tensor(selected_slots, dtype=torch.long),
            "gt_masks_patches": gt_masks_patches,
            "image_id": str(entry["image_id"]),
            "dataset_name": entry["dataset_name"],
            "caption_text": " . ".join(caption_object_texts),
            "object_texts": slot_object_texts,
            "n_unique_objects": torch.tensor(selected_unique_objects, dtype=torch.long),
            "clean_num_objects": torch.tensor(len(entry["objects"]), dtype=torch.long),
        }


@dataclass
class CaptionSlotDataCollator:
    pad_token_id: int

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        max_caption_len = max(inst["caption_input_ids"].shape[0] for inst in instances)
        batch_size = len(instances)
        caption_input_ids = torch.full(
            (batch_size, max_caption_len),
            self.pad_token_id,
            dtype=torch.long,
        )
        caption_attention_mask = torch.zeros((batch_size, max_caption_len), dtype=torch.bool)

        for idx, inst in enumerate(instances):
            length = inst["caption_input_ids"].shape[0]
            caption_input_ids[idx, :length] = inst["caption_input_ids"]
            caption_attention_mask[idx, :length] = True

        return {
            "images": torch.stack([inst["image"] for inst in instances]),
            "target_images": torch.stack([inst["target_image"] for inst in instances]),
            "caption_input_ids": caption_input_ids,
            "caption_attention_mask": caption_attention_mask,
            "ref_spans": torch.stack([inst["ref_spans"] for inst in instances]),
            "n_slots": torch.stack([inst["n_slots"] for inst in instances]),
            "gt_masks_patches": torch.stack([inst["gt_masks_patches"] for inst in instances]),
            "image_ids": [inst["image_id"] for inst in instances],
            "dataset_names": [inst["dataset_name"] for inst in instances],
            "caption_texts": [inst["caption_text"] for inst in instances],
            "object_texts": [inst["object_texts"] for inst in instances],
            "n_unique_objects": torch.stack([inst["n_unique_objects"] for inst in instances]),
            "clean_num_objects": torch.stack([inst["clean_num_objects"] for inst in instances]),
        }


class CaptionSlotTrainer(Trainer):
    def _log_live_model_devices_once(self, inner, images: torch.Tensor) -> None:
        if getattr(self, "_captionslot_logged_live_devices", False):
            return

        def _first_param_device(module):
            if module is None:
                return "none"
            try:
                return str(next(module.parameters()).device)
            except (StopIteration, AttributeError):
                return "none"

        vision_device = "none"
        try:
            vt_list = inner.get_vision_tower_aux_list()
            if vt_list:
                vision_device = ",".join(
                    f"{idx}:{_first_param_device(vt)}"
                    for idx, vt in enumerate(vt_list)
                )
        except Exception:
            vision_device = "unavailable"

        logger.info(
            "[CaptionSlot] live devices: rank=%s images=%s backbone=%s lm_head=%s vision=%s diff_head=%s diff_head_projector=%s cmd=%s latent_queries=%s",
            getattr(self.args, "local_rank", -1),
            str(images.device),
            _first_param_device(getattr(inner, "model", None)),
            _first_param_device(getattr(inner, "lm_head", None)),
            vision_device,
            _first_param_device(getattr(inner, "diff_head", None)),
            _first_param_device(getattr(inner, "diff_head_projector", None)),
            str(getattr(inner, "captionslot_cmd_embeddings", torch.empty(0)).device),
            str(getattr(inner.get_model(), "latent_queries", torch.empty(0)).device),
        )
        self._captionslot_logged_live_devices = True

    def create_optimizer(self):
        if is_sagemaker_mp_enabled():
            return super().create_optimizer()

        if self.optimizer is not None:
            return self.optimizer

        opt_model = self.model
        decay_parameters = get_parameter_names(opt_model, ALL_LAYERNORM_LAYERS)
        decay_parameters = [name for name in decay_parameters if "bias" not in name]

        diff_head_lr = self.args.diff_head_lr if self.args.diff_head_lr is not None else self.args.learning_rate
        latent_query_lr = (
            self.args.captionslot_latent_query_lr
            if self.args.captionslot_latent_query_lr is not None
            else self.args.learning_rate
        )
        llm_lr = (
            self.args.captionslot_llm_lr
            if self.args.captionslot_llm_lr is not None
            else self.args.learning_rate
        )

        projector_names = {
            name for name, param in opt_model.named_parameters()
            if param.requires_grad and "diff_head_projector" in name
        }
        diff_adapter_names = {
            name
            for name, param in opt_model.named_parameters()
            if param.requires_grad and (
                name.startswith("captionslot_context_projector")
                or "diff_head.model." in name  # covers AdaLN+xattn (partial) and full body (unfreeze_diff_head_body)
            )
        }
        latent_query_names = {
            name for name, param in opt_model.named_parameters()
            if param.requires_grad and "latent_queries" in name
        }
        llm_adapter_names = {
            name for name, param in opt_model.named_parameters()
            if param.requires_grad and name.startswith("model.layers.")
        }

        optimizer_grouped_parameters = [
            {
                "params": [
                    p for n, p in opt_model.named_parameters()
                    if (
                        p.requires_grad
                        and n in decay_parameters
                        and n not in projector_names
                        and n not in diff_adapter_names
                        and n not in latent_query_names
                        and n not in llm_adapter_names
                    )
                ],
                "weight_decay": self.args.weight_decay,
                "lr": self.args.learning_rate,
            },
            {
                "params": [
                    p for n, p in opt_model.named_parameters()
                    if (
                        p.requires_grad
                        and n not in decay_parameters
                        and n not in projector_names
                        and n not in diff_adapter_names
                        and n not in latent_query_names
                        and n not in llm_adapter_names
                    )
                ],
                "weight_decay": 0.0,
                "lr": self.args.learning_rate,
            },
        ]

        if projector_names:
            optimizer_grouped_parameters.append(
                {
                    "params": [
                        p for n, p in opt_model.named_parameters()
                        if p.requires_grad and n in projector_names
                    ],
                    "weight_decay": self.args.weight_decay,
                    "lr": diff_head_lr,
                }
            )

        if diff_adapter_names:
            optimizer_grouped_parameters.extend(
                [
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters()
                            if p.requires_grad and n in diff_adapter_names and n in decay_parameters
                        ],
                        "weight_decay": self.args.weight_decay,
                        "lr": diff_head_lr,
                    },
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters()
                            if p.requires_grad and n in diff_adapter_names and n not in decay_parameters
                        ],
                        "weight_decay": 0.0,
                        "lr": diff_head_lr,
                    },
                ]
            )

        if latent_query_names:
            optimizer_grouped_parameters.append(
                {
                    "params": [
                        p for n, p in opt_model.named_parameters()
                        if p.requires_grad and n in latent_query_names
                    ],
                    "weight_decay": 0.0,
                    "lr": latent_query_lr,
                }
            )

        if llm_adapter_names:
            optimizer_grouped_parameters.extend(
                [
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters()
                            if p.requires_grad and n in llm_adapter_names and n in decay_parameters
                        ],
                        "weight_decay": self.args.weight_decay,
                        "lr": llm_lr,
                    },
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters()
                            if p.requires_grad and n in llm_adapter_names and n not in decay_parameters
                        ],
                        "weight_decay": 0.0,
                        "lr": llm_lr,
                    },
                ]
            )

        # Break LLM group down into LoRA vs. base for log clarity.
        llm_lora_names = {n for n in llm_adapter_names if "lora_" in n}
        llm_base_names = llm_adapter_names - llm_lora_names
        logger.info(
            "[CaptionSlot] optimizer groups | main_lr=%s diff_head_lr=%s latent_query_lr=%s llm_lr=%s | "
            "projector=%d diff_adapter=%d latent_queries=%d llm=%d (lora=%d, base=%d)",
            self.args.learning_rate,
            diff_head_lr,
            latent_query_lr,
            llm_lr,
            len(projector_names),
            len(diff_adapter_names),
            len(latent_query_names),
            len(llm_adapter_names),
            len(llm_lora_names),
            len(llm_base_names),
        )

        optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)
        self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
        return self.optimizer

    def create_scheduler(self, num_training_steps: int, optimizer: Optional[torch.optim.Optimizer] = None):
        if bool(getattr(self.args, "captionslot_use_cosine_min_lr_schedule", False)):
            if self.lr_scheduler is not None:
                return self.lr_scheduler

            target_optimizer = optimizer if optimizer is not None else self.optimizer
            if target_optimizer is None:
                raise ValueError("Optimizer must be created before the scheduler.")

            warmup_steps = int(self.args.get_warmup_steps(num_training_steps))
            min_ratio = float(getattr(self.args, "captionslot_min_lr_ratio", 0.10))
            min_ratio = min(max(min_ratio, 0.0), 1.0)
            logger.info(
                "[CaptionSlot] Using cosine-min scheduler: total=%d warmup=%d min_lr_ratio=%.4f",
                num_training_steps,
                warmup_steps,
                min_ratio,
            )

            def cosine_min_lambda(current_step: int) -> float:
                if current_step < warmup_steps:
                    return float(current_step) / float(max(1, warmup_steps))
                progress = float(current_step - warmup_steps) / float(max(1, num_training_steps - warmup_steps))
                progress = min(max(progress, 0.0), 1.0)
                cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
                return min_ratio + (1.0 - min_ratio) * cosine

            from torch.optim.lr_scheduler import LambdaLR

            self.lr_scheduler = LambdaLR(target_optimizer, cosine_min_lambda)
            self._created_lr_scheduler = True
            return self.lr_scheduler

        if not bool(getattr(self.args, "captionslot_use_wsd_schedule", False)):
            return super().create_scheduler(num_training_steps=num_training_steps, optimizer=optimizer)

        if self.lr_scheduler is not None:
            return self.lr_scheduler

        target_optimizer = optimizer if optimizer is not None else self.optimizer
        if target_optimizer is None:
            raise ValueError("Optimizer must be created before the scheduler.")

        warmup_steps = int(self.args.get_warmup_steps(num_training_steps))
        decay_fraction = float(getattr(self.args, "captionslot_wsd_decay_fraction", 0.10))
        decay_fraction = min(max(decay_fraction, 0.0), 1.0)
        decay_steps = int(num_training_steps * decay_fraction)
        decay_start = max(num_training_steps - decay_steps, warmup_steps)

        logger.info(
            "[CaptionSlot] Using WSD scheduler: total=%d warmup=%d stable=[%d,%d) decay=[%d,%d]",
            num_training_steps,
            warmup_steps,
            warmup_steps,
            decay_start,
            decay_start,
            num_training_steps,
        )

        def wsd_lambda(current_step: int) -> float:
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            if current_step < decay_start:
                return 1.0
            # Cosine decay from 1.0 → 0.0 over [decay_start, num_training_steps]
            progress = float(current_step - decay_start) / float(max(1, num_training_steps - decay_start))
            progress = min(max(progress, 0.0), 1.0)
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        from torch.optim.lr_scheduler import LambdaLR

        self.lr_scheduler = LambdaLR(target_optimizer, wsd_lambda)
        self._created_lr_scheduler = True
        return self.lr_scheduler

    def _load_optimizer_and_scheduler(self, checkpoint):
        try:
            return super()._load_optimizer_and_scheduler(checkpoint)
        except ValueError as exc:
            if "different number of parameter groups" not in str(exc):
                raise

            trainer_state_path = os.path.join(checkpoint, "trainer_state.json")
            resumed_step = 0
            if os.path.isfile(trainer_state_path):
                try:
                    with open(trainer_state_path, "r") as f:
                        resumed_step = int(json.load(f).get("global_step", 0))
                except Exception:
                    resumed_step = 0

            logger.warning(
                "[CaptionSlot] Resume optimizer mismatch for %s; "
                "keeping current optimizer state and re-aligning scheduler from global_step=%d. "
                "This is expected after changing trainable parameter groups.",
                checkpoint,
                resumed_step,
            )

            if self.lr_scheduler is not None and resumed_step > 0:
                try:
                    self.lr_scheduler.step(resumed_step)
                except TypeError:
                    for _ in range(resumed_step):
                        self.lr_scheduler.step()
                logger.info(
                    "[CaptionSlot] Scheduler advanced to resumed step=%d after optimizer reset.",
                    resumed_step,
                )

            return

    def _load_rng_state(self, checkpoint):
        try:
            return super()._load_rng_state(checkpoint)
        except pickle.UnpicklingError as exc:
            if "Weights only load failed" not in str(exc):
                raise

            if checkpoint is None:
                return

            if self.args.world_size > 1:
                process_index = self.args.process_index
                rng_file = os.path.join(checkpoint, f"rng_state_{process_index}.pth")
            else:
                rng_file = os.path.join(checkpoint, "rng_state.pth")

            if not os.path.isfile(rng_file):
                logger.warning(
                    "[CaptionSlot] RNG state file missing during fallback resume: %s. "
                    "Continuing without restoring RNG state.",
                    rng_file,
                )
                return

            logger.warning(
                "[CaptionSlot] Retrying RNG state load with weights_only=False for trusted local checkpoint: %s",
                rng_file,
            )
            checkpoint_rng_state = torch.load(rng_file, map_location="cpu", weights_only=False)

            random.setstate(checkpoint_rng_state["python"])
            np.random.set_state(checkpoint_rng_state["numpy"])
            torch.random.set_rng_state(checkpoint_rng_state["cpu"])

            is_torch_xla_available = getattr(hf_utils, "is_torch_xla_available", lambda: False)
            is_torch_npu_available = getattr(hf_utils, "is_torch_npu_available", lambda: False)
            is_torch_hpu_available = getattr(hf_utils, "is_torch_hpu_available", lambda: False)
            is_torch_mlu_available = getattr(hf_utils, "is_torch_mlu_available", lambda: False)
            is_torch_musa_available = getattr(hf_utils, "is_torch_musa_available", lambda: False)

            if is_torch_xla_available():
                from torch_xla.core import xla_model as xm

                xm.set_rng_state(checkpoint_rng_state["xla"])

            is_distributed = self.args.parallel_mode == hf_trainer.ParallelMode.DISTRIBUTED
            if torch.cuda.is_available():
                _set_rng_state_for_device_safe("CUDA", torch.cuda, checkpoint_rng_state, is_distributed)
            if is_torch_npu_available():
                _set_rng_state_for_device_safe("NPU", torch.npu, checkpoint_rng_state, is_distributed)
            if is_torch_hpu_available():
                _set_rng_state_for_device_safe("HPU", torch.hpu, checkpoint_rng_state, is_distributed)
            if is_torch_mlu_available():
                _set_rng_state_for_device_safe("MLU", torch.mlu, checkpoint_rng_state, is_distributed)
            if is_torch_musa_available():
                _set_rng_state_for_device_safe("MUSA", torch.musa, checkpoint_rng_state, is_distributed)

            logger.info("[CaptionSlot] RNG state restored via trusted fallback load.")

    def compute_loss(self, model, inputs, return_outputs=False):
        images = inputs.pop("images")
        target_images = inputs.pop("target_images", images)
        inner = model.module if hasattr(model, "module") else model
        self._log_live_model_devices_once(inner, images)

        outputs = model(
            input_ids=torch.zeros(images.shape[0], 1, dtype=torch.long, device=images.device),
            images=images,
            target_images=target_images,
            labels=None,
            caption_input_ids=inputs.pop("caption_input_ids"),
            caption_attention_mask=inputs.pop("caption_attention_mask"),
            ref_spans=inputs.pop("ref_spans"),
            n_slots=inputs.pop("n_slots"),
            gt_masks_patches=inputs.pop("gt_masks_patches"),
        )
        loss = outputs.loss if hasattr(outputs, "loss") else outputs[0]

        custom = getattr(self, "_custom_losses", {})
        for key, attr in [
            ("loss_recon", "captionslot_loss_recon"),
            ("loss_mask_bce", "captionslot_loss_mask_bce"),
            ("loss_mask_tversky", "captionslot_loss_mask_tversky"),
            ("loss_object_cam_ce", "captionslot_loss_object_cam_ce"),
            ("loss_register_cam_ce", "captionslot_loss_register_cam_ce"),
            ("slot_img_attention_mass", "captionslot_slot_img_attention_mass"),
            ("register_img_attention_mass", "captionslot_register_img_attention_mass"),
            ("avg_slots", "captionslot_avg_slots"),
            ("object_slot_attn_soft_iou", "captionslot_object_slot_attn_soft_iou"),
            ("object_slot_attn_l1", "captionslot_object_slot_attn_l1"),
            ("object_slot_attn_cosine", "captionslot_object_slot_attn_cosine"),
        ]:
            value = getattr(inner, attr, None)
            if value is not None:
                custom.setdefault(key, []).append(
                    value.item() if torch.is_tensor(value) else float(value)
                )
        unique_object_counts = inputs.get("n_unique_objects")
        if unique_object_counts is not None:
            custom.setdefault("avg_unique_objects", []).append(
                unique_object_counts.detach().float().mean().item()
            )
        self._custom_losses = custom
        return (loss, outputs) if return_outputs else loss

    def log(self, logs: Dict[str, float]) -> None:
        custom = getattr(self, "_custom_losses", {})
        prefix = getattr(self, "_custom_loss_prefix", None)
        for key, values in custom.items():
            if values:
                metric_name = f"{prefix}_{key}" if prefix else key
                logs[metric_name] = sum(values) / len(values)
        custom.clear()
        super().log(logs)

    def prediction_step(self, model, inputs, prediction_loss_only: bool, ignore_keys=None):
        if "images" not in inputs:
            return super().prediction_step(model, inputs, prediction_loss_only=prediction_loss_only, ignore_keys=ignore_keys)
        inputs = self._prepare_inputs(inputs)
        with torch.no_grad():
            loss = self.compute_loss(model, inputs, return_outputs=False)
        loss = loss.detach()
        return (loss, None, None)

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix: str = "eval"):
        prev_custom = getattr(self, "_custom_losses", {})
        prev_prefix = getattr(self, "_custom_loss_prefix", None)
        self._custom_losses = {}
        self._custom_loss_prefix = metric_key_prefix
        try:
            metrics = super().evaluate(
                eval_dataset=eval_dataset,
                ignore_keys=ignore_keys,
                metric_key_prefix=metric_key_prefix,
            )
            extra_metrics = {}
            try:
                extra_start = time.time()
                logger.info("[CaptionSlot] Computing extra eval metrics/reconstruction probes.")
                extra_metrics.update(
                    self._compute_captionslot_recon_metrics(
                        self.model,
                        eval_dataset=eval_dataset,
                        metric_key_prefix=metric_key_prefix,
                    )
                )
                extra_metrics.update(
                    self._compute_captionslot_online_attention_metrics(
                        self.model,
                        eval_dataset=eval_dataset,
                        metric_key_prefix=metric_key_prefix,
                    )
                )
                if extra_metrics:
                    metrics.update(extra_metrics)
                    self.log(extra_metrics)
                logger.info("[CaptionSlot] Extra eval metrics finished in %.1fs.", time.time() - extra_start)
            except Exception as exc:
                logger.warning("[CaptionSlot] Failed to compute online eval metrics: %s", exc)
            self._maybe_save_best_eval_checkpoint(metrics, metric_key_prefix=metric_key_prefix)
            try:
                self._maybe_log_captionslot_reconstructions(
                    self.model,
                    eval_dataset=eval_dataset,
                    metric_key_prefix=metric_key_prefix,
                )
            except Exception as exc:
                logger.warning("[CaptionSlot] Failed to log reconstructions: %s", exc)
            try:
                self._maybe_log_captionslot_slot_attention(
                    self.model,
                    eval_dataset=eval_dataset,
                    metric_key_prefix=metric_key_prefix,
                )
            except Exception as exc:
                logger.warning("[CaptionSlot] Failed to log slot attention maps: %s", exc)
            return metrics
        finally:
            self._custom_loss_prefix = prev_prefix
            self._custom_losses = prev_custom

    def _maybe_save_best_eval_checkpoint(
        self,
        metrics: Dict[str, float],
        metric_key_prefix: str = "eval",
    ) -> None:
        if not _env_flag("CAPTIONSLOT_SAVE_BEST_CHECKPOINT", default=False):
            return

        dist_ready = torch.distributed.is_available() and torch.distributed.is_initialized()
        sync_device = getattr(self.args, "device", torch.device("cpu"))

        def sync_save_status(saved: bool) -> None:
            if not dist_ready:
                return
            flag = torch.tensor(
                [1 if saved else 0],
                device=sync_device,
                dtype=torch.int32,
            )
            torch.distributed.broadcast(flag, src=0)
            if int(flag.item()) > 0:
                torch.distributed.barrier()

        if not self.is_world_process_zero():
            sync_save_status(False)
            return

        metric_name = os.environ.get("CAPTIONSLOT_BEST_METRIC_NAME", f"{metric_key_prefix}_loss").strip()
        if not metric_name:
            metric_name = f"{metric_key_prefix}_loss"

        metric_value = metrics.get(metric_name)
        if metric_value is None:
            logger.warning("[CaptionSlot] Best-checkpoint metric %s was not found in eval metrics.", metric_name)
            sync_save_status(False)
            return

        try:
            metric_value = float(metric_value)
        except (TypeError, ValueError):
            logger.warning("[CaptionSlot] Could not convert best-checkpoint metric %s=%r to float.", metric_name, metric_value)
            sync_save_status(False)
            return

        greater_is_better = _env_flag("CAPTIONSLOT_BEST_GREATER_IS_BETTER", default=False)
        best_metric = self.state.best_metric
        is_better = best_metric is None or (
            metric_value > best_metric if greater_is_better else metric_value < best_metric
        )
        if not is_better:
            sync_save_status(False)
            return

        best_subdir = os.environ.get("CAPTIONSLOT_BEST_CHECKPOINT_SUBDIR", "best-checkpoint").strip()
        if not best_subdir:
            best_subdir = "best-checkpoint"
        best_checkpoint_dir = os.path.join(self.args.output_dir, best_subdir)

        self.state.best_metric = metric_value
        self.state.best_model_checkpoint = best_checkpoint_dir

        if self.is_world_process_zero():
            logger.info(
                "[CaptionSlot] New best %s=%.6f at step=%s. Saving dedicated best checkpoint to %s",
                metric_name,
                metric_value,
                self.state.global_step,
                best_checkpoint_dir,
            )
            os.makedirs(best_checkpoint_dir, exist_ok=True)
            self.save_model(best_checkpoint_dir)
            with open(os.path.join(best_checkpoint_dir, "best_metric.json"), "w") as f:
                json.dump(
                    {
                        "best_metric_name": metric_name,
                        "best_metric_value": metric_value,
                        "best_global_step": int(self.state.global_step),
                        "greater_is_better": greater_is_better,
                        "metric_key_prefix": metric_key_prefix,
                    },
                    f,
                    indent=2,
                    sort_keys=True,
                )
            self.state.save_to_json(os.path.join(best_checkpoint_dir, "trainer_state.json"))

        sync_save_status(True)

    def _compute_captionslot_recon_metrics(
        self,
        model,
        eval_dataset=None,
        metric_key_prefix: str = "eval",
    ) -> Dict[str, float]:
        if not self.is_world_process_zero():
            return {}

        dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
        if dataset is None or len(dataset) == 0:
            return {}

        decoder = self._get_eval_decoder(model)
        if decoder is None:
            return {}

        eval_batch_size = max(int(getattr(self.args, "per_device_eval_batch_size", 1)), 1)
        loader = DataLoader(
            dataset,
            batch_size=eval_batch_size,
            shuffle=False,
            collate_fn=self.data_collator,
            num_workers=0,
        )

        mean, std = self._get_image_stats(dataset, processor_index=1)
        inner = model.module if hasattr(model, "module") else model
        device = inner.captionslot_cmd_embeddings.device
        was_training = inner.training

        total = 0
        l1_sum = 0.0
        mse_sum = 0.0
        psnr_sum = 0.0

        try:
            inner.eval()
            with torch.no_grad():
                for batch in loader:
                    images = batch["images"].to(device)
                    with self._get_autocast_context(device):
                        output = inner.generate_captionslot(
                            images=images,
                            caption_input_ids=batch["caption_input_ids"].to(device),
                            caption_attention_mask=batch["caption_attention_mask"].to(device),
                            ref_spans=batch["ref_spans"].to(device),
                            n_slots=batch["n_slots"].to(device),
                            guidance_level=1.0,
                        )
                        recon_images = self._decode_generated_images(
                            decoder,
                            output.get("generated"),
                            device=device,
                        )
                    source_images = self._denormalize_images(batch["target_images"], mean, std)
                    if recon_images.numel() == 0 or source_images.numel() == 0:
                        continue

                    recon_images = recon_images.to(dtype=torch.float32)
                    source_images = source_images.to(dtype=torch.float32)
                    per_image_l1 = torch.mean(torch.abs(recon_images - source_images), dim=(1, 2, 3))
                    per_image_mse = torch.mean((recon_images - source_images) ** 2, dim=(1, 2, 3))
                    per_image_psnr = -10.0 * torch.log10(per_image_mse.clamp_min(1e-8))

                    total += per_image_l1.numel()
                    l1_sum += per_image_l1.sum().item()
                    mse_sum += per_image_mse.sum().item()
                    psnr_sum += per_image_psnr.sum().item()
        finally:
            inner.train(was_training)

        if total == 0:
            return {}

        return {
            f"{metric_key_prefix}_recon_pixel_l1": l1_sum / total,
            f"{metric_key_prefix}_recon_pixel_mse": mse_sum / total,
            f"{metric_key_prefix}_recon_psnr": psnr_sum / total,
        }

    def _compute_captionslot_online_attention_metrics(
        self,
        model,
        eval_dataset=None,
        metric_key_prefix: str = "eval",
    ) -> Dict[str, float]:
        if not self.is_world_process_zero():
            return {}

        dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
        if dataset is None or len(dataset) == 0:
            return {}

        eval_batch_size = max(int(getattr(self.args, "per_device_eval_batch_size", 1)), 1)
        loader = DataLoader(
            dataset,
            batch_size=eval_batch_size,
            shuffle=False,
            collate_fn=self.data_collator,
            num_workers=0,
        )

        inner = model.module if hasattr(model, "module") else model
        device = inner.captionslot_cmd_embeddings.device
        was_training = inner.training

        active_slot_total = 0
        active_slot_capacity = 0
        dice_sum = 0.0
        iou_sum = 0.0
        object_dice_sum = 0.0
        object_iou_sum = 0.0
        object_total = 0
        entropy_sum = 0.0
        object_slot_soft_iou_sum = 0.0
        object_slot_l1_sum = 0.0
        object_slot_cosine_sum = 0.0
        object_slot_pair_total = 0
        unique_object_total = 0
        clean_object_total = 0
        processed_samples = 0
        slots_per_object = max(int(getattr(inner.config, "captionslot_slots_per_object", 1)), 1)

        try:
            inner.eval()
            with torch.no_grad():
                for batch in loader:
                    images = batch["images"].to(device)
                    with self._get_autocast_context(device):
                        output = inner.generate_captionslot(
                            images=images,
                            caption_input_ids=batch["caption_input_ids"].to(device),
                            caption_attention_mask=batch["caption_attention_mask"].to(device),
                            ref_spans=batch["ref_spans"].to(device),
                            n_slots=batch["n_slots"].to(device),
                            guidance_level=1.0,
                            return_generated=False,
                        )
                    attn_maps = output.get("attn_maps")
                    if attn_maps is None or attn_maps.numel() == 0:
                        continue

                    pred_attn = attn_maps.detach().cpu().float().clamp(0.0, 1.0)
                    gt_masks = batch["gt_masks_patches"].detach().cpu().float().clamp(0.0, 1.0)
                    slot_counts = batch["n_slots"].detach().cpu()
                    unique_object_counts = batch["n_unique_objects"].detach().cpu()
                    clean_object_counts = batch["clean_num_objects"].detach().cpu()

                    active_slot_capacity += int(pred_attn.shape[0] * pred_attn.shape[1])
                    for sample_idx in range(pred_attn.shape[0]):
                        processed_samples += 1
                        n_active = int(slot_counts[sample_idx].item())
                        n_unique = int(unique_object_counts[sample_idx].item())
                        active_slot_total += n_active
                        unique_object_total += n_unique
                        clean_object_total += int(clean_object_counts[sample_idx].item())
                        for slot_idx in range(n_active):
                            pred = pred_attn[sample_idx, slot_idx]
                            gt = gt_masks[sample_idx, slot_idx]
                            intersection = torch.minimum(pred, gt).sum()
                            union = (pred + gt - torch.minimum(pred, gt)).sum().clamp_min(1e-6)
                            soft_dice = (2.0 * intersection) / (pred.sum() + gt.sum()).clamp_min(1e-6)
                            soft_iou = intersection / union
                            entropy_sum += self._normalized_patch_entropy(self._normalize_patch_distribution(pred))
                            dice_sum += float(soft_dice.item())
                            iou_sum += float(soft_iou.item())
                        for object_idx in range(max(n_unique, 0)):
                            start_idx = object_idx * slots_per_object
                            end_idx = min(start_idx + slots_per_object, n_active)
                            if end_idx <= start_idx:
                                break
                            # MEAN aggregation across slots in same object (matches training-time loss).
                            pred = pred_attn[sample_idx, start_idx:end_idx].mean(dim=0)
                            gt = gt_masks[sample_idx, start_idx:end_idx].amax(dim=0)
                            intersection = torch.minimum(pred, gt).sum()
                            union = (pred + gt - torch.minimum(pred, gt)).sum().clamp_min(1e-6)
                            soft_dice = (2.0 * intersection) / (pred.sum() + gt.sum()).clamp_min(1e-6)
                            object_dice_sum += float(soft_dice.item())
                            object_iou_sum += float((intersection / union).item())
                            object_total += 1
                        if slots_per_object > 1:
                            for object_idx in range(n_unique):
                                start_idx = object_idx * slots_per_object
                                end_idx = min(start_idx + slots_per_object, n_active)
                                if end_idx - start_idx < 2:
                                    break
                                group = pred_attn[sample_idx, start_idx:end_idx]
                                for left_idx in range(group.shape[0]):
                                    for right_idx in range(left_idx + 1, group.shape[0]):
                                        left = group[left_idx]
                                        right = group[right_idx]
                                        intersection = torch.minimum(left, right).sum()
                                        union = (left + right - torch.minimum(left, right)).sum().clamp_min(1e-6)
                                        object_slot_soft_iou_sum += float((intersection / union).item())
                                        object_slot_l1_sum += float(torch.mean(torch.abs(left - right)).item())
                                        object_slot_cosine_sum += float(
                                            F.cosine_similarity(
                                                left.view(1, -1),
                                                right.view(1, -1),
                                                dim=-1,
                                                eps=1e-6,
                                            ).item()
                                        )
                                        object_slot_pair_total += 1
        finally:
            inner.train(was_training)

        if active_slot_total == 0:
            return {}

        metrics = {
            f"{metric_key_prefix}_active_slot_fraction": active_slot_total / max(active_slot_capacity, 1),
            f"{metric_key_prefix}_slot_soft_dice": dice_sum / active_slot_total,
            f"{metric_key_prefix}_slot_soft_iou": iou_sum / active_slot_total,
            f"{metric_key_prefix}_slot_attn_entropy": entropy_sum / active_slot_total,
        }
        if processed_samples > 0:
            metrics[f"{metric_key_prefix}_avg_unique_objects"] = unique_object_total / processed_samples
        if clean_object_total > 0:
            metrics[f"{metric_key_prefix}_supervised_object_fraction"] = unique_object_total / clean_object_total
        if object_total > 0:
            metrics[f"{metric_key_prefix}_object_soft_dice"] = object_dice_sum / object_total
            metrics[f"{metric_key_prefix}_object_soft_iou"] = object_iou_sum / object_total
        if object_slot_pair_total > 0:
            metrics[f"{metric_key_prefix}_object_slot_attn_soft_iou"] = object_slot_soft_iou_sum / object_slot_pair_total
            metrics[f"{metric_key_prefix}_object_slot_attn_l1"] = object_slot_l1_sum / object_slot_pair_total
            metrics[f"{metric_key_prefix}_object_slot_attn_cosine"] = object_slot_cosine_sum / object_slot_pair_total
        return metrics

    def _maybe_log_captionslot_reconstructions(
        self, model, eval_dataset=None, metric_key_prefix: str = "eval",
    ) -> None:
        if not self._wandb_enabled():
            return

        dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
        if dataset is None or len(dataset) == 0:
            return

        decoder = self._get_eval_decoder(model)
        if decoder is None:
            return

        try:
            import wandb
        except Exception:
            return

        max_items = min(16, len(dataset))
        if max_items == 0:
            return

        mean, std = self._get_image_stats(dataset, processor_index=1)
        loader = DataLoader(dataset, batch_size=4, shuffle=False, collate_fn=self.data_collator, num_workers=0)
        inner = model.module if hasattr(model, "module") else model
        device = inner.captionslot_cmd_embeddings.device
        was_training = inner.training
        images_to_log = []

        try:
            inner.eval()
            with torch.no_grad():
                for batch in loader:
                    images = batch["images"].to(device)
                    with self._get_autocast_context(device):
                        output = inner.generate_captionslot(
                            images=images,
                            caption_input_ids=batch["caption_input_ids"].to(device),
                            caption_attention_mask=batch["caption_attention_mask"].to(device),
                            ref_spans=batch["ref_spans"].to(device),
                            n_slots=batch["n_slots"].to(device),
                            guidance_level=1.0,
                        )
                        recon_images = self._decode_generated_images(
                            decoder, output.get("generated"), device=device,
                        )
                    source_images = self._denormalize_images(batch["target_images"], mean, std)

                    for idx in range(source_images.shape[0]):
                        if len(images_to_log) >= max_items:
                            break
                        comparison = self._make_image_comparison(source_images[idx], recon_images[idx])
                        images_to_log.append(
                            wandb.Image(
                                comparison.permute(1, 2, 0).numpy(),
                                caption=(
                                    f"{batch['dataset_names'][idx]}:{batch['image_ids'][idx]} | "
                                    "GT (left) | Recon (right)"
                                ),
                            )
                        )
                    if len(images_to_log) >= max_items:
                        break
        finally:
            inner.train(was_training)

        if images_to_log:
            wandb.log(
                {f"{metric_key_prefix}/reconstructions": images_to_log},
                step=int(getattr(self.state, "global_step", 0)),
            )

    def _maybe_log_captionslot_slot_attention(
        self, model, eval_dataset=None, metric_key_prefix: str = "eval",
    ) -> None:
        if not _env_flag("CAPTIONSLOT_LOG_SLOT_ATTENTION", default=True):
            return
        if not self._wandb_enabled():
            return

        dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
        if dataset is None or len(dataset) == 0:
            return

        try:
            import wandb
        except Exception:
            return

        global_step = int(getattr(self.state, "global_step", 0))
        log_every_steps = _env_int("CAPTIONSLOT_SLOT_ATTENTION_LOG_EVERY_STEPS", 5000)
        if log_every_steps > 0 and global_step > 0 and global_step % log_every_steps != 0:
            return

        max_rows = min(max(_env_int("CAPTIONSLOT_SLOT_ATTENTION_LOG_MAX_ROWS", 8), 0), len(dataset))
        if max_rows == 0:
            return

        mean, std = self._get_image_stats(dataset)
        loader = DataLoader(dataset, batch_size=4, shuffle=False, collate_fn=self.data_collator, num_workers=0)
        inner = model.module if hasattr(model, "module") else model
        device = inner.captionslot_cmd_embeddings.device
        decoder = self._get_eval_decoder(model)
        was_training = inner.training
        rows = []
        max_slot_cols = 0
        max_object_cols = 0
        slots_per_object = max(int(getattr(inner.config, "captionslot_slots_per_object", 1)), 1)

        try:
            inner.eval()
            with torch.no_grad():
                for batch in loader:
                    images = batch["images"].to(device)
                    with self._get_autocast_context(device):
                        output = inner.generate_captionslot(
                            images=images,
                            caption_input_ids=batch["caption_input_ids"].to(device),
                            caption_attention_mask=batch["caption_attention_mask"].to(device),
                            ref_spans=batch["ref_spans"].to(device),
                            n_slots=batch["n_slots"].to(device),
                            guidance_level=1.0,
                            return_generated=decoder is not None,
                        )
                    attn_maps = output.get("attn_maps")
                    if attn_maps is None or attn_maps.numel() == 0:
                        continue
                    pred_attn = attn_maps.detach().cpu().float()
                    source_images = self._denormalize_images(batch["images"], mean, std)
                    recon_images = None
                    if decoder is not None:
                        recon_images = self._decode_generated_images(
                            decoder,
                            output.get("generated"),
                            device=device,
                        )
                        if recon_images.numel() == 0:
                            recon_images = None

                    for sample_idx in range(images.shape[0]):
                        if len(rows) >= max_rows:
                            break
                        n_active = int(batch["n_slots"][sample_idx].item())
                        source = source_images[sample_idx]
                        object_texts = list(batch["object_texts"][sample_idx])
                        if len(object_texts) < n_active:
                            object_texts.extend(
                                self._decode_object_texts_from_spans(
                                    batch["caption_input_ids"][sample_idx],
                                    batch["ref_spans"][sample_idx],
                                    n_active - len(object_texts),
                                    start_slot=len(object_texts),
                                )
                            )
                        row = [
                            batch["dataset_names"][sample_idx],
                            batch["image_ids"][sample_idx],
                            int(batch["clean_num_objects"][sample_idx].item()),
                            int(batch["n_unique_objects"][sample_idx].item()),
                            n_active,
                            batch["caption_texts"][sample_idx],
                            wandb.Image(source.permute(1, 2, 0).numpy()),
                            None if recon_images is None else wandb.Image(recon_images[sample_idx].permute(1, 2, 0).numpy()),
                        ]
                        slot_limit = n_active
                        max_slot_cols = max(max_slot_cols, slot_limit)
                        for slot_idx in range(slot_limit):
                            gt_overlay = self._make_attention_overlay(
                                source,
                                batch["gt_masks_patches"][sample_idx, slot_idx].float(),
                                color=(0.15, 0.85, 0.25),
                            )
                            attn_overlay = self._make_attention_overlay(
                                source,
                                pred_attn[sample_idx, slot_idx],
                                color=(1.0, 0.15, 0.15),
                            )
                            row.extend([
                                object_texts[slot_idx] if slot_idx < len(object_texts) else "",
                                wandb.Image(gt_overlay.permute(1, 2, 0).numpy(), caption=f"slot_{slot_idx}_gt"),
                                wandb.Image(attn_overlay.permute(1, 2, 0).numpy(), caption=f"slot_{slot_idx}_attn"),
                            ])
                        if slots_per_object > 1:
                            object_limit = n_active // slots_per_object
                            max_object_cols = max(max_object_cols, object_limit)
                            for object_idx in range(object_limit):
                                pair_start = object_idx * slots_per_object
                                pair_end = min(pair_start + slots_per_object, n_active)
                                pair_text = object_texts[pair_start] if pair_start < len(object_texts) else ""
                                pair_gt_overlay = self._make_attention_overlay(
                                    source,
                                    batch["gt_masks_patches"][sample_idx, pair_start].float(),
                                    color=(0.15, 0.85, 0.25),
                                )
                                pair_attn_overlay = self._make_attention_overlay(
                                    source,
                                    pred_attn[sample_idx, pair_start:pair_end].amax(dim=0),
                                    color=(1.0, 0.15, 0.15),
                                )
                                row.extend([
                                    pair_text,
                                    wandb.Image(pair_gt_overlay.permute(1, 2, 0).numpy(), caption=f"object_{object_idx}_gt"),
                                    wandb.Image(pair_attn_overlay.permute(1, 2, 0).numpy(), caption=f"object_{object_idx}_attn_merged"),
                                ])
                        rows.append(row)
                    if len(rows) >= max_rows:
                        break
        finally:
            inner.train(was_training)

        if not rows:
            return

        columns = ["dataset", "image_id", "clean_num_objects", "n_unique_objects", "n_slots", "caption", "source", "recon"]
        for slot_idx in range(max_slot_cols):
            columns.extend([f"slot_{slot_idx}_text", f"slot_{slot_idx}_gt", f"slot_{slot_idx}_attn"])
        for object_idx in range(max_object_cols):
            columns.extend([f"object_{object_idx}_text", f"object_{object_idx}_gt", f"object_{object_idx}_attn_merged"])
        for row in rows:
            while len(row) < len(columns):
                row.append(None)
        table = wandb.Table(columns=columns, data=rows)
        table_name = f"{metric_key_prefix}/slot_attention"
        if _env_flag("CAPTIONSLOT_SLOT_ATTENTION_STEP_SUFFIX", default=True):
            table_name = f"{table_name}_step_{global_step}"
        wandb.log(
            {table_name: table},
            step=global_step,
        )

    def _wandb_enabled(self) -> bool:
        if not self.is_world_process_zero():
            return False
        report_to = getattr(self.args, "report_to", None)
        if report_to is None:
            return False
        targets = {report_to} if isinstance(report_to, str) else set(report_to)
        if "wandb" not in targets and "all" not in targets:
            return False
        try:
            import wandb
        except Exception:
            return False
        return wandb.run is not None

    def _get_image_stats(self, dataset, processor_index: int = 0):
        base = dataset
        while isinstance(base, torch.utils.data.Subset):
            base = base.dataset
        processors = base.data_args.image_processor_aux_list
        processor = processors[min(processor_index, len(processors) - 1)]
        mean = torch.tensor(processor.image_mean, dtype=torch.float32).view(1, 3, 1, 1)
        std = torch.tensor(processor.image_std, dtype=torch.float32).view(1, 3, 1, 1)
        return mean, std

    def _denormalize_images(self, images: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
        images = images.detach().cpu().float()
        images = torch.nan_to_num(images, nan=0.0, posinf=1.0, neginf=0.0)
        images = images * std + mean
        return images.clamp(0.0, 1.0)

    def _get_eval_decoder(self, model):
        decoder = getattr(self, "_eval_decoder", None)
        if decoder is not None:
            return decoder

        try:
            from huggingface_hub import hf_hub_download
            from scale_rae.model.multimodal_decoder import MultimodalDecoder
        except Exception as exc:
            logger.warning("[CaptionSlot] Decoder import failed: %s", exc)
            self._eval_decoder = None
            return None

        inner = model.module if hasattr(model, "module") else model
        repo_id = "nyu-visionx/siglip2_decoder"
        vision_towers = list(getattr(inner.config, "mm_vision_tower_aux_list", ["google/siglip2-so400m-patch14-224"]))
        encoder_path = vision_towers[1] if len(vision_towers) > 1 else vision_towers[0]
        encoder_path = encoder_path.split("-interp")[0]
        num_patches = int(
            getattr(
                inner.config,
                "diffusion_target_token_len",
                getattr(inner, "diffusion_target_token_len", 256),
            )
        )

        try:
            config_path = hf_hub_download(repo_id=repo_id, filename="config.json")
            ckpt_path = hf_hub_download(repo_id=repo_id, filename="model.pt")
            decoder = MultimodalDecoder(
                pretrained_encoder_path=encoder_path,
                general_decoder_config=config_path,
                num_patches=num_patches,
                drop_cls_token=True,
                decoder_path=ckpt_path,
            )
            self._eval_decoder = decoder
            return decoder
        except Exception as exc:
            logger.warning("[CaptionSlot] Failed to load eval decoder: %s", exc)
            self._eval_decoder = None
            return None

    def _get_autocast_context(self, device: torch.device):
        device_type = getattr(device, "type", str(device))
        if device_type != "cuda":
            return nullcontext()
        if bool(getattr(self.args, "bf16", False)):
            return torch.autocast(device_type=device_type, dtype=torch.bfloat16)
        if bool(getattr(self.args, "fp16", False)):
            return torch.autocast(device_type=device_type, dtype=torch.float16)
        return nullcontext()

    def _decode_generated_images(self, decoder, generated, device=None) -> torch.Tensor:
        if generated is None:
            return torch.empty(0, 3, 1, 1)
        decoder = decoder.to(device=device)
        decoder_dtype = next(decoder.parameters()).dtype
        if generated.dtype != decoder_dtype:
            generated = generated.to(dtype=decoder_dtype)
        if hasattr(decoder, "image_mean") and hasattr(decoder, "image_std"):
            decoder.image_mean = decoder.image_mean.to(device=device, dtype=decoder_dtype)
            decoder.image_std = decoder.image_std.to(device=device, dtype=decoder_dtype)
        empty_cls = torch.zeros(
            (generated.shape[0], 1, generated.shape[-1]),
            device=device, dtype=decoder_dtype,
        )
        image_features = torch.cat([empty_cls, generated], dim=1)
        recon = decoder(image_features)
        recon = torch.nan_to_num(recon, nan=0.0, posinf=1.0, neginf=0.0)
        return recon.clamp(0.0, 1.0).detach().cpu().float()

    def _make_image_comparison(self, source: torch.Tensor, recon: torch.Tensor) -> torch.Tensor:
        import torch.nn.functional as Fnn
        if source.shape != recon.shape:
            recon = Fnn.interpolate(recon.unsqueeze(0), size=source.shape[1:], mode="bilinear", align_corners=False)[0]
        return torch.cat([source, recon], dim=2)

    def _make_attention_overlay(self, source: torch.Tensor, attn: torch.Tensor, color=(1.0, 0.15, 0.15)) -> torch.Tensor:
        C, H, W = source.shape
        P = attn.shape[0]
        grid_side = int(P ** 0.5)
        if grid_side * grid_side != P:
            return source.clone()
        attn_2d = attn.view(grid_side, grid_side)
        import torch.nn.functional as Fnn
        attn_map = Fnn.interpolate(attn_2d.unsqueeze(0).unsqueeze(0), size=(H, W), mode="bilinear", align_corners=False)[0, 0]
        attn_map = attn_map.clamp(0.0, 1.0)
        color_t = torch.tensor(color, dtype=torch.float32).view(3, 1, 1)
        overlay = source * (1.0 - 0.6 * attn_map.unsqueeze(0)) + color_t * 0.6 * attn_map.unsqueeze(0)
        return overlay.clamp(0.0, 1.0)

    def _decode_object_texts_from_spans(
        self,
        caption_input_ids: torch.Tensor,
        ref_spans: torch.Tensor,
        max_count: int,
        start_slot: int = 0,
    ) -> List[str]:
        tokenizer = getattr(self, "tokenizer", None)
        if tokenizer is None or max_count <= 0:
            return []
        decoded: List[str] = []
        end_slot = min(ref_spans.shape[0], start_slot + max_count)
        for span in ref_spans[start_slot:end_slot]:
            start = int(span[0].item())
            end = int(span[1].item())
            if start < 0 or end <= start:
                decoded.append("")
                continue
            token_slice = caption_input_ids[start:end].tolist()
            decoded.append(tokenizer.decode(token_slice, skip_special_tokens=True).strip())
        return decoded

    def _normalize_patch_distribution(self, patch_map: torch.Tensor) -> torch.Tensor:
        patch_map = torch.nan_to_num(patch_map.detach().float(), nan=0.0, posinf=0.0, neginf=0.0)
        patch_map = patch_map.clamp_min(0.0)
        denom = patch_map.sum().clamp_min(1e-8)
        return patch_map / denom

    def _normalized_patch_entropy(self, patch_dist: torch.Tensor) -> float:
        if patch_dist.numel() == 0:
            return 0.0
        entropy = -(patch_dist * torch.log(patch_dist.clamp_min(1e-8))).sum()
        max_entropy = torch.log(torch.tensor(float(patch_dist.numel()), dtype=patch_dist.dtype))
        return float((entropy / max_entropy.clamp_min(1e-8)).item())


def freeze_for_captionslot_phase1(model: ScaleRAEQwenForCausalLM) -> ScaleRAEQwenForCausalLM:
    model.requires_grad_(False)
    stage = int(getattr(model.config, "captionslot_training_stage", 1))
    train_latent_queries = bool(getattr(model.config, "captionslot_train_latent_queries", False))
    control_mode = str(getattr(model.config, "captionslot_control_mode", "slots")).strip().lower()
    unfreeze_diff_head_body = bool(getattr(model.config, "captionslot_unfreeze_diff_head_body", False))
    llm_last_n = max(int(getattr(model.config, "captionslot_unfreeze_llm_last_n_layers", 0)), 0)
    llm_attn_only = bool(getattr(model.config, "captionslot_unfreeze_llm_attn_only", True))
    try:
        num_llm_layers = len(model.get_model().layers)
    except Exception:
        num_llm_layers = int(getattr(model.config, "num_hidden_layers", 0))
    llm_start_layer = max(num_llm_layers - llm_last_n, 0)

    n_trainable = 0
    trainable_names: List[str] = []
    for name, param in model.named_parameters():
        if unfreeze_diff_head_body:
            diff_head_trainable = "diff_head.model." in name
        else:
            diff_head_trainable = (
                _is_captionslot_diffusion_adaln_param(name)
                or _is_captionslot_sparse_xattn_param(name)
            )
        llm_trainable = _is_captionslot_llm_unfreeze_param(
            name=name,
            start_layer=llm_start_layer,
            num_layers=num_llm_layers if llm_last_n > 0 else 0,
            attn_only=llm_attn_only,
        )
        should_train = (
            ((control_mode != "caption_only") and (
                name.startswith("captionslot_cmd_embeddings")
                or name.startswith("captionslot_slot_embedding")
                or name.startswith("captionslot_reg_embeddings")
            ))
            or name.startswith("captionslot_context_projector")
            or name.startswith("diff_head_projector")
            or diff_head_trainable
            or llm_trainable
            or ((stage >= 2 or train_latent_queries) and "latent_queries" in name)
            or ("lora_" in name)  # LoRA adapters (lora_A / lora_B) are always trainable when injected
        )
        if should_train:
            param.requires_grad = True
            if param.is_floating_point() and param.dtype != torch.float32:
                param.data = param.data.to(torch.float32)
            n_trainable += param.numel()
            trainable_names.append(name)

    diff_head_desc = "full diff_head body" if unfreeze_diff_head_body else "diff_head AdaLN modulation, sparse slot/register cross-attention"
    llm_desc = (
        f", last {llm_last_n} LLM {'attn+input_norm' if llm_attn_only else 'blocks'}"
        if llm_last_n > 0
        else ""
    )
    n_lora_trainable = sum(
        param.numel() for name, param in model.named_parameters()
        if param.requires_grad and "lora_" in name
    )
    logger.info(
        "[CaptionSlot] trainable params: %d  (unfreeze_diff_head_body=%s, llm_last_n=%d, llm_attn_only=%s, lora=%d)",
        n_trainable,
        unfreeze_diff_head_body,
        llm_last_n,
        llm_attn_only,
        n_lora_trainable,
    )
    if control_mode == "caption_only":
        logger.info("[CaptionOnly] trainable groups: diff_head_projector, %s%s", diff_head_desc,
                    (", latent_queries" if (stage >= 2 or train_latent_queries) else "") + llm_desc)
    else:
        logger.info(
            "[CaptionSlot] trainable groups: cmd/slot/register embeddings, captionslot_context_projector, "
            "diff_head_projector, %s%s", diff_head_desc,
            (", latent_queries" if (stage >= 2 or train_latent_queries) else "") + llm_desc,
        )
    logger.debug("[CaptionSlot] trainable parameter names: %s", trainable_names)
    return model


def _load_checkpoint_tensor(checkpoint_dir: str, key: str) -> Optional[torch.Tensor]:
    if not checkpoint_dir or not os.path.isdir(checkpoint_dir):
        return None

    try:
        from safetensors.torch import load_file as load_safetensors_file
    except Exception:
        load_safetensors_file = None

    index_path = os.path.join(checkpoint_dir, "model.safetensors.index.json")
    if load_safetensors_file is not None and os.path.isfile(index_path):
        try:
            with open(index_path, "r") as f:
                weight_map = json.load(f).get("weight_map", {})
            shard_name = weight_map.get(key)
            if shard_name:
                shard = load_safetensors_file(os.path.join(checkpoint_dir, shard_name), device="cpu")
                return shard.get(key)
        except Exception as exc:
            logger.warning("[CaptionSlot] Failed reading %s from safetensors index: %s", key, exc)
            return None

    safetensors_path = os.path.join(checkpoint_dir, "model.safetensors")
    if load_safetensors_file is not None and os.path.isfile(safetensors_path):
        try:
            return load_safetensors_file(safetensors_path, device="cpu").get(key)
        except Exception as exc:
            logger.warning("[CaptionSlot] Failed reading %s from model.safetensors: %s", key, exc)
            return None

    bin_path = os.path.join(checkpoint_dir, "pytorch_model.bin")
    if os.path.isfile(bin_path):
        try:
            state = torch.load(bin_path, map_location="cpu")
            if isinstance(state, dict) and "state_dict" in state:
                state = state["state_dict"]
            return state.get(key) if isinstance(state, dict) else None
        except Exception as exc:
            logger.warning("[CaptionSlot] Failed reading %s from pytorch_model.bin: %s", key, exc)
            return None

    return None


def warm_start_resized_captionslot_parameters(
    model: ScaleRAEQwenForCausalLM,
    checkpoint_dir: str,
) -> None:
    """Copy overlapping rows for resized CaptionSlot embeddings skipped by HF load."""
    target = getattr(model, "captionslot_slot_embedding", None)
    if target is None:
        return

    source = _load_checkpoint_tensor(checkpoint_dir, "captionslot_slot_embedding")
    if source is None or tuple(source.shape) == tuple(target.shape):
        return
    if source.dim() != target.dim() or source.shape[1:] != target.shape[1:]:
        logger.warning(
            "[CaptionSlot] Cannot partial-load captionslot_slot_embedding: checkpoint=%s model=%s",
            tuple(source.shape),
            tuple(target.shape),
        )
        return

    rows = min(source.shape[0], target.shape[0])
    with torch.no_grad():
        target[:rows].copy_(source[:rows].to(device=target.device, dtype=target.dtype))
    logger.info(
        "[CaptionSlot] Warm-started captionslot_slot_embedding rows: copied %d/%d from %s",
        rows,
        target.shape[0],
        checkpoint_dir,
    )


def _log_captionslot_runtime_dtypes(model: ScaleRAEQwenForCausalLM) -> None:
    def _first_param_dtype(module):
        if module is None:
            return "none"
        try:
            return str(next(module.parameters()).dtype)
        except (StopIteration, AttributeError):
            return "none"

    vision_dtype = "none"
    try:
        vt_list = model.get_vision_tower_aux_list()
        if vt_list:
            vision_dtype = ",".join(
                f"{idx}:{_first_param_dtype(vt)}"
                for idx, vt in enumerate(vt_list)
            )
    except Exception:
        vision_dtype = "unavailable"

    logger.info(
        "[CaptionSlot] runtime dtypes: backbone=%s lm_head=%s vision=%s diff_head=%s diff_head_projector=%s cmd=%s",
        _first_param_dtype(getattr(model, "model", None)),
        _first_param_dtype(getattr(model, "lm_head", None)),
        vision_dtype,
        _first_param_dtype(getattr(model, "diff_head", None)),
        _first_param_dtype(getattr(model, "diff_head_projector", None)),
        str(getattr(model, "captionslot_cmd_embeddings", torch.empty((), dtype=torch.float32)).dtype),
    )


def _log_captionslot_runtime_devices(model: ScaleRAEQwenForCausalLM) -> None:
    def _first_param_device(module):
        if module is None:
            return "none"
        try:
            return str(next(module.parameters()).device)
        except (StopIteration, AttributeError):
            return "none"

    vision_device = "none"
    try:
        vt_list = model.get_vision_tower_aux_list()
        if vt_list:
            vision_device = ",".join(
                f"{idx}:{_first_param_device(vt)}"
                for idx, vt in enumerate(vt_list)
            )
    except Exception:
        vision_device = "unavailable"

    logger.info(
        "[CaptionSlot] runtime devices: backbone=%s lm_head=%s vision=%s diff_head=%s diff_head_projector=%s cmd=%s latent_queries=%s",
        _first_param_device(getattr(model, "model", None)),
        _first_param_device(getattr(model, "lm_head", None)),
        vision_device,
        _first_param_device(getattr(model, "diff_head", None)),
        _first_param_device(getattr(model, "diff_head_projector", None)),
        str(getattr(model, "captionslot_cmd_embeddings", torch.empty(0)).device),
        str(getattr(model.get_model(), "latent_queries", torch.empty(0)).device),
    )


def _register_im_start_end_token_ids(model, tokenizer) -> None:
    if DEFAULT_IM_START_TOKEN not in tokenizer.get_vocab():
        return

    im_start_id = tokenizer.convert_tokens_to_ids(DEFAULT_IM_START_TOKEN)
    im_end_id = tokenizer.convert_tokens_to_ids(DEFAULT_IM_END_TOKEN)
    model.im_start_id = im_start_id
    model.im_end_id = im_end_id
    model.config.im_start_id = im_start_id
    model.config.im_end_id = im_end_id


def _register_captionslot_template_token_ids(model, tokenizer) -> None:
    block_map = {
        "captionslot_system_prefix_ids": "<|im_start|>system\nYou are a helpful assistant.",
        "captionslot_system_suffix_ids": "<|im_end|>\n",
        "captionslot_user_prefix_ids": "<|im_start|>user\n",
        "captionslot_user_text_prefix_ids": "",
        "captionslot_user_suffix_ids": "<|im_end|>\n",
        "captionslot_assistant_prefix_ids": "<|im_start|>assistant\n",
        "captionslot_assistant_suffix_ids": "<|im_end|>",
    }
    for attr_name, text in block_map.items():
        token_ids = tokenizer.encode(text, add_special_tokens=False)
        setattr(model, attr_name, token_ids)
        setattr(model.config, attr_name, token_ids)


def make_data_module(tokenizer, data_args, model_config, training_args=None) -> Dict:
    dataset_names = _split_csv(data_args.captionslot_datasets)
    if not dataset_names:
        raise ValueError("captionslot_datasets must contain at least one dataset name.")

    train_splits = _split_csv(data_args.captionslot_train_splits)
    eval_splits = _split_csv(data_args.captionslot_eval_splits)
    if not train_splits:
        train_splits = ["train"]
    if not eval_splits:
        eval_splits = ["val"]

    train_dataset = RefCOCOGroupedDataset(
        coco_root=data_args.captionslot_coco_root,
        dataset_names=dataset_names,
        split_names=train_splits,
        tokenizer=tokenizer,
        data_args=data_args,
        model_config=model_config,
    )

    try:
        eval_dataset = RefCOCOGroupedDataset(
            coco_root=data_args.captionslot_coco_root,
            dataset_names=dataset_names,
            split_names=eval_splits,
            tokenizer=tokenizer,
            data_args=data_args,
            model_config=model_config,
        )
    except ValueError:
        eval_dataset = None

    eval_size = min(len(train_dataset), int(getattr(training_args, "captionslot_eval_num_images", 128)))
    if eval_dataset is None or len(eval_dataset) == 0:
        eval_dataset = torch.utils.data.Subset(train_dataset, list(range(eval_size)))
    elif len(eval_dataset) > eval_size:
        eval_dataset = torch.utils.data.Subset(eval_dataset, list(range(eval_size)))

    return {
        "train_dataset": train_dataset,
        "eval_dataset": eval_dataset,
        "data_collator": CaptionSlotDataCollator(pad_token_id=tokenizer.pad_token_id),
    }


def train():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    logger.info(
        "[CaptionSlot] train() start | model=%s | output_dir=%s | local_rank=%s",
        model_args.model_name_or_path,
        training_args.output_dir,
        getattr(training_args, "local_rank", -1),
    )

    compute_dtype = (
        torch.float16 if training_args.fp16
        else (torch.bfloat16 if training_args.bf16 else torch.float32)
    )

    if training_args.gradient_checkpointing:
        logger.warning("[CaptionSlot] Disabling gradient checkpointing for the custom attention-bias path.")
        training_args.gradient_checkpointing = False

    from transformers import AutoConfig

    parsed_vision_towers = None
    parsed_token_lens = None
    if model_args.vision_tower_aux_list is not None:
        parsed_vision_towers = json.loads(model_args.vision_tower_aux_list)
        parsed_token_lens = json.loads(model_args.vision_tower_aux_token_len_list)

    config = AutoConfig.from_pretrained(model_args.model_name_or_path)
    config.vision_loss = model_args.vision_loss
    config.vision_loss_mode = model_args.vision_loss_mode
    config.vision_coef = model_args.vision_coef
    config.diffusion_model_hidden_size = model_args.diffusion_model_hidden_size
    config.diffusion_model_channels = model_args.diffusion_model_channels
    config.diffusion_model_depth = model_args.diffusion_model_depth
    config.diffusion_model_heads = model_args.diffusion_model_heads
    config.diffusion_model_z_channels = model_args.diffusion_model_z_channels
    config.dit_cls = model_args.dit_cls
    config.use_aurora = False
    config.use_captionslot = model_args.use_captionslot
    if parsed_vision_towers is not None:
        config.mm_vision_tower_aux_list = parsed_vision_towers
        config.mm_vision_tower_aux_token_len_list = parsed_token_lens
        config.vision_tower_aux_token_len_list = parsed_token_lens
    if parsed_token_lens:
        config.image_feature_token_len = int(
            model_args.image_feature_token_len
            if model_args.image_feature_token_len is not None
            else parsed_token_lens[0]
        )
        config.diffusion_target_token_len = int(
            model_args.diffusion_target_token_len
            if model_args.diffusion_target_token_len is not None
            else parsed_token_lens[-1]
        )
    else:
        config.image_feature_token_len = int(
            model_args.image_feature_token_len
            if model_args.image_feature_token_len is not None
            else getattr(config, "image_feature_token_len", 256)
        )
        config.diffusion_target_token_len = int(
            model_args.diffusion_target_token_len
            if model_args.diffusion_target_token_len is not None
            else getattr(config, "diffusion_target_token_len", config.image_feature_token_len)
        )
    config.captionslot_max_slots = model_args.captionslot_max_slots
    config.captionslot_slots_per_object = model_args.captionslot_slots_per_object
    config.captionslot_n_register = model_args.captionslot_n_register
    config.captionslot_cmd_length = model_args.captionslot_cmd_length
    config.captionslot_recon_loss_weight = model_args.captionslot_recon_loss_weight
    config.captionslot_mask_bce_loss_weight = model_args.captionslot_mask_bce_loss_weight
    tversky_loss_weight = (
        model_args.captionslot_mask_tversky_loss_weight
        if model_args.captionslot_mask_tversky_loss_weight is not None
        else model_args.captionslot_mask_dice_loss_weight
    )
    config.captionslot_mask_tversky_loss_weight = tversky_loss_weight
    config.captionslot_mask_dice_loss_weight = tversky_loss_weight
    config.captionslot_mask_balanced_bce = model_args.captionslot_mask_balanced_bce
    config.captionslot_mask_tversky_alpha = model_args.captionslot_mask_tversky_alpha
    config.captionslot_mask_tversky_beta = model_args.captionslot_mask_tversky_beta
    config.captionslot_object_cam_loss_weight = model_args.captionslot_object_cam_loss_weight
    config.captionslot_register_cam_loss_weight = model_args.captionslot_register_cam_loss_weight
    config.captionslot_cam_layers = model_args.captionslot_cam_layers
    config.captionslot_cam_eps = model_args.captionslot_cam_eps
    config.captionslot_caption_loss_weight = model_args.captionslot_caption_loss_weight
    config.captionslot_diversity_loss_weight = model_args.captionslot_diversity_loss_weight
    config.captionslot_training_stage = model_args.captionslot_training_stage
    config.captionslot_condition_gate_init = model_args.captionslot_condition_gate_init
    config.captionslot_train_latent_queries = model_args.captionslot_train_latent_queries
    config.captionslot_unfreeze_diff_head_body = model_args.captionslot_unfreeze_diff_head_body
    config.captionslot_unfreeze_llm_last_n_layers = model_args.captionslot_unfreeze_llm_last_n_layers
    config.captionslot_unfreeze_llm_attn_only = model_args.captionslot_unfreeze_llm_attn_only
    config.captionslot_attention_use_layer_norm = model_args.captionslot_attention_use_layer_norm
    config.captionslot_attention_temperature = model_args.captionslot_attention_temperature
    config.captionslot_prior_bias_scale = model_args.captionslot_prior_bias_scale
    config.captionslot_control_mode = model_args.captionslot_control_mode
    config.captionslot_add_cross_attention = model_args.captionslot_add_cross_attention
    config.captionslot_cross_attention_start_block = model_args.captionslot_cross_attention_start_block
    config.captionslot_cross_attention_every_n_blocks = model_args.captionslot_cross_attention_every_n_blocks
    config.captionslot_cross_attention_include_registers = model_args.captionslot_cross_attention_include_registers
    config.captionslot_cross_attention_gate_init = model_args.captionslot_cross_attention_gate_init
    if model_args.diffusion_norm_stats_path:
        config.diffusion_norm_stats_path = model_args.diffusion_norm_stats_path

    model = ScaleRAEQwenForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        torch_dtype=compute_dtype,
        ignore_mismatched_sizes=True,
    )
    model.config.use_cache = False
    warm_start_resized_captionslot_parameters(model, model_args.model_name_or_path)
    logger.info("[CaptionSlot] model checkpoint loaded")

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )
    tokenizer.pad_token = "<|endoftext|>"
    tokenizer.pad_token_id = 151643

    if model_args.vision_tower_aux_list is not None:
        model_args.vision_tower_aux_list = json.loads(model_args.vision_tower_aux_list)
        model_args.vision_tower_aux_token_len_list = json.loads(model_args.vision_tower_aux_token_len_list)
        model_args.unfreeze_mm_vision_tower = training_args.unfreeze_mm_vision_tower

        model.get_model().initialize_vision_modules(model_args=model_args, fsdp=training_args.fsdp)
        model.load_vision_head(model_args=model_args)
        logger.info("[CaptionSlot] vision tower + diffusion head initialized")

        vision_tower_aux_list = model.get_vision_tower_aux_list()
        if not training_args.unfreeze_mm_vision_tower:
            for vt in vision_tower_aux_list:
                vt.to(dtype=compute_dtype, device=training_args.device)
        logger.info("[CaptionSlot] vision modules moved to target device=%s", training_args.device)
        logger.info(
            "[CaptionSlot] vision roles: encoder_a=%s tokens=%s -> LLM, encoder_b=%s tokens=%s -> diffusion target",
            model_args.vision_tower_aux_list[0],
            config.image_feature_token_len,
            model_args.vision_tower_aux_list[1] if len(model_args.vision_tower_aux_list) > 1 else model_args.vision_tower_aux_list[0],
            config.diffusion_target_token_len,
        )

        data_args.image_processor_aux_list = [vt.image_processor for vt in vision_tower_aux_list]
        data_args.is_multimodal = True
        data_args.vision_tower_aux_token_len_list = model_args.vision_tower_aux_token_len_list
        model.config.image_aspect_ratio = data_args.image_aspect_ratio
        model.config.vision_tower_aux_token_len_list = model_args.vision_tower_aux_token_len_list
        model.config.image_feature_token_len = config.image_feature_token_len
        model.config.diffusion_target_token_len = config.diffusion_target_token_len
        model.config.si_token_len = model_args.si_token_len
        model.config.miv_token_len = model_args.miv_token_len

    # Optional: inject LoRA adapters into the LLM attention modules BEFORE freezing,
    # so freeze_for_captionslot_phase1 can correctly handle the new lora_* params.
    if bool(getattr(training_args, "captionslot_lora_enable", False)):
        try:
            from peft import LoraConfig, inject_adapter_in_model
        except ImportError as exc:  # pragma: no cover - peft is required only with LoRA
            raise ImportError(
                "peft >= 0.6 is required for captionslot_lora_enable=True (install with `pip install peft>=0.7`)."
            ) from exc

        lora_targets = [
            tok.strip()
            for tok in str(training_args.captionslot_lora_target_modules).split(",")
            if tok.strip()
        ]
        if not lora_targets:
            lora_targets = ["q_proj", "k_proj", "v_proj", "o_proj"]

        lora_config = LoraConfig(
            r=int(training_args.captionslot_lora_r),
            lora_alpha=int(training_args.captionslot_lora_alpha),
            lora_dropout=float(training_args.captionslot_lora_dropout),
            bias="none",
            target_modules=lora_targets,
            task_type="CAUSAL_LM",
        )
        inject_adapter_in_model(lora_config, model, adapter_name="default")
        n_lora_params = sum(
            p.numel() for n, p in model.named_parameters() if "lora_" in n
        )
        n_lora_modules = sum(
            1 for n, _ in model.named_modules() if n.endswith("lora_A.default") or n.endswith("lora_B.default")
        )
        logger.info(
            "[CaptionSlot/LoRA] injected | r=%d alpha=%d dropout=%.3f targets=%s | "
            "lora_params=%d | lora_modules=%d",
            int(training_args.captionslot_lora_r),
            int(training_args.captionslot_lora_alpha),
            float(training_args.captionslot_lora_dropout),
            ",".join(lora_targets),
            n_lora_params,
            n_lora_modules,
        )
    else:
        logger.info("[CaptionSlot/LoRA] disabled (captionslot_lora_enable=False)")

    model = freeze_for_captionslot_phase1(model)
    model.to(training_args.device)
    if model_args.vision_tower_aux_list is not None:
        for vt in model.get_vision_tower_aux_list():
            vt.to(dtype=compute_dtype, device=training_args.device)
    _log_captionslot_runtime_dtypes(model)
    _log_captionslot_runtime_devices(model)
    model.initialize_vision_tokenizer(model_args, tokenizer=tokenizer)
    _register_im_start_end_token_ids(model, tokenizer)
    _register_captionslot_template_token_ids(model, tokenizer)

    data_module = make_data_module(tokenizer, data_args, model.config, training_args=training_args)
    logger.info(
        "[CaptionSlot] datasets ready | train=%d | eval=%d",
        len(data_module["train_dataset"]),
        len(data_module["eval_dataset"]),
    )
    trainer = CaptionSlotTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        **data_module,
    )
    logger.info("[CaptionSlot] trainer initialized; starting train()")

    resume = training_args.resume_from_checkpoint or None
    trainer.train(resume_from_checkpoint=resume)
    skip_final_save = os.environ.get("CAPTIONSLOT_SKIP_FINAL_SAVE", "").strip().lower() in {"1", "true", "yes"}
    if skip_final_save:
        logger.info("[CaptionSlot] Skipping final save because CAPTIONSLOT_SKIP_FINAL_SAVE is enabled.")
    else:
        trainer.save_state()
        trainer.save_model(training_args.output_dir)
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    train()
