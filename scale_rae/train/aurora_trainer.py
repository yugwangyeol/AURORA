"""
AURORA trainer with mask supervision and optional inpainting batches.
"""

import json
import logging
import math
import os
import re
import random
from bisect import bisect_right
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F
import transformers
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Sampler
from transformers import Trainer
from transformers.trainer import ALL_LAYERNORM_LAYERS, get_parameter_names, is_sagemaker_mp_enabled
from transformers.training_args import ParallelMode
from transformers.trainer_utils import seed_worker
from transformers.utils import is_torch_npu_available, is_torch_tpu_available

from scale_rae.model.language_model.scale_rae_qwen2 import ScaleRAEQwenForCausalLM

logger_module = logging.getLogger(__name__)

if is_torch_tpu_available(check_device=False):
    import torch_xla.core.xla_model as xm


def _is_aurora_diffusion_condition_param(name: str) -> bool:
    return (
        name.startswith("diff_head.model.y_embedder")
        or ("diff_head.model." in name and "adaLN_modulation.1" in name)
    )


def _get_mask_utils():
    try:
        import pycocotools.mask as mask_utils
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "pycocotools is required for COCO mask decoding in AURORA training."
        ) from exc
    return mask_utils


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    version: Optional[str] = field(default="v0")
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

    use_aurora: bool = field(default=True)
    aurora_max_slots: int = field(default=10)
    aurora_n_register: int = field(default=8)
    aurora_cmd_length: int = field(default=8)
    aurora_mask_loss_weight: float = field(default=1.0)
    aurora_diversity_loss_weight: float = field(default=0.1)
    aurora_inpaint_weight: float = field(default=0.5)
    aurora_fail_on_nan: bool = field(default=True)
    aurora_training_stage: int = field(default=1)
    aurora_grad_clip_max_norm: float = field(default=1.0)
    aurora_train_diffusion_condition: bool = field(default=True)
    diffusion_norm_stats_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to .pt file with {'running_mean': Tensor[D], 'running_var': Tensor[D]} for diffusion target normalization."},
    )


@dataclass
class DataArguments:
    data_path: str = field(default=None)
    image_folder: Optional[str] = field(default=None)
    aurora_reconstruction_image_folder: Optional[str] = field(default=None)
    aurora_inpaint_data_path: Optional[str] = field(default=None)
    aurora_inpaint_image_folder: Optional[str] = field(default=None)
    aurora_include_reconstruction: bool = field(default=True)
    aurora_include_inpainting: bool = field(default=True)
    image_aspect_ratio: str = field(default="square")
    max_images_per_sample: int = field(default=1)
    coco_annotation_path: Optional[str] = field(default=None)
    aurora_min_area: float = field(default=1024.0)

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
    aurora_inpaint_warmup_steps: int = field(default=1000)
    aurora_inpaint_ramp_steps: int = field(default=4000)
    aurora_eval_num_images: int = field(default=100)
    aurora_eval_log_image_count: int = field(default=100)
    aurora_eval_visual_batch_size: int = field(default=4)
    aurora_eval_log_reconstructions: bool = field(default=True)
    aurora_eval_log_attention_overlays: bool = field(default=True)
    aurora_eval_attention_overlay_count: int = field(default=50)
    aurora_eval_decoder_repo: str = field(default="nyu-visionx/siglip2_decoder")


@dataclass
class COCOAnnotationIndex:
    anns_by_image: Dict[str, List[dict]]
    image_info_by_id: Dict[str, dict]
    ordered_image_ids: List[str]


def load_coco_annotation_index(
    annotation_path: Optional[str],
    min_area: float,
) -> Optional[COCOAnnotationIndex]:
    if annotation_path is None or not os.path.exists(annotation_path):
        logger_module.warning("[AURORA] COCO annotation not found: %s", annotation_path)
        return None

    with open(annotation_path, "r") as f:
        coco = json.load(f)

    anns_by_image: Dict[str, List[dict]] = defaultdict(list)
    for ann in coco.get("annotations", []):
        if ann.get("iscrowd", 0):
            continue
        if ann.get("area", 0.0) < min_area:
            continue
        anns_by_image[str(ann["image_id"])].append(ann)

    for image_id in list(anns_by_image.keys()):
        anns_by_image[image_id].sort(key=lambda x: x.get("area", 0.0), reverse=True)

    image_info_by_id = {
        str(image["id"]): image
        for image in coco.get("images", [])
    }
    ordered_image_ids = [
        image_id for image_id in image_info_by_id.keys() if image_id in anns_by_image
    ]

    logger_module.info(
        "[AURORA] Loaded %d annotated images from %s",
        len(ordered_image_ids),
        annotation_path,
    )
    return COCOAnnotationIndex(
        anns_by_image=anns_by_image,
        image_info_by_id=image_info_by_id,
        ordered_image_ids=ordered_image_ids,
    )


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


def mask_to_patch_mask(mask: torch.Tensor, grid_size: int) -> torch.Tensor:
    mask = mask.float().unsqueeze(0).unsqueeze(0)
    patch = F.interpolate(mask, size=(grid_size, grid_size), mode="area")
    return patch.squeeze(0).squeeze(0).reshape(-1).clamp(0.0, 1.0)


def normalize_image_id(image_id: str) -> str:
    image_id = str(image_id)
    digits = re.sub(r"\D", "", image_id)
    if not digits:
        return image_id
    normalized = digits.lstrip("0")
    return normalized if normalized else "0"


def build_gt_patch_masks(
    annotation_index: Optional[COCOAnnotationIndex],
    image_id: str,
    image_size,
    max_objects: int,
    grid_size: int,
) -> torch.Tensor:
    patch_masks = torch.zeros(max_objects, grid_size * grid_size, dtype=torch.float32)
    if annotation_index is None:
        return patch_masks

    anns = annotation_index.anns_by_image.get(image_id, [])
    width, height = image_size
    for idx, ann in enumerate(anns[:max_objects]):
        mask_np = decode_segmentation(ann["segmentation"], height=height, width=width)
        patch_masks[idx] = mask_to_patch_mask(torch.tensor(mask_np), grid_size=grid_size)
    return patch_masks


def load_mask_image(mask_path: str, grid_size: int) -> torch.Tensor:
    mask = Image.open(mask_path).convert("L")
    mask_np = (np.array(mask, dtype=np.float32) > 127).astype(np.float32)
    return mask_to_patch_mask(torch.tensor(mask_np), grid_size=grid_size)


class AURORADataset(Dataset):
    def __init__(
        self,
        data_path: str,
        data_args: DataArguments,
        model_configs=None,
        annotation_index: Optional[COCOAnnotationIndex] = None,
        image_folder: Optional[str] = None,
    ):
        super().__init__()
        self.data_path = data_path
        self.data_args = data_args
        self.model_configs = model_configs
        self.image_folder = image_folder if image_folder is not None else data_args.image_folder
        self.annotation_index = annotation_index
        self.samples = None
        self._build_offset_index()

    def _build_offset_index(self):
        with open(self.data_path, "r", encoding="utf-8") as f:
            while True:
                ch = f.read(1)
                if not ch:
                    break
                if not ch.isspace():
                    if ch == "[":
                        f.seek(0)
                        self.samples = json.load(f)
                        self.length = len(self.samples)
                        logger_module.info(
                            "[AURORA] Loaded JSON-array manifest with %d samples from %s",
                            self.length,
                            self.data_path,
                        )
                        return
                    break

        self.offsets = []
        with open(self.data_path, "rb") as f:
            off = 0
            for raw_line in f:
                if raw_line.strip():
                    self.offsets.append(off)
                off += len(raw_line)
        self.length = len(self.offsets)

    def __len__(self):
        return self.length

    def _parse_image_id(self, dat: dict, image_path: str) -> str:
        if "base_image_id" in dat:
            return normalize_image_id(dat["base_image_id"])
        if "image_id" in dat:
            return normalize_image_id(dat["image_id"])
        basename = os.path.splitext(os.path.basename(image_path))[0]
        digits = re.sub(r"^0+", "", re.sub(r"\D", "", basename))
        return digits if digits else basename

    def __getitem__(self, idx):
        max_retries = min(max(len(self), 1), 8)
        last_exc = None
        current_idx = idx
        for attempt in range(max_retries):
            try:
                return self._getitem(current_idx)
            except Exception as exc:
                last_exc = exc
                logger_module.warning(
                    "[AURORA] Failed to load index %d on attempt %d/%d: %s",
                    current_idx,
                    attempt + 1,
                    max_retries,
                    exc,
                )
                if attempt + 1 >= max_retries:
                    break
                current_idx = random.randint(0, len(self) - 1)
        raise RuntimeError(
            f"[AURORA] Exhausted retries while loading sample starting from index {idx}"
        ) from last_exc

    def _load_json(self, idx: int) -> dict:
        if self.samples is not None:
            return self.samples[idx]
        with open(self.data_path, "rb") as f:
            f.seek(self.offsets[idx])
            line = f.readline()
        return json.loads(line)

    def _resolve_path(self, image_file: str) -> str:
        image_folder = self.image_folder or ""
        return image_file if os.path.isabs(image_file) else os.path.join(image_folder, image_file)

    def _build_gt_patch_masks(self, image_id: str, image_size) -> torch.Tensor:
        max_objects = max(getattr(self.model_configs, "aurora_max_slots", 10) - 1, 0)
        grid_size = int(self.data_args.vision_tower_aux_token_len_list[0] ** 0.5)
        return build_gt_patch_masks(
            annotation_index=self.annotation_index,
            image_id=image_id,
            image_size=image_size,
            max_objects=max_objects,
            grid_size=grid_size,
        )

    def _getitem(self, idx: int) -> Dict[str, torch.Tensor]:
        dat = self._load_json(idx)
        image_file = dat.get("image", dat.get("source_path", ""))
        if not image_file:
            raise ValueError(f"Sample {idx} is missing an image path: {dat}")

        full_path = self._resolve_path(image_file)
        if not os.path.exists(full_path):
            raise FileNotFoundError(full_path)

        image = Image.open(full_path).convert("RGB")
        image_id = self._parse_image_id(dat, image_file)
        gt_masks_patches = self._build_gt_patch_masks(image_id, image.size)
        anns_by_image = self.annotation_index.anns_by_image if self.annotation_index is not None else {}
        n_objects = min(
            len(anns_by_image.get(image_id, [])),
            max(getattr(self.model_configs, "aurora_max_slots", 10) - 1, 0),
        )

        processor = self.data_args.image_processor_aux_list[0]
        sample = {
            "image": processor.preprocess(image, return_tensors="pt")["pixel_values"][0],
            "n_objects": torch.tensor(n_objects, dtype=torch.long),
            "gt_masks_patches": gt_masks_patches,
            "image_id": image_id,
        }

        target_path = dat.get("target_path")
        mask_path = dat.get("mask_path")
        if target_path and mask_path:
            full_target = self._resolve_path(target_path)
            full_mask = self._resolve_path(mask_path)
            if os.path.exists(full_target) and os.path.exists(full_mask):
                target_image = Image.open(full_target).convert("RGB")
                grid_size = int(self.data_args.vision_tower_aux_token_len_list[0] ** 0.5)
                inpaint_mask_patches = load_mask_image(full_mask, grid_size=grid_size)
                sample["target_image"] = processor.preprocess(
                    target_image, return_tensors="pt"
                )["pixel_values"][0]
                sample["inpaint_mask_patches"] = inpaint_mask_patches

                # Fallback for inpainting-only manifests without COCO instance annotations:
                # use the provided removed-object mask as the first supervision slot so the
                # AR slot path and inpainting loss can still be exercised end-to-end.
                if n_objects == 0 and gt_masks_patches.shape[0] > 0:
                    gt_masks_patches[0] = inpaint_mask_patches
                    n_objects = 1
                    sample["n_objects"] = torch.tensor(n_objects, dtype=torch.long)
                    sample["gt_masks_patches"] = gt_masks_patches

        return sample


class COCOReconstructionDataset(Dataset):
    def __init__(
        self,
        image_root: str,
        data_args: DataArguments,
        model_configs=None,
        annotation_index: Optional[COCOAnnotationIndex] = None,
    ):
        super().__init__()
        self.image_root = image_root
        self.data_args = data_args
        self.model_configs = model_configs
        self.annotation_index = annotation_index
        self.image_ids = annotation_index.ordered_image_ids if annotation_index is not None else []

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        max_retries = min(max(len(self), 1), 8)
        last_exc = None
        current_idx = idx
        for attempt in range(max_retries):
            try:
                return self._getitem(current_idx)
            except Exception as exc:
                last_exc = exc
                logger_module.warning(
                    "[AURORA] Failed to load reconstruction index %d on attempt %d/%d: %s",
                    current_idx,
                    attempt + 1,
                    max_retries,
                    exc,
                )
                if attempt + 1 >= max_retries:
                    break
                current_idx = random.randint(0, len(self) - 1)
        raise RuntimeError(
            f"[AURORA] Exhausted retries while loading reconstruction sample starting from index {idx}"
        ) from last_exc

    def _getitem(self, idx: int) -> Dict[str, torch.Tensor]:
        image_id = self.image_ids[idx]
        image_info = self.annotation_index.image_info_by_id[image_id]
        image_path = os.path.join(self.image_root, image_info["file_name"])
        if not os.path.exists(image_path):
            raise FileNotFoundError(image_path)

        image = Image.open(image_path).convert("RGB")
        max_objects = max(getattr(self.model_configs, "aurora_max_slots", 10) - 1, 0)
        grid_size = int(self.data_args.vision_tower_aux_token_len_list[0] ** 0.5)
        gt_masks_patches = build_gt_patch_masks(
            annotation_index=self.annotation_index,
            image_id=image_id,
            image_size=image.size,
            max_objects=max_objects,
            grid_size=grid_size,
        )
        n_objects = min(
            len(self.annotation_index.anns_by_image.get(image_id, [])),
            max_objects,
        )

        processor = self.data_args.image_processor_aux_list[0]
        sample = {
            "image": processor.preprocess(image, return_tensors="pt")["pixel_values"][0],
            "n_objects": torch.tensor(n_objects, dtype=torch.long),
            "gt_masks_patches": gt_masks_patches,
            "image_id": image_id,
        }

        return sample


class AURORAMixedDataset(Dataset):
    def __init__(
        self,
        datasets: Sequence[Dataset],
        task_names: Sequence[str],
        data_args: DataArguments,
    ):
        super().__init__()
        if len(datasets) != len(task_names):
            raise ValueError(
                f"AURORAMixedDataset expected matching datasets/task_names, got "
                f"{len(datasets)} datasets and {len(task_names)} task names."
            )
        self.datasets = list(datasets)
        self.task_names = list(task_names)
        self.data_args = data_args
        self.cumulative_sizes = []
        self.task_ranges: Dict[str, tuple[int, int]] = {}
        total = 0
        for task_name, dataset in zip(self.task_names, self.datasets):
            start = total
            total += len(dataset)
            self.cumulative_sizes.append(total)
            self.task_ranges[task_name] = (start, total)

    def __len__(self):
        return self.cumulative_sizes[-1] if self.cumulative_sizes else 0

    def __getitem__(self, idx):
        dataset_idx = bisect_right(self.cumulative_sizes, idx)
        prev_cum = 0 if dataset_idx == 0 else self.cumulative_sizes[dataset_idx - 1]
        sample_idx = idx - prev_cum
        return self.datasets[dataset_idx][sample_idx]

    def has_task(self, task_name: str) -> bool:
        return self.get_task_length(task_name) > 0

    def get_task_length(self, task_name: str) -> int:
        task_range = self.task_ranges.get(task_name)
        if task_range is None:
            return 0
        start, end = task_range
        return max(end - start, 0)

    def get_task_range(self, task_name: str) -> Optional[tuple[int, int]]:
        return self.task_ranges.get(task_name)


class AlternatingTaskBatchSampler(Sampler[List[int]]):
    def __init__(
        self,
        dataset: AURORAMixedDataset,
        batch_size: int,
        warmup_steps: int,
        num_processes: int,
        drop_last: bool,
        seed: int,
        step_provider,
    ):
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")

        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.warmup_steps = max(int(warmup_steps), 0)
        self.num_processes = max(int(num_processes), 1)
        self.drop_last = bool(drop_last)
        self.seed = int(seed)
        self.step_provider = step_provider
        self._iteration_index = 0
        # Required by accelerate's prepare_data_loader
        self.sampler = None

        global_batch = self.batch_size * self.num_processes
        total_samples = max(len(dataset), 1)
        if self.drop_last:
            self.steps_per_epoch = max(total_samples // global_batch, 1)
        else:
            self.steps_per_epoch = max(math.ceil(total_samples / float(global_batch)), 1)

    def __len__(self):
        return self.steps_per_epoch * self.num_processes

    def _shuffle_task_indices(
        self,
        task_name: str,
        generator: torch.Generator,
    ) -> List[int]:
        task_range = self.dataset.get_task_range(task_name)
        if task_range is None:
            return []
        start, end = task_range
        length = max(end - start, 0)
        if length == 0:
            return []
        order = torch.randperm(length, generator=generator).tolist()
        return [start + idx for idx in order]

    def _next_task_batch(
        self,
        task_name: str,
        task_state: Dict[str, Dict[str, object]],
        generator: torch.Generator,
    ) -> List[int]:
        state = task_state[task_name]
        batch: List[int] = []
        while len(batch) < self.batch_size:
            order = state["order"]
            pos = int(state["pos"])
            if pos >= len(order):
                state["order"] = self._shuffle_task_indices(task_name, generator)
                order = state["order"]
                pos = 0
                state["pos"] = 0
            if not order:
                raise ValueError(f"No samples are available for task '{task_name}'.")
            take_end = min(pos + (self.batch_size - len(batch)), len(order))
            batch.extend(order[pos:take_end])
            state["pos"] = take_end
        return batch

    def _task_for_global_step(self, global_step: int) -> str:
        if not self.dataset.has_task("inpainting"):
            return "reconstruction"
        if global_step < self.warmup_steps:
            return "reconstruction"
        return "reconstruction" if (global_step % 2 == 0) else "inpainting"

    def __iter__(self):
        generator = torch.Generator()
        generator.manual_seed(self.seed + self._iteration_index)
        self._iteration_index += 1

        task_state = {}
        for task_name in self.dataset.task_names:
            task_state[task_name] = {
                "order": self._shuffle_task_indices(task_name, generator),
                "pos": 0,
            }

        start_step = int(self.step_provider())
        fallback_task = "reconstruction" if self.dataset.has_task("reconstruction") else self.dataset.task_names[0]

        for local_step in range(self.steps_per_epoch):
            global_step = start_step + local_step
            task_name = self._task_for_global_step(global_step)
            if not self.dataset.has_task(task_name):
                task_name = fallback_task
            for _ in range(self.num_processes):
                yield self._next_task_batch(task_name, task_state, generator)


@dataclass
class AURORADataCollator:
    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        batch = {
            "images": torch.stack([inst["image"] for inst in instances]),
            "n_objects": torch.stack([inst["n_objects"] for inst in instances]),
            "gt_masks_patches": torch.stack([inst["gt_masks_patches"] for inst in instances]),
            "image_ids": [inst["image_id"] for inst in instances],
        }

        has_inpaint = torch.tensor(
            [
                ("target_image" in inst) and ("inpaint_mask_patches" in inst)
                for inst in instances
            ],
            dtype=torch.bool,
        )
        batch["has_inpaint"] = has_inpaint
        if has_inpaint.any():
            zero_image = torch.zeros_like(instances[0]["image"])
            zero_mask = torch.zeros_like(instances[0]["gt_masks_patches"][0])
            batch["target_images"] = torch.stack([
                inst["target_image"] if "target_image" in inst else zero_image
                for inst in instances
            ])
            batch["inpaint_mask_patches"] = torch.stack(
                [
                    inst["inpaint_mask_patches"] if "inpaint_mask_patches" in inst else zero_mask
                    for inst in instances
                ]
            )
        return batch


class AURORATrainer(Trainer):
    def _load_rng_state(self, checkpoint):
        # PyTorch 2.6 changed torch.load default behavior to weights_only=True.
        # Our RNG checkpoint contains numpy/python RNG tuples, so explicitly opt out
        # when loading trusted local resume checkpoints.
        if checkpoint is None:
            return

        if self.args.world_size > 1:
            process_index = self.args.process_index
            rng_file = os.path.join(checkpoint, f"rng_state_{process_index}.pth")
            if not os.path.isfile(rng_file):
                logger_module.info(
                    "Didn't find an RNG file for process %s, if you are resuming a training "
                    "that wasn't launched in a distributed fashion, reproducibility is not guaranteed.",
                    process_index,
                )
                return
        else:
            rng_file = os.path.join(checkpoint, "rng_state.pth")
            if not os.path.isfile(rng_file):
                logger_module.info(
                    "Didn't find an RNG file, if you are resuming a training that was launched "
                    "in a distributed fashion, reproducibility is not guaranteed."
                )
                return

        try:
            checkpoint_rng_state = torch.load(rng_file, weights_only=False)
        except TypeError:
            checkpoint_rng_state = torch.load(rng_file)

        random.setstate(checkpoint_rng_state["python"])
        np.random.set_state(checkpoint_rng_state["numpy"])
        torch.random.set_rng_state(checkpoint_rng_state["cpu"])
        if torch.cuda.is_available():
            if self.args.parallel_mode == ParallelMode.DISTRIBUTED:
                torch.cuda.random.set_rng_state_all(checkpoint_rng_state["cuda"])
            else:
                try:
                    torch.cuda.random.set_rng_state(checkpoint_rng_state["cuda"])
                except Exception as exc:
                    logger_module.info(
                        "Didn't manage to restore GPU RNG state: %s\n"
                        "This won't yield the same results as if training had not been interrupted.",
                        exc,
                    )
        if is_torch_tpu_available():
            xm.set_rng_state(checkpoint_rng_state["xla"])
        if is_torch_npu_available():
            if self.args.parallel_mode == ParallelMode.DISTRIBUTED:
                torch.npu.random.set_rng_state_all(checkpoint_rng_state["npu"])
            else:
                try:
                    torch.npu.random.set_rng_state(checkpoint_rng_state["npu"])
                except Exception as exc:
                    logger_module.info(
                        "Didn't manage to restore NPU RNG state: %s\n"
                        "This won't yield the same results as if training had not been interrupted.",
                        exc,
                    )

    def create_optimizer(self):
        if is_sagemaker_mp_enabled():
            return super().create_optimizer()

        if self.optimizer is not None:
            return self.optimizer

        opt_model = self.model
        decay_parameters = get_parameter_names(opt_model, ALL_LAYERNORM_LAYERS)
        decay_parameters = [name for name in decay_parameters if "bias" not in name]

        condition_lr = self.args.diff_head_lr if self.args.diff_head_lr is not None else self.args.learning_rate
        condition_names = {
            name
            for name, param in opt_model.named_parameters()
            if param.requires_grad and _is_aurora_diffusion_condition_param(name)
        }

        optimizer_grouped_parameters = [
            {
                "params": [
                    p for n, p in opt_model.named_parameters()
                    if p.requires_grad and n not in condition_names and n in decay_parameters
                ],
                "weight_decay": self.args.weight_decay,
            },
            {
                "params": [
                    p for n, p in opt_model.named_parameters()
                    if p.requires_grad and n not in condition_names and n not in decay_parameters
                ],
                "weight_decay": 0.0,
            },
            {
                "params": [
                    p for n, p in opt_model.named_parameters()
                    if p.requires_grad and n in condition_names and n in decay_parameters
                ],
                "weight_decay": self.args.weight_decay,
                "lr": condition_lr,
            },
            {
                "params": [
                    p for n, p in opt_model.named_parameters()
                    if p.requires_grad and n in condition_names and n not in decay_parameters
                ],
                "weight_decay": 0.0,
                "lr": condition_lr,
            },
        ]
        optimizer_grouped_parameters = [
            group for group in optimizer_grouped_parameters if len(group["params"]) > 0
        ]

        optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)
        self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
        return self.optimizer

    def _sanitize_nonfinite_gradients(self, model) -> None:
        inner = model.module if hasattr(model, "module") else model
        cleaned = []
        for name, param in inner.named_parameters():
            if not param.requires_grad or param.grad is None:
                continue
            if torch.isfinite(param.grad).all():
                continue
            param.grad = torch.nan_to_num(param.grad, nan=0.0, posinf=0.0, neginf=0.0)
            cleaned.append(name)
        if cleaned:
            logger_module.warning(
                "[AURORA] Sanitized non-finite gradients for %d parameter(s): %s",
                len(cleaned),
                ", ".join(cleaned[:8]) + (" ..." if len(cleaned) > 8 else ""),
            )

    def _clip_aurora_gradients(self, model) -> None:
        inner = model.module if hasattr(model, "module") else model
        max_norm = float(getattr(inner.config, 'aurora_grad_clip_max_norm', 1.0))
        if max_norm <= 0:
            return
        aurora_params = [
            p for n, p in inner.named_parameters()
            if p.requires_grad and p.grad is not None and "aurora" in n
        ]
        if aurora_params:
            torch.nn.utils.clip_grad_norm_(aurora_params, max_norm=max_norm)

    def training_step(self, model, inputs):
        inner = model.module if hasattr(model, "module") else model
        inner._aurora_global_step = int(getattr(self.state, "global_step", 0))
        loss = super().training_step(model, inputs)
        self._clip_aurora_gradients(model)
        self._sanitize_nonfinite_gradients(model)
        return loss

    def switch_to_stage2(self, model=None):
        """Switch AURORA training from stage 1 to stage 2.
        Call this between training runs or at a specific step.
        Unfreezes latent_queries and enables inpainting supervision."""
        if model is None:
            model = self.model
        inner = model.module if hasattr(model, "module") else model
        inner.config.aurora_training_stage = 2
        if inner.get_model().latent_queries is not None:
            inner.get_model().latent_queries.requires_grad_(True)
            logger_module.info(
                "[AURORA] Switched to Stage 2: latent_queries unfrozen, inpainting supervision enabled."
            )

    def _get_train_batch_size(self) -> int:
        return max(int(getattr(self, "_train_batch_size", self.args.train_batch_size)), 1)

    def get_train_dataloader(self) -> DataLoader:
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        train_dataset = self.train_dataset
        mixed_dataset = train_dataset if isinstance(train_dataset, AURORAMixedDataset) else None
        if mixed_dataset is None or not (mixed_dataset.has_task("reconstruction") and mixed_dataset.has_task("inpainting")):
            return super().get_train_dataloader()

        data_collator = self._get_collator_with_removed_columns(
            self.data_collator,
            description="training",
        )
        batch_sampler = AlternatingTaskBatchSampler(
            dataset=mixed_dataset,
            batch_size=self._get_train_batch_size(),
            warmup_steps=int(getattr(self.args, "aurora_inpaint_warmup_steps", 0)),
            num_processes=max(int(getattr(self.args, "world_size", 1)), 1),
            drop_last=bool(self.args.dataloader_drop_last),
            seed=int(getattr(self.args, "seed", 42)),
            step_provider=lambda: int(getattr(self.state, "global_step", 0)),
        )
        dataloader_params = {
            "batch_sampler": batch_sampler,
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "persistent_workers": self.args.dataloader_persistent_workers,
            "worker_init_fn": seed_worker,
        }
        # Skip accelerate.prepare — it tries to replace batch_sampler internals
        # which is incompatible with our custom AlternatingTaskBatchSampler.
        # DDP already handles gradient sync at the model level.
        return DataLoader(train_dataset, **dataloader_params)

    def _wandb_enabled(self) -> bool:
        if not self.is_world_process_zero():
            return False
        report_to = getattr(self.args, "report_to", None)
        if report_to is None:
            return False
        if isinstance(report_to, str):
            targets = {report_to}
        else:
            targets = set(report_to)
        if "wandb" not in targets and "all" not in targets:
            return False
        try:
            import wandb
        except Exception as exc:  # pragma: no cover - runtime environment specific
            logger_module.warning("[AURORA] W&B import failed, skipping reconstruction logging: %s", exc)
            return False
        return wandb.run is not None

    def _unwrap_dataset(self, dataset):
        while isinstance(dataset, torch.utils.data.Subset):
            dataset = dataset.dataset
        return dataset

    def _get_aurora_image_stats(self, dataset):
        base_dataset = self._unwrap_dataset(dataset)
        processor = base_dataset.data_args.image_processor_aux_list[0]
        mean = torch.tensor(processor.image_mean, dtype=torch.float32).view(1, 3, 1, 1)
        std = torch.tensor(processor.image_std, dtype=torch.float32).view(1, 3, 1, 1)
        return mean, std

    def _denormalize_aurora_images(
        self,
        images: torch.Tensor,
        mean: torch.Tensor,
        std: torch.Tensor,
    ) -> torch.Tensor:
        images = images.detach().cpu().float()
        images = torch.nan_to_num(images, nan=0.0, posinf=1.0, neginf=0.0)
        images = images * std + mean
        return images.clamp(0.0, 1.0)

    def _get_aurora_eval_decoder(self, model):
        decoder = getattr(self, "_aurora_eval_decoder", None)
        if decoder is not None:
            return decoder

        try:
            from huggingface_hub import hf_hub_download
            from scale_rae.model.multimodal_decoder import MultimodalDecoder
        except Exception as exc:  # pragma: no cover - runtime environment specific
            logger_module.warning("[AURORA] Decoder import failed, skipping reconstruction logging: %s", exc)
            self._aurora_eval_decoder = None
            return None

        inner = model.module if hasattr(model, "module") else model
        repo_id = getattr(self.args, "aurora_eval_decoder_repo", "nyu-visionx/siglip2_decoder")
        encoder_path = getattr(inner.config, "mm_vision_tower_aux_list", ["google/siglip2-so400m-patch14-224"])[0]
        encoder_path = encoder_path.split("-interp")[0]
        num_patches = int(getattr(inner, "num_image_tokens", 256))

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
            self._aurora_eval_decoder = decoder
            return decoder
        except Exception as exc:  # pragma: no cover - runtime environment specific
            logger_module.warning("[AURORA] Failed to load eval decoder from %s: %s", repo_id, exc)
            self._aurora_eval_decoder = None
            return None

    def _decode_aurora_generated_images(
        self,
        decoder,
        generated: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
        if generated is None:
            return torch.empty(0, 3, 1, 1)
        decoder = decoder.to(device=device, dtype=generated.dtype)
        if hasattr(decoder, "image_mean") and hasattr(decoder, "image_std"):
            decoder.image_mean = decoder.image_mean.to(device=device, dtype=generated.dtype)
            decoder.image_std = decoder.image_std.to(device=device, dtype=generated.dtype)

        empty_cls = torch.zeros(
            (generated.shape[0], 1, generated.shape[-1]),
            device=device,
            dtype=generated.dtype,
        )
        image_features = torch.cat([empty_cls, generated], dim=1)
        recon = decoder(image_features)
        recon = torch.nan_to_num(recon, nan=0.0, posinf=1.0, neginf=0.0)
        return recon.clamp(0.0, 1.0).detach().cpu().float()

    def _aurora_match_attention_maps(
        self,
        pred_attn: torch.Tensor,
        gt_masks: torch.Tensor,
    ) -> Dict[str, object]:
        gt_masks = gt_masks.detach().cpu().float()
        pred_attn = pred_attn.detach().cpu().float()

        valid_gt_indices = [
            idx for idx in range(gt_masks.shape[0])
            if torch.count_nonzero(gt_masks[idx] > 0.5).item() > 0
        ]
        if not valid_gt_indices:
            return {"gt_indices": [], "gt_ious": [], "pairs": []}

        gt_valid = gt_masks[valid_gt_indices]
        num_gt = gt_valid.shape[0]
        gt_ious = torch.zeros(num_gt, dtype=torch.float32)
        if pred_attn.numel() == 0 or pred_attn.shape[0] == 0:
            return {"gt_indices": valid_gt_indices, "gt_ious": gt_ious.tolist(), "pairs": []}

        num_pred = pred_attn.shape[0]
        pair_iou = torch.zeros(num_pred, num_gt, dtype=torch.float32)
        for gt_idx in range(num_gt):
            gt_bin = gt_valid[gt_idx] > 0.5
            k = int(gt_bin.sum().item())
            if k <= 0:
                continue
            topk = torch.topk(pred_attn, k=min(k, pred_attn.shape[-1]), dim=-1).indices
            pred_bin = torch.zeros_like(pred_attn, dtype=torch.float32)
            pred_bin.scatter_(1, topk, 1.0)
            gt_bin_f = gt_bin.unsqueeze(0).float()
            intersection = (pred_bin * gt_bin_f).sum(dim=-1)
            union = (pred_bin + gt_bin_f).clamp(max=1.0).sum(dim=-1).clamp_min(1.0)
            pair_iou[:, gt_idx] = intersection / union

        used_pred = set()
        used_gt = set()
        pairs = []
        max_matches = min(num_pred, num_gt)
        for _ in range(max_matches):
            best_iou = -1.0
            best_pred = -1
            best_gt = -1
            for pred_idx in range(num_pred):
                if pred_idx in used_pred:
                    continue
                for gt_idx in range(num_gt):
                    if gt_idx in used_gt:
                        continue
                    value = float(pair_iou[pred_idx, gt_idx].item())
                    if value > best_iou:
                        best_iou = value
                        best_pred = pred_idx
                        best_gt = gt_idx
            if best_pred < 0 or best_gt < 0:
                break
            used_pred.add(best_pred)
            used_gt.add(best_gt)
            gt_ious[best_gt] = max(best_iou, 0.0)
            pairs.append(
                {
                    "pred_idx": best_pred,
                    "gt_idx": valid_gt_indices[best_gt],
                    "iou": float(gt_ious[best_gt].item()),
                }
            )

        return {
            "gt_indices": valid_gt_indices,
            "gt_ious": gt_ious.tolist(),
            "pairs": pairs,
        }

    def _aurora_patch_map_to_image(
        self,
        patch_map: torch.Tensor,
        height: int,
        width: int,
    ) -> torch.Tensor:
        patch_map = torch.nan_to_num(patch_map.detach().cpu().float(), nan=0.0, posinf=0.0, neginf=0.0)
        side = int(round(math.sqrt(patch_map.numel())))
        if side * side != patch_map.numel():
            raise ValueError(f"Expected a square patch map, got {patch_map.numel()} patches")
        dense_map = patch_map.view(1, 1, side, side)
        dense_map = F.interpolate(dense_map, size=(height, width), mode="bilinear", align_corners=False)
        dense_map = dense_map.squeeze(0).squeeze(0).clamp(min=0.0)
        max_value = float(dense_map.amax().item())
        if max_value > 0.0:
            dense_map = dense_map / max_value
        return dense_map

    def _make_aurora_attention_overlay(
        self,
        image: torch.Tensor,
        patch_map: torch.Tensor,
        color: Sequence[float],
    ) -> torch.Tensor:
        base = torch.nan_to_num(image.detach().cpu().float(), nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
        heat = self._aurora_patch_map_to_image(
            patch_map,
            height=base.shape[-2],
            width=base.shape[-1],
        )
        color_tensor = torch.tensor(color, dtype=base.dtype).view(3, 1, 1)
        alpha = 0.55 * heat.unsqueeze(0)
        blended = base * (1.0 - alpha) + color_tensor * alpha
        return blended.clamp(0.0, 1.0)

    def _maybe_log_aurora_reconstructions(self, model, eval_dataset=None, metric_key_prefix: str = "eval") -> None:
        if not getattr(self.args, "aurora_eval_log_reconstructions", True):
            return
        if not self._wandb_enabled():
            return

        dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
        if dataset is None or len(dataset) == 0:
            return

        decoder = self._get_aurora_eval_decoder(model)
        if decoder is None:
            return

        try:
            import wandb
        except Exception:  # pragma: no cover - guarded in _wandb_enabled
            return

        max_items = min(
            max(int(getattr(self.args, "aurora_eval_log_image_count", 100)), 0),
            len(dataset),
        )
        if max_items == 0:
            return

        visual_batch_size = max(int(getattr(self.args, "aurora_eval_visual_batch_size", 4)), 1)
        mean, std = self._get_aurora_image_stats(dataset)
        loader = DataLoader(
            dataset,
            batch_size=visual_batch_size,
            shuffle=False,
            collate_fn=self.data_collator,
            num_workers=0,
        )

        inner = model.module if hasattr(model, "module") else model
        device = inner.aurora_cmd_embeddings.device
        was_training = inner.training
        rows = []

        try:
            inner.eval()
            with torch.no_grad():
                for batch in loader:
                    images = batch["images"].to(device)
                    output = inner.generate_aurora(images, guidance_level=1.0)
                    recon_images = self._decode_aurora_generated_images(
                        decoder,
                        output.get("generated"),
                        device=device,
                    )
                    source_images = self._denormalize_aurora_images(batch["images"], mean, std)

                    for idx in range(source_images.shape[0]):
                        if len(rows) >= max_items:
                            break
                        rows.append(
                            [
                                batch["image_ids"][idx],
                                wandb.Image(source_images[idx].permute(1, 2, 0).numpy()),
                                wandb.Image(recon_images[idx].permute(1, 2, 0).numpy()),
                            ]
                        )
                    if len(rows) >= max_items:
                        break
        finally:
            inner.train(was_training)

        if not rows:
            return

        columns = ["image_id", "source", "reconstruction"]
        table = wandb.Table(columns=columns, data=rows)
        wandb.log(
            {f"{metric_key_prefix}/reconstructions": table},
            step=int(getattr(self.state, "global_step", 0)),
        )

    def _compute_aurora_reconstruction_metrics(
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

        decoder = self._get_aurora_eval_decoder(model)
        if decoder is None:
            return {}

        eval_batch_size = max(int(getattr(self.args, "aurora_eval_visual_batch_size", 4)), 1)
        mean, std = self._get_aurora_image_stats(dataset)
        loader = DataLoader(
            dataset,
            batch_size=eval_batch_size,
            shuffle=False,
            collate_fn=self.data_collator,
            num_workers=0,
        )

        inner = model.module if hasattr(model, "module") else model
        device = inner.aurora_cmd_embeddings.device
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
                    output = inner.generate_aurora(images, guidance_level=1.0)
                    recon_images = self._decode_aurora_generated_images(
                        decoder,
                        output.get("generated"),
                        device=device,
                    )
                    source_images = self._denormalize_aurora_images(batch["images"], mean, std)
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

    def _compute_aurora_attention_metrics(
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
        device = inner.aurora_cmd_embeddings.device
        was_training = inner.training

        object_total = 0
        iou_sum = 0.0
        iou_50 = 0

        try:
            inner.eval()
            with torch.no_grad():
                for batch in loader:
                    images = batch["images"].to(device)
                    gt_masks = batch["gt_masks_patches"]
                    k_gt = batch["n_objects"]
                    output = inner.generate_aurora(images, guidance_level=1.0, return_generated=False)
                    attn_maps = output.get("attn_maps")
                    if attn_maps is not None and attn_maps.numel() > 0:
                        pred_attn = attn_maps.detach().cpu().float()  # (B, K, P)
                    else:
                        pred_attn = torch.zeros(
                            images.shape[0],
                            0,
                            gt_masks.shape[-1],
                            dtype=torch.float32,
                        )

                    for sample_idx in range(images.shape[0]):
                        gt_count = int(k_gt[sample_idx].item())
                        match_info = self._aurora_match_attention_maps(
                            pred_attn[sample_idx],
                            gt_masks[sample_idx, :gt_count],
                        )
                        gt_ious = match_info["gt_ious"]
                        if not gt_ious:
                            continue
                        object_total += len(gt_ious)
                        iou_sum += float(sum(gt_ious))
                        iou_50 += sum(1 for iou in gt_ious if iou >= 0.5)
        finally:
            inner.train(was_training)

        if object_total == 0:
            return {}

        return {
            f"{metric_key_prefix}_attn_patch_iou": iou_sum / object_total,
            f"{metric_key_prefix}_attn_patch_iou_50": iou_50 / object_total,
        }

    def _maybe_log_aurora_attention_overlays(
        self,
        model,
        eval_dataset=None,
        metric_key_prefix: str = "eval",
    ) -> None:
        if not getattr(self.args, "aurora_eval_log_attention_overlays", True):
            return
        if not self._wandb_enabled():
            return

        dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
        if dataset is None or len(dataset) == 0:
            return

        try:
            import wandb
        except Exception:  # pragma: no cover - guarded in _wandb_enabled
            return

        max_rows = max(int(getattr(self.args, "aurora_eval_attention_overlay_count", 50)), 0)
        if max_rows == 0:
            return

        visual_batch_size = max(int(getattr(self.args, "aurora_eval_visual_batch_size", 4)), 1)
        mean, std = self._get_aurora_image_stats(dataset)
        loader = DataLoader(
            dataset,
            batch_size=visual_batch_size,
            shuffle=False,
            collate_fn=self.data_collator,
            num_workers=0,
        )

        inner = model.module if hasattr(model, "module") else model
        device = inner.aurora_cmd_embeddings.device
        was_training = inner.training
        rows = []

        try:
            inner.eval()
            with torch.no_grad():
                for batch in loader:
                    images = batch["images"].to(device)
                    gt_masks = batch["gt_masks_patches"]
                    k_gt = batch["n_objects"]
                    source_images = self._denormalize_aurora_images(batch["images"], mean, std)
                    output = inner.generate_aurora(images, guidance_level=1.0, return_generated=False)
                    attn_maps = output.get("attn_maps")

                    if attn_maps is not None and attn_maps.numel() > 0:
                        pred_attn = attn_maps.detach().cpu().float()  # (B, K, P)
                    else:
                        pred_attn = torch.zeros(
                            images.shape[0],
                            0,
                            gt_masks.shape[-1],
                            dtype=torch.float32,
                        )

                    for sample_idx in range(images.shape[0]):
                        match_info = self._aurora_match_attention_maps(
                            pred_attn[sample_idx],
                            gt_masks[sample_idx, : int(k_gt[sample_idx].item())],
                        )
                        source = source_images[sample_idx]
                        for pair in match_info["pairs"]:
                            if len(rows) >= max_rows:
                                break
                            pred_overlay = self._make_aurora_attention_overlay(
                                source,
                                pred_attn[sample_idx, pair["pred_idx"]],
                                color=(1.0, 0.15, 0.15),
                            )
                            gt_overlay = self._make_aurora_attention_overlay(
                                source,
                                gt_masks[sample_idx, pair["gt_idx"]],
                                color=(0.15, 0.85, 0.25),
                            )
                            rows.append(
                                [
                                    batch["image_ids"][sample_idx],
                                    int(pair["gt_idx"]),
                                    int(pair["pred_idx"]),
                                    float(pair["iou"]),
                                    wandb.Image(source.permute(1, 2, 0).numpy()),
                                    wandb.Image(pred_overlay.permute(1, 2, 0).numpy()),
                                    wandb.Image(gt_overlay.permute(1, 2, 0).numpy()),
                                ]
                            )
                        if len(rows) >= max_rows:
                            break
                    if len(rows) >= max_rows:
                        break
        finally:
            inner.train(was_training)

        if not rows:
            return

        columns = [
            "image_id",
            "gt_index",
            "pred_index",
            "attn_patch_iou",
            "source",
            "pred_overlay",
            "gt_overlay",
        ]
        table = wandb.Table(columns=columns, data=rows)
        wandb.log(
            {f"{metric_key_prefix}/attention_overlays": table},
            step=int(getattr(self.state, "global_step", 0)),
        )

    def _compute_aurora_inpaint_weight(self, model) -> float:
        inner = model.module if hasattr(model, "module") else model
        base_weight = float(getattr(inner.config, "aurora_inpaint_weight", 0.5))
        warmup_steps = max(int(getattr(self.args, "aurora_inpaint_warmup_steps", 0)), 0)
        ramp_steps = max(int(getattr(self.args, "aurora_inpaint_ramp_steps", 0)), 0)
        step = int(getattr(self.state, "global_step", 0))

        if base_weight <= 0.0:
            return 0.0
        if step < warmup_steps:
            return 0.0
        if ramp_steps <= 0:
            return base_weight

        progress = min(max((step - warmup_steps + 1) / float(ramp_steps), 0.0), 1.0)
        return base_weight * progress

    def compute_loss(self, model, inputs, return_outputs=False):
        images = inputs.pop("images")
        n_objects_tensor = inputs.pop("n_objects", None)
        inner = model.module if hasattr(model, "module") else model
        inpaint_weight = 0.0
        if int(getattr(inner.config, "aurora_training_stage", 1)) >= 2:
            inpaint_weight = self._compute_aurora_inpaint_weight(model)
        kwargs = {
            "n_objects": n_objects_tensor,
            "gt_masks_patches": inputs.pop("gt_masks_patches", None),
            "target_images": inputs.pop("target_images", None),
            "inpaint_mask_patches": inputs.pop("inpaint_mask_patches", None),
            "has_inpaint": inputs.pop("has_inpaint", None),
            "aurora_inpaint_weight_override": inpaint_weight,
        }

        outputs = model(
            input_ids=torch.zeros(images.shape[0], 1, dtype=torch.long, device=images.device),
            images=images,
            labels=None,
            **kwargs,
        )
        loss = outputs.loss if hasattr(outputs, "loss") else outputs[0]

        custom = getattr(self, "_custom_losses", {})
        for key, attr in [
            ("loss_recon", "aurora_loss_recon"),
            ("loss_mask", "aurora_loss_mask"),
            ("loss_div", "aurora_loss_div"),
            ("loss_inpaint", "aurora_loss_inpaint"),
        ]:
            value = getattr(inner, attr, None)
            if value is not None:
                custom.setdefault(key, []).append(
                    value.item() if torch.is_tensor(value) else float(value)
                )
        current_inpaint_weight = getattr(inner, "aurora_inpaint_weight", None)
        if current_inpaint_weight is not None and int(getattr(inner.config, "aurora_training_stage", 1)) >= 2:
            custom.setdefault("inpaint_weight", []).append(
                current_inpaint_weight.item()
                if torch.is_tensor(current_inpaint_weight)
                else float(current_inpaint_weight)
            )
        self._custom_losses = custom
        return (loss, outputs) if return_outputs else loss

    def prediction_step(
        self,
        model,
        inputs,
        prediction_loss_only: bool,
        ignore_keys=None,
    ):
        if "images" not in inputs:
            return super().prediction_step(
                model,
                inputs,
                prediction_loss_only=prediction_loss_only,
                ignore_keys=ignore_keys,
            )

        inputs = self._prepare_inputs(inputs)
        with torch.no_grad():
            loss = self.compute_loss(model, inputs, return_outputs=False)
        loss = loss.detach()
        return (loss, None, None)

    def log(self, logs: Dict[str, float]) -> None:
        custom = getattr(self, "_custom_losses", {})
        prefix = getattr(self, "_custom_loss_prefix", None)
        for key, values in custom.items():
            if values:
                metric_name = f"{prefix}_{key}" if prefix else key
                logs[metric_name] = sum(values) / len(values)
        custom.clear()
        super().log(logs)

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
            recon_metrics = {}
            try:
                recon_metrics = self._compute_aurora_reconstruction_metrics(
                    self.model,
                    eval_dataset=eval_dataset,
                    metric_key_prefix=metric_key_prefix,
                )
            except Exception as exc:  # pragma: no cover - best effort metrics only
                logger_module.warning("[AURORA] Failed to compute reconstruction metrics: %s", exc)
            if recon_metrics:
                metrics.update(recon_metrics)
                self.log(recon_metrics)
            attn_metrics = {}
            try:
                attn_metrics = self._compute_aurora_attention_metrics(
                    self.model,
                    eval_dataset=eval_dataset,
                    metric_key_prefix=metric_key_prefix,
                )
            except Exception as exc:  # pragma: no cover - best effort metrics only
                logger_module.warning("[AURORA] Failed to compute attention metrics: %s", exc)
            if attn_metrics:
                metrics.update(attn_metrics)
                self.log(attn_metrics)
            try:
                self._maybe_log_aurora_attention_overlays(
                    self.model,
                    eval_dataset=eval_dataset,
                    metric_key_prefix=metric_key_prefix,
                )
            except Exception as exc:  # pragma: no cover - best effort logging only
                logger_module.warning("[AURORA] Failed to log attention overlays: %s", exc)
            try:
                self._maybe_log_aurora_reconstructions(
                    self.model,
                    eval_dataset=eval_dataset,
                    metric_key_prefix=metric_key_prefix,
                )
            except Exception as exc:  # pragma: no cover - best effort logging only
                logger_module.warning("[AURORA] Failed to log reconstruction images: %s", exc)
            return metrics
        finally:
            self._custom_loss_prefix = prev_prefix
            self._custom_losses = prev_custom


def make_data_module(tokenizer, data_args, model_configs, training_args=None) -> Dict:
    annotation_index = load_coco_annotation_index(
        annotation_path=getattr(data_args, "coco_annotation_path", None),
        min_area=getattr(data_args, "aurora_min_area", 1024.0),
    )
    if annotation_index is not None:
        _get_mask_utils()

    datasets: List[Dataset] = []
    task_names: List[str] = []
    recon_dataset: Optional[Dataset] = None
    recon_image_root = (
        data_args.aurora_reconstruction_image_folder
        or (
            data_args.image_folder
            if data_args.data_path is None and data_args.image_folder is not None
            else None
        )
    )
    if (
        getattr(data_args, "aurora_include_reconstruction", True)
        and recon_image_root is not None
        and annotation_index is not None
    ):
        recon_dataset = COCOReconstructionDataset(
            image_root=recon_image_root,
            data_args=data_args,
            model_configs=model_configs,
            annotation_index=annotation_index,
        )
        logger_module.info(
            "[AURORA] Added %d reconstruction samples from %s",
            len(recon_dataset),
            recon_image_root,
        )
        datasets.append(recon_dataset)
        task_names.append("reconstruction")

    inpaint_data_path = data_args.aurora_inpaint_data_path or data_args.data_path
    inpaint_image_folder = data_args.aurora_inpaint_image_folder or data_args.image_folder
    if getattr(data_args, "aurora_include_inpainting", True) and inpaint_data_path:
        inpaint_dataset = AURORADataset(
            data_path=inpaint_data_path,
            data_args=data_args,
            model_configs=model_configs,
            annotation_index=annotation_index,
            image_folder=inpaint_image_folder,
        )
        logger_module.info(
            "[AURORA] Added %d inpainting samples from %s",
            len(inpaint_dataset),
            inpaint_data_path,
        )
        datasets.append(inpaint_dataset)
        task_names.append("inpainting")

    if not datasets:
        raise ValueError(
            "No AURORA datasets were configured. Provide COCO reconstruction paths and/or an inpainting manifest."
        )

    train_dataset = datasets[0] if len(datasets) == 1 else AURORAMixedDataset(datasets, task_names, data_args)
    eval_limit = 100
    if training_args is not None:
        eval_limit = max(int(getattr(training_args, "aurora_eval_num_images", 100)), 0)

    if eval_limit == 0:
        eval_dataset = torch.utils.data.Subset(train_dataset, [])
    elif isinstance(train_dataset, AURORAMixedDataset):
        active_indices = [idx for idx, dataset in enumerate(datasets) if len(dataset) > 0]
        quotas = [0 for _ in datasets]
        target_total = min(eval_limit, len(train_dataset))
        if active_indices and target_total > 0:
            base = target_total // len(active_indices)
            remainder = target_total % len(active_indices)
            for offset, dataset_idx in enumerate(active_indices):
                requested = base + (1 if offset < remainder else 0)
                quotas[dataset_idx] = min(len(datasets[dataset_idx]), requested)

            remaining = target_total - sum(quotas)
            while remaining > 0:
                progressed = False
                for dataset_idx in active_indices:
                    if quotas[dataset_idx] >= len(datasets[dataset_idx]):
                        continue
                    quotas[dataset_idx] += 1
                    remaining -= 1
                    progressed = True
                    if remaining == 0:
                        break
                if not progressed:
                    break

        eval_indices: List[int] = []
        eval_mix_summary = []
        for dataset_idx, (task_name, dataset) in enumerate(zip(task_names, datasets)):
            quota = quotas[dataset_idx]
            if quota <= 0:
                continue
            task_range = train_dataset.get_task_range(task_name)
            if task_range is None:
                continue
            start, _ = task_range
            if quota >= len(dataset):
                local_indices = list(range(len(dataset)))
            else:
                local_indices = torch.linspace(0, len(dataset) - 1, steps=quota).round().long().tolist()
            eval_indices.extend(start + int(local_idx) for local_idx in local_indices)
            eval_mix_summary.append(f"{task_name}={len(local_indices)}")

        eval_dataset = torch.utils.data.Subset(train_dataset, eval_indices)
        logger_module.info(
            "[AURORA] Eval subset: %d samples (%s)",
            len(eval_indices),
            ", ".join(eval_mix_summary) if eval_mix_summary else "empty",
        )
    else:
        eval_size = min(eval_limit, len(train_dataset))
        eval_dataset = torch.utils.data.Subset(train_dataset, list(range(eval_size)))
    return dict(
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=AURORADataCollator(),
    )


def freeze_for_aurora_phase1(model):
    model.requires_grad_(False)
    stage = int(getattr(model.config, "aurora_training_stage", 1))
    train_diffusion_condition = bool(
        getattr(model.config, "aurora_train_diffusion_condition", True)
    )
    trainable_keywords = [
        "aurora_cmd_embeddings",
        "aurora_obj_embedding_pool",
        "aurora_reg_embeddings",
        "diff_head_projector",
    ]
    if stage >= 2:
        trainable_keywords.append("latent_queries")
    n_trainable = 0
    for name, param in model.named_parameters():
        should_train = any(keyword in name for keyword in trainable_keywords)
        if train_diffusion_condition and _is_aurora_diffusion_condition_param(name):
            should_train = True
        if should_train:
            param.requires_grad = True
            if param.is_floating_point() and param.dtype != torch.float32:
                param.data = param.data.to(torch.float32)
            n_trainable += param.numel()
    logger_module.info(
        "[AURORA] Stage %d trainable params: %d (diffusion_condition=%s)",
        stage,
        n_trainable,
        train_diffusion_condition,
    )
    return model


def train():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    compute_dtype = (
        torch.float16 if training_args.fp16
        else (torch.bfloat16 if training_args.bf16 else torch.float32)
    )

    if model_args.use_aurora and training_args.gradient_checkpointing:
        logger_module.warning(
            "[AURORA] Disabling gradient checkpointing because the slot-chain path requires KV-cache."
        )
        training_args.gradient_checkpointing = False

    from transformers import AutoConfig

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

    config.use_aurora = model_args.use_aurora
    config.aurora_max_slots = model_args.aurora_max_slots
    config.aurora_n_register = model_args.aurora_n_register
    config.aurora_cmd_length = model_args.aurora_cmd_length
    config.aurora_mask_loss_weight = model_args.aurora_mask_loss_weight
    config.aurora_diversity_loss_weight = model_args.aurora_diversity_loss_weight
    config.aurora_inpaint_weight = model_args.aurora_inpaint_weight
    config.aurora_fail_on_nan = model_args.aurora_fail_on_nan
    config.aurora_training_stage = model_args.aurora_training_stage
    config.aurora_grad_clip_max_norm = model_args.aurora_grad_clip_max_norm
    config.aurora_train_diffusion_condition = model_args.aurora_train_diffusion_condition
    print(f"[AURORA-TRAINER] model_args.diffusion_norm_stats_path = {model_args.diffusion_norm_stats_path}")
    if model_args.diffusion_norm_stats_path:
        config.diffusion_norm_stats_path = model_args.diffusion_norm_stats_path
    print(f"[AURORA-TRAINER] config.diffusion_norm_stats_path = {getattr(config, 'diffusion_norm_stats_path', 'NOT SET')}")

    model = ScaleRAEQwenForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        torch_dtype=compute_dtype,
        ignore_mismatched_sizes=True,
    )
    model.config.use_cache = False

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

        vision_tower_aux_list = model.get_vision_tower_aux_list()
        if not training_args.unfreeze_mm_vision_tower:
            for vt in vision_tower_aux_list:
                vt.to(dtype=compute_dtype, device=training_args.device)

        data_args.image_processor_aux_list = [vt.image_processor for vt in vision_tower_aux_list]
        data_args.is_multimodal = True
        data_args.vision_tower_aux_token_len_list = model_args.vision_tower_aux_token_len_list

        model.config.image_aspect_ratio = data_args.image_aspect_ratio
        model.config.vision_tower_aux_token_len_list = model_args.vision_tower_aux_token_len_list
        model.config.si_token_len = model_args.si_token_len
        model.config.miv_token_len = model_args.miv_token_len

    model = freeze_for_aurora_phase1(model)
    model.initialize_vision_tokenizer(model_args, tokenizer=tokenizer)

    data_module = make_data_module(tokenizer, data_args, model.config, training_args=training_args)
    trainer = AURORATrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        **data_module,
    )

    resume = training_args.resume_from_checkpoint or None
    trainer.train(resume_from_checkpoint=resume)
    trainer.save_state()
    trainer.save_model(training_args.output_dir)
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    train()
