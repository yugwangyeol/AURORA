"""VOC and MOVi evaluation adapters for PGOT's autoregressive evaluator."""

from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from pgot.constants import OVT_TOKEN, SCENE_END_TOKEN, THING_TOKEN
from pgot.train.pgot_dataset import _coda_center_crop_image, _pil_resample


def _remap_contiguous(mask: np.ndarray) -> np.ndarray:
    """Map the labels present in a mask to 0..N while retaining background 0."""
    labels = np.unique(mask)
    out = np.zeros(mask.shape, dtype=np.int64)
    next_label = 1
    for label in labels:
        label = int(label)
        if label == 0:
            continue
        out[mask == label] = next_label
        next_label += 1
    return out


def _dummy_caption(tokenizer, object_count: int, max_objects: int, max_tokens: int):
    """Build a collatable placeholder that AR evaluation always replaces."""
    parts: List[str] = []
    used = min(int(object_count), int(max_objects))
    for index in range(used):
        parts.append(
            f"{THING_TOKEN} Object {index + 1}: an object. {OVT_TOKEN}."
        )
    text = " ".join(parts) + f" {SCENE_END_TOKEN}"
    ids = tokenizer.encode(text, add_special_tokens=False)
    if len(ids) > int(max_tokens):
        raise ValueError(
            f"Placeholder caption has {len(ids)} tokens, exceeding max_tokens={max_tokens}"
        )
    return text, torch.tensor(ids, dtype=torch.long)


class CodaEvalPGOTDataset(Dataset):
    """Expose CODA's VOC/MOVi validation data in PGOT's batch schema.

    The placeholder caption and OVT tensors exist only to satisfy the common
    collator. ``run_eval`` requires autoregressive caption mode for this class,
    so no ground-truth object count or label reaches the model forward pass.
    """

    def __init__(
        self,
        dataset_name: str,
        data_root: str,
        tokenizer,
        image_processor,
        target_image_processor=None,
        eval_size: int = 512,
        grid_size: int = 32,
        rae_grid_size: int = 16,
        max_caption_tokens: int = 1024,
        n_ovt_per_object: int = 1,
        max_objects: int = 50,
    ):
        super().__init__()
        if dataset_name not in {"voc", "movi-c", "movi-e"}:
            raise ValueError(f"Unsupported CODA evaluation dataset: {dataset_name}")
        if int(n_ovt_per_object) != 1:
            raise ValueError("CODA dataset AR evaluation requires n_ovt_per_object=1")

        self.dataset_name = dataset_name
        self.root = Path(data_root)
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.target_image_processor = target_image_processor or image_processor
        self.eval_size = int(eval_size)
        self.grid_size = int(grid_size)
        self.rae_grid_size = int(rae_grid_size)
        self.max_caption_tokens = int(max_caption_tokens)
        self.n_ovt_per_object = int(n_ovt_per_object)
        self.max_objects = int(max_objects)
        self.ovt_token_id = int(tokenizer.convert_tokens_to_ids(OVT_TOKEN))

        if dataset_name == "voc":
            split_file = self.root / "sets" / "val.txt"
            if not split_file.is_file():
                raise FileNotFoundError(f"Missing VOC split: {split_file}")
            self.records = [line.strip() for line in split_file.read_text().splitlines() if line.strip()]
        else:
            validation_root = self.root / "validation"
            self.records = sorted(validation_root.glob("**/*.jpg"))
            if not self.records:
                raise FileNotFoundError(f"No MOVi validation images found under {validation_root}")

    def __len__(self):
        return len(self.records)

    def _load_voc(self, index: int):
        sample_id = self.records[index]
        image = Image.open(self.root / "images" / f"{sample_id}.jpg").convert("RGB")
        instance = Image.open(self.root / "SegmentationObject" / f"{sample_id}.png")
        semantic = Image.open(self.root / "SegmentationClass" / f"{sample_id}.png")

        image = _coda_center_crop_image(image, self.eval_size)
        instance = _coda_center_crop_image(
            instance, self.eval_size, resample=_pil_resample("NEAREST")
        )
        semantic = _coda_center_crop_image(
            semantic, self.eval_size, resample=_pil_resample("NEAREST")
        )
        instance_np = np.asarray(instance, dtype=np.uint8).copy()
        semantic_np = np.asarray(semantic, dtype=np.uint8).copy()
        overlap = (instance_np == 255) | (semantic_np == 255)
        instance_np[overlap] = 0
        semantic_np[overlap] = 0
        return (
            image,
            _remap_contiguous(instance_np),
            _remap_contiguous(semantic_np),
            overlap.astype(np.uint8),
        )

    def _load_movi(self, index: int):
        image_path = self.records[index]
        mask_path = image_path.with_name(f"{image_path.stem}_mask.png")
        image = Image.open(image_path).convert("RGB").resize(
            (self.eval_size, self.eval_size), _pil_resample("BILINEAR")
        )
        mask = Image.open(mask_path).resize(
            (self.eval_size, self.eval_size), _pil_resample("NEAREST")
        )
        instance = np.asarray(mask, dtype=np.int64).copy()
        overlap = np.zeros(instance.shape, dtype=np.uint8)
        return image, instance, None, overlap

    def __getitem__(self, index: int) -> Dict:
        if self.dataset_name == "voc":
            image, instance, semantic, overlap = self._load_voc(index)
        else:
            image, instance, semantic, overlap = self._load_movi(index)

        image_tensor = self.image_processor.preprocess(
            image, return_tensors="pt"
        )["pixel_values"][0]
        target_image_tensor = self.target_image_processor.preprocess(
            image, return_tensors="pt"
        )["pixel_values"][0]

        gt_object_count = int(np.count_nonzero(np.unique(instance)))
        caption_text, caption_ids = _dummy_caption(
            self.tokenizer,
            gt_object_count,
            self.max_objects,
            self.max_caption_tokens,
        )
        ovt_positions = (
            caption_ids == self.ovt_token_id
        ).nonzero(as_tuple=False).flatten()
        n_ovt_max = self.max_objects * self.n_ovt_per_object
        keep = min(int(ovt_positions.numel()), n_ovt_max)
        ovt_pos_padded = torch.zeros(n_ovt_max, dtype=torch.long)
        ovt_valid = torch.zeros(n_ovt_max, dtype=torch.bool)
        if keep:
            ovt_pos_padded[:keep] = ovt_positions[:keep]
            ovt_valid[:keep] = True

        result = {
            "image": image_tensor,
            "target_image": target_image_tensor,
            "caption_input_ids": caption_ids,
            "ovt_positions_in_caption": ovt_pos_padded,
            "ovt_valid_mask": ovt_valid,
            "ovt_is_thing": ovt_valid.clone(),
            "ovt_category_ids": torch.full((n_ovt_max,), -1, dtype=torch.long),
            "gt_masks_per_ovt": torch.zeros(
                (n_ovt_max, self.grid_size * self.grid_size), dtype=torch.float32
            ),
            "gt_rae_masks_per_ovt": torch.zeros(
                (n_ovt_max, self.rae_grid_size * self.rae_grid_size), dtype=torch.float32
            ),
            "image_id": int(index),
            "n_objects": gt_object_count,
            "caption_text": caption_text,
            "gt_mask": torch.from_numpy(instance).long(),
            "overlap_mask": torch.from_numpy(overlap).to(torch.uint8),
        }
        if semantic is not None:
            result["sem_mask"] = torch.from_numpy(semantic).long()
        return result
