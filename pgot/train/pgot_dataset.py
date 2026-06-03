"""PGOT Dataset & DataCollator.

Loads pre-processed Pix2Cap JSONL (one sample per line) and converts each into
a training instance:
    image_tensor               (preprocessed by SigLIP processor)
    target_image_tensor        (preprocessed by 2nd processor for diffusion target)
    caption_input_ids          (Qwen2 tokenized caption with <ovt> tokens inline)
    ovt_positions_in_caption   (long, position of each <ovt> token in caption)
    ovt_valid_mask             (bool)
    gt_masks_per_ovt           (float, [M_max, P]) — per-OVT patch mask
    image_id, n_objects, etc.

The caption is built EXACTLY like the preprocessing script:
    "Person 1: ... <ovt><ovt>. Giraffe 1: ... <ovt><ovt>. ... <scene_end>"

Each object contributes `n_ovt_per_object` consecutive <ovt> tokens, all sharing
the same patch mask.
"""

import json
import os
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Set

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset


def _mask_to_patch_mask(mask: torch.Tensor, grid_size: int) -> torch.Tensor:
    """Resample binary mask (H, W) -> (grid_size*grid_size,) float fraction.

    Uses bilinear interpolation so that a partial patch overlap contributes
    proportionally. Same approach as AURORA.
    """
    if mask.ndim == 2:
        mask = mask.unsqueeze(0).unsqueeze(0).float()
    else:
        mask = mask.float()
    patches = F.interpolate(
        mask,
        size=(grid_size, grid_size),
        mode="bilinear",
        align_corners=False,
    )
    return patches.flatten(start_dim=1).squeeze(0)  # (grid_size**2,)


class Pix2CapPGOTDataset(Dataset):
    """Loads the JSONL written by preprocess/prepare_pgot_data.py."""

    def __init__(
        self,
        jsonl_path: str,
        tokenizer,
        image_processor,
        target_image_processor=None,
        grid_size: int = 16,
        max_caption_tokens: int = 2048,
        n_ovt_per_object: int = 2,
        max_objects: int = 50,
        ovt_token: str = "<ovt>",
        scene_end_token: str = "<scene_end>",
        rebuild_caption: bool = True,
        panoptic_categories_json: Optional[str] = None,
    ):
        super().__init__()
        self.jsonl_path = jsonl_path
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.target_image_processor = target_image_processor or image_processor
        self.grid_size = grid_size
        self.max_caption_tokens = max_caption_tokens
        self.n_ovt_per_object = n_ovt_per_object
        self.max_objects = max_objects
        self.ovt_token = ovt_token
        self.scene_end_token = scene_end_token
        self.rebuild_caption = rebuild_caption
        self.thing_category_ids = self._load_thing_category_ids(panoptic_categories_json)

        self.ovt_token_id = tokenizer.convert_tokens_to_ids(ovt_token)
        self.scene_end_token_id = tokenizer.convert_tokens_to_ids(scene_end_token)
        assert self.ovt_token_id != tokenizer.unk_token_id, f"{ovt_token} not in tokenizer"
        assert self.scene_end_token_id != tokenizer.unk_token_id, f"{scene_end_token} not in tokenizer"

        self.samples = []
        with open(jsonl_path) as f:
            for line in f:
                self.samples.append(json.loads(line))

    @staticmethod
    def _load_thing_category_ids(panoptic_categories_json: Optional[str]) -> Set[int]:
        if panoptic_categories_json and os.path.exists(panoptic_categories_json):
            with open(panoptic_categories_json) as f:
                d = json.load(f)
            return {
                int(c["id"])
                for c in d.get("categories", [])
                if int(c.get("isthing", 0)) == 1
            }
        # COCO panoptic thing categories are the instance ids 1..90 (with gaps).
        # This fallback keeps older JSONL-only setups usable.
        return set(range(1, 91))

    def __len__(self):
        return len(self.samples)

    # --------------------------------------------------------------
    def _load_panoptic_id_map(self, mask_path: str) -> np.ndarray:
        """Load a panoptic PNG and decode RGB -> int segment id."""
        rgb = np.array(Image.open(mask_path).convert("RGB"))  # (H, W, 3)
        seg_id = (
            rgb[..., 0].astype(np.int64)
            + rgb[..., 1].astype(np.int64) * 256
            + rgb[..., 2].astype(np.int64) * 256 * 256
        )
        return seg_id  # (H, W)

    def _segment_mask(self, seg_id_map: np.ndarray, segment_id: int) -> torch.Tensor:
        mask = torch.from_numpy(seg_id_map == segment_id)
        return mask  # bool (H, W)

    def _build_caption_with_ovt(self, segments: List[dict]) -> str:
        """Re-build caption from segments — mirrors preprocess/prepare_pgot_data.py."""
        from collections import defaultdict
        cat_counter = defaultdict(int)
        parts = []
        ovt_str = self.ovt_token * self.n_ovt_per_object
        for seg in segments:
            cat = seg["category"]
            cat_counter[cat] += 1
            cat_label = f"{cat.capitalize()} {cat_counter[cat]}"
            desc = seg["description"].rstrip(".").rstrip()
            parts.append(f"{cat_label}: {desc}. {ovt_str}.")
        return " ".join(parts) + f" {self.scene_end_token}"

    # --------------------------------------------------------------
    def __getitem__(self, idx: int) -> Dict:
        sample = self.samples[idx]

        # 1) Image
        image = Image.open(sample["image_path"]).convert("RGB")
        image_tensor = self.image_processor.preprocess(image, return_tensors="pt")["pixel_values"][0]
        target_image_tensor = self.target_image_processor.preprocess(image, return_tensors="pt")["pixel_values"][0]

        # 2) Caption
        if self.rebuild_caption:
            caption_text = self._build_caption_with_ovt(sample["segments"])
        else:
            caption_text = sample["caption"]
        # Tokenize WITHOUT special tokens — we don't want BOS/EOS inserted here
        token_ids = self.tokenizer.encode(caption_text, add_special_tokens=False)

        # 3) Cap caption length and truncate trailing objects if it overflows
        segments = sample["segments"][: self.max_objects]
        if len(token_ids) > self.max_caption_tokens:
            token_ids = token_ids[: self.max_caption_tokens]

        caption_input_ids = torch.tensor(token_ids, dtype=torch.long)

        # 4) Locate <ovt> positions inside the caption sequence
        ovt_positions = (caption_input_ids == self.ovt_token_id).nonzero(as_tuple=False).flatten().tolist()
        # Truncate ovt positions to a multiple of n_ovt_per_object
        n_ovt_max = self.max_objects * self.n_ovt_per_object
        n_keep = (len(ovt_positions) // self.n_ovt_per_object) * self.n_ovt_per_object
        n_keep = min(n_keep, n_ovt_max)
        ovt_positions = ovt_positions[:n_keep]

        # 5) Per-OVT patch masks (same mask for both OVTs of the same object)
        seg_id_map = self._load_panoptic_id_map(sample["panoptic_mask_path"])
        gt_masks_per_ovt = torch.zeros((n_ovt_max, self.grid_size * self.grid_size), dtype=torch.float32)
        ovt_is_thing = torch.zeros(n_ovt_max, dtype=torch.bool)
        n_obj_actual = n_keep // self.n_ovt_per_object
        for obj_idx in range(n_obj_actual):
            seg_info = segments[obj_idx]
            mask_hw = self._segment_mask(seg_id_map, int(seg_info["segment_id"])).float()
            patch_mask = _mask_to_patch_mask(mask_hw, self.grid_size)
            is_thing = int(seg_info.get("category_id", -1)) in self.thing_category_ids
            for j in range(self.n_ovt_per_object):
                ovt_idx = obj_idx * self.n_ovt_per_object + j
                gt_masks_per_ovt[ovt_idx] = patch_mask
                ovt_is_thing[ovt_idx] = is_thing

        # 6) Pad ovt positions
        ovt_pos_padded = torch.zeros(n_ovt_max, dtype=torch.long)
        ovt_valid = torch.zeros(n_ovt_max, dtype=torch.bool)
        for i, p in enumerate(ovt_positions):
            ovt_pos_padded[i] = p
            ovt_valid[i] = True

        return {
            "image": image_tensor,
            "target_image": target_image_tensor,
            "caption_input_ids": caption_input_ids,
            "ovt_positions_in_caption": ovt_pos_padded,
            "ovt_valid_mask": ovt_valid,
            "ovt_is_thing": ovt_is_thing,
            "gt_masks_per_ovt": gt_masks_per_ovt,
            "image_id": int(sample["image_id"]),
            "n_objects": n_obj_actual,
            "caption_text": caption_text,
        }


@dataclass
class PGOTDataCollator:
    pad_token_id: int

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        B = len(instances)
        max_cap_len = max(int(inst["caption_input_ids"].shape[0]) for inst in instances)
        caption_input_ids = torch.full((B, max_cap_len), self.pad_token_id, dtype=torch.long)
        caption_attention_mask = torch.zeros((B, max_cap_len), dtype=torch.bool)
        caption_labels = torch.full((B, max_cap_len), -100, dtype=torch.long)
        for i, inst in enumerate(instances):
            L = int(inst["caption_input_ids"].shape[0])
            caption_input_ids[i, :L] = inst["caption_input_ids"]
            caption_attention_mask[i, :L] = True
            caption_labels[i, :L] = inst["caption_input_ids"]

        return {
            "images": torch.stack([inst["image"] for inst in instances]),
            "target_images": torch.stack([inst["target_image"] for inst in instances]),
            "caption_input_ids": caption_input_ids,
            "caption_attention_mask": caption_attention_mask,
            "caption_labels": caption_labels,
            "ovt_positions_in_caption": torch.stack(
                [inst["ovt_positions_in_caption"] for inst in instances]
            ),
            "ovt_valid_mask": torch.stack([inst["ovt_valid_mask"] for inst in instances]),
            "ovt_is_thing": torch.stack([inst["ovt_is_thing"] for inst in instances]),
            "gt_masks_per_ovt": torch.stack([inst["gt_masks_per_ovt"] for inst in instances]),
            "image_ids": [inst["image_id"] for inst in instances],
            "n_objects_list": [inst["n_objects"] for inst in instances],
            "caption_texts": [inst["caption_text"] for inst in instances],
        }
