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
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset


def _mask_to_patch_mask(
    mask: torch.Tensor,
    grid_size: int,
    mode: str = "bilinear",
) -> torch.Tensor:
    """Resample binary mask (H, W) -> (grid_size*grid_size,) float fraction.

    Uses bilinear interpolation so that a partial patch overlap contributes
    proportionally. Same approach as AURORA.
    """
    if mask.ndim == 2:
        mask = mask.unsqueeze(0).unsqueeze(0).float()
    else:
        mask = mask.float()
    kwargs = {"size": (grid_size, grid_size), "mode": mode}
    if mode in {"bilinear", "bicubic"}:
        kwargs["align_corners"] = False
    patches = F.interpolate(mask, **kwargs)
    return patches.flatten(start_dim=1).squeeze(0)  # (grid_size**2,)


def _pil_resample(name: str):
    resampling = getattr(Image, "Resampling", Image)
    return getattr(resampling, name)


def _coda_center_crop_image(
    image: Image.Image,
    size: int,
    resample=None,
) -> Image.Image:
    """CODA COCO transform: resize min side to size, then center-crop square."""
    if resample is None:
        resample = _pil_resample("BILINEAR")
    w, h = image.size
    factor = max(float(size) / float(h), float(size) / float(w))
    rw = int(round(w * factor))
    rh = int(round(h * factor))
    image = image.resize((rw, rh), resample)
    left = max((rw - size) // 2, 0)
    top = max((rh - size) // 2, 0)
    return image.crop((left, top, left + size, top + size))


class Pix2CapPGOTDataset(Dataset):
    """Loads either a Pix2Cap-panoptic or COCO-instance PGOT JSONL.

    COCO-instance manifests are produced by
    ``preprocess/prepare_coco_instance_pgot.py`` and are detected through the
    per-record ``dataset_type`` field. Keeping both formats in one class keeps
    old checkpoints and scripts operational while giving E1 a thing-only path.
    """

    def __init__(
        self,
        jsonl_path: str,
        tokenizer,
        image_processor,
        target_image_processor=None,
        grid_size: int = 16,
        rae_grid_size: int = 16,
        max_caption_tokens: int = 2048,
        n_ovt_per_object: int = 2,
        max_objects: int = 50,
        ovt_token: str = "<ovt>",
        scene_end_token: str = "<scene_end>",
        thing_token: str = "<thing>",
        stuff_token: str = "<stuff>",
        rebuild_caption: bool = True,
        panoptic_categories_json: Optional[str] = None,
        image_preprocess_mode: str = "default",
        coda_crop_size: int = 512,
    ):
        super().__init__()
        self.jsonl_path = jsonl_path
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.target_image_processor = target_image_processor or image_processor
        self.grid_size = grid_size
        self.rae_grid_size = rae_grid_size
        self.max_caption_tokens = max_caption_tokens
        self.n_ovt_per_object = n_ovt_per_object
        self.max_objects = max_objects
        self.ovt_token = ovt_token
        self.scene_end_token = scene_end_token
        self.thing_token = thing_token
        self.stuff_token = stuff_token
        self.rebuild_caption = rebuild_caption
        self.thing_category_ids = self._load_thing_category_ids(panoptic_categories_json)
        self.image_preprocess_mode = str(image_preprocess_mode)
        self.coda_crop_size = int(coda_crop_size)
        if self.image_preprocess_mode not in {"default", "coda_center_crop"}:
            raise ValueError(f"Unknown image_preprocess_mode={self.image_preprocess_mode}")

        self.ovt_token_id = tokenizer.convert_tokens_to_ids(ovt_token)
        self.scene_end_token_id = tokenizer.convert_tokens_to_ids(scene_end_token)
        self.thing_token_id = tokenizer.convert_tokens_to_ids(thing_token)
        self.stuff_token_id = tokenizer.convert_tokens_to_ids(stuff_token)
        assert self.ovt_token_id != tokenizer.unk_token_id, f"{ovt_token} not in tokenizer"
        assert self.scene_end_token_id != tokenizer.unk_token_id, f"{scene_end_token} not in tokenizer"
        assert self.thing_token_id != tokenizer.unk_token_id, f"{thing_token} not in tokenizer"
        assert self.stuff_token_id != tokenizer.unk_token_id, f"{stuff_token} not in tokenizer"

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
        mask_img = Image.open(mask_path).convert("RGB")
        if self.image_preprocess_mode == "coda_center_crop":
            mask_img = _coda_center_crop_image(
                mask_img,
                self.coda_crop_size,
                resample=_pil_resample("NEAREST"),
            )
        rgb = np.array(mask_img)  # (H, W, 3)
        seg_id = (
            rgb[..., 0].astype(np.int64)
            + rgb[..., 1].astype(np.int64) * 256
            + rgb[..., 2].astype(np.int64) * 256 * 256
        )
        return seg_id  # (H, W)

    def _segment_mask(self, seg_id_map: np.ndarray, segment_id: int) -> torch.Tensor:
        mask = torch.from_numpy(seg_id_map == segment_id)
        return mask  # bool (H, W)

    def _decode_coco_mask(self, sample: dict, segment: dict) -> torch.Tensor:
        """Decode an official COCO polygon/RLE and apply the exact CODA crop."""
        try:
            from pycocotools import mask as mask_utils
        except ImportError as exc:
            raise ImportError(
                "COCO-instance PGOT data requires pycocotools in the training environment."
            ) from exc

        height, width = int(sample["height"]), int(sample["width"])
        segmentation = segment["segmentation"]
        if isinstance(segmentation, list):
            rle = mask_utils.merge(mask_utils.frPyObjects(segmentation, height, width))
        elif isinstance(segmentation, dict) and isinstance(segmentation.get("counts"), list):
            rle = mask_utils.frPyObjects(segmentation, height, width)
        else:
            rle = segmentation
        decoded = mask_utils.decode(rle)
        if decoded.ndim == 3:
            decoded = decoded.any(axis=2)
        mask_img = Image.fromarray(decoded.astype(np.uint8) * 255, mode="L")
        if self.image_preprocess_mode == "coda_center_crop":
            mask_img = _coda_center_crop_image(
                mask_img,
                self.coda_crop_size,
                resample=_pil_resample("NEAREST"),
            )
        return torch.from_numpy(np.asarray(mask_img, dtype=np.uint8) > 0)

    def _prepare_coco_instance_segments(self, sample: dict):
        """Decode, remove crop-invisible instances, and order by visible area."""
        prepared = []
        for segment in sample["segments"]:
            mask = self._decode_coco_mask(sample, segment)
            visible_area = int(mask.sum().item())
            if visible_area <= 0:
                continue
            ys, xs = mask.nonzero(as_tuple=True)
            enriched = dict(segment)
            enriched["is_thing"] = True
            enriched["visible_area"] = visible_area
            enriched["crop_x"] = int(xs.min().item())
            enriched["crop_y"] = int(ys.min().item())
            prepared.append((enriched, mask))
        prepared.sort(
            key=lambda item: (
                -int(item[0]["visible_area"]),
                int(item[0]["crop_x"]),
                int(item[0]["crop_y"]),
                int(item[0].get("ann_id", 0)),
            )
        )
        if len(prepared) > self.max_objects:
            raise RuntimeError(
                f"image_id={sample.get('image_id')} has {len(prepared)} visible instances "
                f"after crop, exceeding max_objects={self.max_objects}; regenerate the "
                "manifest rather than leaking overflow objects into registers."
            )
        return [x[0] for x in prepared], [x[1] for x in prepared]

    def _build_caption_with_ovt(self, segments: List[dict]) -> Tuple[str, int]:
        """Build complete object chunks and never truncate through an object.

        The returned text always ends in ``<scene_end>``.  If the token budget
        is exhausted, the last *whole* ``marker + caption + OVT`` chunk is
        removed, keeping OVT positions and GT masks exactly aligned.
        """
        from collections import defaultdict
        cat_counter = defaultdict(int)
        parts = []
        ovt_str = self.ovt_token * self.n_ovt_per_object
        scene_end_ids = self.tokenizer.encode(
            f" {self.scene_end_token}", add_special_tokens=False
        )
        used = 0
        for seg in segments[: self.max_objects]:
            cat = seg["category"]
            cat_counter[cat] += 1
            cat_label = f"{cat.capitalize()} {cat_counter[cat]}"
            desc = seg["description"].rstrip(".").rstrip()
            marker = (
                self.thing_token
                if bool(seg.get("is_thing", False))
                or int(seg.get("category_id", -1)) in self.thing_category_ids
                else self.stuff_token
            )
            chunk = f"{marker} {cat_label}: {desc}. {ovt_str}."
            candidate = " ".join(parts + [chunk])
            candidate_ids = self.tokenizer.encode(candidate, add_special_tokens=False)
            if len(candidate_ids) + len(scene_end_ids) > self.max_caption_tokens:
                break
            parts.append(chunk)
            used += 1
        text = " ".join(parts) + f" {self.scene_end_token}"
        return text, used

    # --------------------------------------------------------------
    def __getitem__(self, idx: int) -> Dict:
        sample = self.samples[idx]
        is_coco_instance = str(sample.get("dataset_type", "")).lower() == "coco_instance"

        # 1) Image
        image = Image.open(sample["image_path"]).convert("RGB")
        if self.image_preprocess_mode == "coda_center_crop":
            image = _coda_center_crop_image(image, self.coda_crop_size)
        image_tensor = self.image_processor.preprocess(image, return_tensors="pt")["pixel_values"][0]
        target_image_tensor = self.target_image_processor.preprocess(image, return_tensors="pt")["pixel_values"][0]

        # Decode/sort instance masks before caption construction so caption,
        # OVT positions, and GT masks share exactly the same visible-area order.
        coco_masks = None
        if is_coco_instance:
            source_segments, coco_masks = self._prepare_coco_instance_segments(sample)
        else:
            source_segments = sample["segments"]

        # 2) Caption
        if self.rebuild_caption:
            caption_text, n_segments_in_caption = self._build_caption_with_ovt(
                source_segments
            )
        else:
            caption_text = sample["caption"]
            n_segments_in_caption = min(len(source_segments), self.max_objects)
        # Tokenize WITHOUT special tokens — we don't want BOS/EOS inserted here
        token_ids = self.tokenizer.encode(caption_text, add_special_tokens=False)

        # 3) Rebuilt captions are packed by whole object chunks above.  The
        # legacy non-rebuilt path retains a hard cap for backward compatibility.
        segments = source_segments[:n_segments_in_caption]
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
        seg_id_map = None
        if not is_coco_instance:
            seg_id_map = self._load_panoptic_id_map(sample["panoptic_mask_path"])
        gt_masks_per_ovt = torch.zeros((n_ovt_max, self.grid_size * self.grid_size), dtype=torch.float32)
        gt_rae_masks_per_ovt = torch.zeros(
            (n_ovt_max, self.rae_grid_size * self.rae_grid_size),
            dtype=torch.float32,
        )
        ovt_is_thing = torch.zeros(n_ovt_max, dtype=torch.bool)
        n_obj_actual = n_keep // self.n_ovt_per_object
        for obj_idx in range(n_obj_actual):
            seg_info = segments[obj_idx]
            if is_coco_instance:
                mask_hw = coco_masks[obj_idx].float()
            else:
                mask_hw = self._segment_mask(seg_id_map, int(seg_info["segment_id"])).float()
            patch_mask = _mask_to_patch_mask(mask_hw, self.grid_size)
            # E4 binding targets are built directly from the cropped binary
            # mask at the DiT/RAE 16x16 grid. Area resampling preserves the
            # fractional ownership of boundary cells and does not route a
            # predicted 32x32 attention map back into reconstruction.
            rae_patch_mask = _mask_to_patch_mask(
                mask_hw,
                self.rae_grid_size,
                mode="area",
            )
            is_thing = bool(seg_info.get("is_thing", False)) or (
                int(seg_info.get("category_id", -1)) in self.thing_category_ids
            )
            for j in range(self.n_ovt_per_object):
                ovt_idx = obj_idx * self.n_ovt_per_object + j
                gt_masks_per_ovt[ovt_idx] = patch_mask
                gt_rae_masks_per_ovt[ovt_idx] = rae_patch_mask
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
            "gt_rae_masks_per_ovt": gt_rae_masks_per_ovt,
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
            "gt_rae_masks_per_ovt": torch.stack(
                [inst["gt_rae_masks_per_ovt"] for inst in instances]
            ),
            "image_ids": [inst["image_id"] for inst in instances],
            "n_objects_list": [inst["n_objects"] for inst in instances],
            "caption_texts": [inst["caption_text"] for inst in instances],
        }
