"""COCO annotation-derived class/point queries. No Pix2Cap or mask routing."""

from collections import Counter
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from pycocotools.coco import COCO
from torch.utils.data import Dataset

from pgot.constants import OVT_TOKEN, SCENE_END_TOKEN, THING_TOKEN
from pgot.train.pgot_dataset import _coda_center_crop_image, _pil_resample


QUERY_MODES = ("class_point", "class_only", "class_point_short", "class_point_natural")


def natural_position(x, y):
    """Describe the same normalized point using fixed thirds, not extra GT facts.

    x increases left-to-right, y top-to-bottom. A boundary belongs to the
    following bin. The center-of-mass point need not lie inside its GT mask.
    """
    if not (0 <= x <= 1 and 0 <= y <= 1):
        raise ValueError("Normalized coordinates must be in [0, 1]")
    horizontal = "left" if x < 1 / 3 else "center" if x < 2 / 3 else "right"
    vertical = "upper" if y < 1 / 3 else "middle" if y < 2 / 3 else "lower"
    if horizontal == "center":
        return "center of the image" if vertical == "middle" else f"{vertical}-center part of the image"
    return f"{vertical}-{horizontal} part of the image"


def build_query_text(objects, selected, query_mode):
    if query_mode not in QUERY_MODES:
        raise ValueError("Unknown query mode")
    chunks, records, seen = [], [], Counter()
    for slot, index in enumerate(selected):
        obj = objects[int(index)]
        name = obj["category"]
        seen[name] += 1
        x, y = obj["center_normalized_xy"]
        description = f"a {name}"
        if query_mode == "class_point":
            description += (f" centered at normalized image coordinates (x={x:.4f}, y={y:.4f}),"
                            " with x measured from the left and y from the top")
        elif query_mode == "class_point_short":
            description += f" at normalized coordinates (x={x:.4f}, y={y:.4f})"
        elif query_mode == "class_point_natural":
            description += (f" in the {natural_position(x, y)},"
                            f" at normalized coordinates (x={x:.4f}, y={y:.4f})")
        chunks.append(f"{THING_TOKEN} {name.capitalize()} {seen[name]}: {description}. {OVT_TOKEN}.")
        records.append(dict(obj, query_index=slot, target_gt_index=int(index), text=description))
    return " ".join(chunks + [SCENE_END_TOKEN]), records


class BindingCocoDataset(Dataset):
    def __init__(self, root, tokenizer, image_processor, target_processor,
                 eval_size=512, max_bindings=7, max_objects=64, max_caption_tokens=4096,
                 seed=0, query_mode="class_point", ignore_overlaps=False,
                 center_source="mask", require_full_split=True):
        self.root = Path(root)
        self.coco = COCO(str(self.root / "annotations/instances_val2017.json"))
        self.ids = sorted(self.coco.imgs)
        if require_full_split and len(self.ids) != 5000:
            raise ValueError("Expected the complete official 5000-image COCO val2017 split")
        self.size, self.max_bindings, self.max_objects = eval_size, max_bindings, max_objects
        self.seed, self.query_mode = seed, query_mode
        self.ignore_overlaps, self.center_source = ignore_overlaps, center_source
        self.tokenizer, self.image_processor, self.target_processor = tokenizer, image_processor, target_processor
        self.max_caption_tokens = max_caption_tokens
        self.capacity = max_bindings if max_bindings else max_objects
        if not (0 <= max_bindings <= max_objects) or self.capacity < 1:
            raise ValueError("Require 0 <= max_bindings <= max_objects, max_objects >= 1")
        if center_source not in {"mask", "bbox"}:
            raise ValueError("Unknown center source")
        if query_mode not in QUERY_MODES:
            raise ValueError("Unknown query mode")

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        iid = int(self.ids[index])
        info = self.coco.imgs[iid]
        image_path = self.root / "val2017" / info["file_name"]
        with Image.open(image_path) as original:
            image = _coda_center_crop_image(original.convert("RGB"), self.size)
        annotations = sorted(self.coco.loadAnns(self.coco.getAnnIds(imgIds=[iid], iscrowd=False)),
                             key=lambda ann: int(ann["id"]))
        masks, objects = [], []
        for ann in annotations:
            if ann.get("ignore", False) or ann.get("area", 0) <= 0:
                continue
            mask = Image.fromarray(self.coco.annToMask(ann).astype(np.uint8))
            mask = np.asarray(_coda_center_crop_image(mask, self.size, resample=_pil_resample("NEAREST"))) > 0
            if mask.any():
                masks.append(mask)
                objects.append({"annotation_id": int(ann["id"]), "category_id": int(ann["category_id"]),
                                "category": self.coco.cats[ann["category_id"]]["name"]})
        targets = np.stack(masks) if masks else np.zeros((0, self.size, self.size), dtype=bool)
        ignore = targets.sum(axis=0) > 1 if self.ignore_overlaps else np.zeros((self.size, self.size), dtype=bool)
        keep = [i for i, mask in enumerate(targets) if (mask & ~ignore).any()]
        targets, objects = targets[keep], [objects[i] for i in keep]
        for obj, mask in zip(objects, targets):
            yy, xx = np.nonzero(mask & ~ignore)
            if self.center_source == "mask":
                x, y = xx.mean() + .5, yy.mean() + .5
            else:
                x, y = (xx.min() + xx.max() + 1) / 2, (yy.min() + yy.max() + 1) / 2
            obj["center_normalized_xy"] = [float(x / self.size), float(y / self.size)]
        n = len(objects)
        if self.max_bindings == 0 and n > self.max_objects:
            raise ValueError(f"Image {iid} has {n} targets > max_objects={self.max_objects}; do not silently truncate")
        count = min(n, self.capacity)
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, iid]))
        selected = rng.permutation(n)[:count].tolist()
        text, queries = build_query_text(objects, selected, self.query_mode)
        ids = torch.tensor(self.tokenizer.encode(text, add_special_tokens=False), dtype=torch.long)
        if ids.numel() > self.max_caption_tokens:
            raise ValueError(f"Image {iid} query caption exceeds token limit; never truncate query/target pairs")
        positions = (ids == self.tokenizer.convert_tokens_to_ids(OVT_TOKEN)).nonzero().flatten()
        if positions.numel() != count:
            raise ValueError("Query-to-OVT count mismatch")
        pos, valid = torch.zeros(self.capacity, dtype=torch.long), torch.zeros(self.capacity, dtype=torch.bool)
        pos[:count], valid[:count] = positions, True
        return {"image": self.image_processor.preprocess(image, return_tensors="pt")["pixel_values"][0],
                "target_image": self.target_processor.preprocess(image, return_tensors="pt")["pixel_values"][0],
                "caption_ids": ids, "ovt_positions": pos, "ovt_valid": valid,
                "image_id": iid, "image_path": str(image_path), "targets": targets,
                "ignore": ignore, "objects": objects, "selected": selected,
                "queries": queries, "caption": text}


class BindingCollator:
    def __init__(self, pad_token_id):
        self.pad = pad_token_id

    def __call__(self, samples):
        length = max(s["caption_ids"].numel() for s in samples)
        ids = torch.full((len(samples), length), self.pad, dtype=torch.long)
        attention = torch.zeros_like(ids, dtype=torch.bool)
        for i, sample in enumerate(samples):
            n = sample["caption_ids"].numel()
            ids[i, :n], attention[i, :n] = sample["caption_ids"], True
        return {"images": torch.stack([s["image"] for s in samples]),
                "target_images": torch.stack([s["target_image"] for s in samples]),
                "caption_input_ids": ids, "caption_attention_mask": attention,
                "ovt_positions_in_caption": torch.stack([s["ovt_positions"] for s in samples]),
                "ovt_valid_mask": torch.stack([s["ovt_valid"] for s in samples]), "samples": samples}
