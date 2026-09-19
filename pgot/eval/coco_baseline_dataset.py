"""Raw COCO images with exactly the existing PGOT CODA-style GT caches."""
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

class COCOBaselineDataset(Dataset):
    def __init__(self, root, instance_cache, category_cache, eval_size=512, max_samples=None):
        self.root = Path(root)
        self.instance_root, self.category_root = Path(instance_cache), Path(category_cache)
        self.instance, self.instance_overlap, ids, instance_meta = self._load(self.instance_root)
        self.category, self.category_overlap, category_ids, category_meta = self._load(self.category_root)
        if not np.array_equal(ids, category_ids):
            raise ValueError("Instance/category cache image IDs differ")
        if instance_meta.get("mode") != "coda" or category_meta.get("gt_granularity") != "category":
            raise ValueError("Expected CODA-style instance and thing-category caches")
        if self.instance.shape[1:] != (eval_size, eval_size) or self.category.shape != self.instance.shape:
            raise ValueError("Cache resolution does not match --eval_size")
        self.ids = ids[:max_samples].tolist() if max_samples else ids.tolist()

    @staticmethod
    def _load(root):
        meta = json.loads((root / "meta.json").read_text())
        masks = np.load(root / "masks.npy", mmap_mode="r")
        overlap = np.load(root / "overlap_masks.npy", mmap_mode="r")
        ids = np.load(root / "image_ids.npy")
        if masks.shape != overlap.shape or len(ids) != len(masks) or len(set(ids.tolist())) != len(ids):
            raise ValueError(f"Invalid cache: {root}")
        return masks, overlap, ids, meta

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        image_id = int(self.ids[index])
        image = Image.open(self.root / "val2017" / f"{image_id:012d}.jpg").convert("RGB")
        instance, category = self.instance[index].copy(), self.category[index].copy()
        overlap, category_overlap = self.instance_overlap[index].copy(), self.category_overlap[index].copy()
        # Preserve cache labels exactly: PGOT metrics handle overlap internally.
        # Reindexing here would change mBO/mIoU denominators for empty label IDs.
        return {
            "id": image_id, "image": np.asarray(image, dtype=np.uint8).copy(),
            "instance": torch.from_numpy(instance.astype(np.int64)),
            "semantic": torch.from_numpy(category.astype(np.int64)),
            "overlap": torch.from_numpy(overlap.astype(np.int64)),
            "semantic_overlap": torch.from_numpy(category_overlap.astype(np.int64)),
        }
