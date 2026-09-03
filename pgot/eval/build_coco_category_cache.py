"""Build CODA-style COCO *category*-level GT masks for mBO^c / mIoU^c.

The instance cache (``coco_inst_mask_cache_coda512``) labels every COCO
instance separately, which is what mBO^i / mIoU^i measure.  This script writes
the same on-disk layout but merges every instance of the same thing category
into a single GT region, which is what mBO^c / mIoU^c measure.

Because the layout matches, ``run_eval`` needs no new GT code path: point
``--coco_mask_cache`` at the directory produced here.

Overlap handling follows the instance convention, with one deliberate change:
a pixel is an overlap pixel only when *two different categories* cover it.
Same-category instance overlaps disappear once the instances are merged.
"""

import argparse
import json
import shutil
from pathlib import Path

import cv2
import numpy as np
from pycocotools.coco import COCO
from tqdm import tqdm


def preprocess_coda(mask: np.ndarray, size: int) -> np.ndarray:
    """Resize min side to ``size`` then center crop. Mirrors the instance cache."""
    h, w = mask.shape[:2]
    factor = max(size / h, size / w)
    rh, rw = int(round(h * factor)), int(round(w * factor))
    mask = cv2.resize(mask, (rw, rh), interpolation=cv2.INTER_NEAREST)
    top = max((rh - size) // 2, 0)
    left = max((rw - size) // 2, 0)
    return mask[top:top + size, left:left + size]


def relabel_contiguous(label: np.ndarray) -> np.ndarray:
    """Renumber surviving regions to 1..C so empty labels cannot score as 0.

    The center crop can remove a category entirely.  ``mbo_metric`` one-hots the
    GT map over 0..max, so a missing intermediate label would become an all-zero
    GT column and be averaged in as IoU 0.  The instance cache is already
    contiguous; keep the same invariant here.
    """
    present = np.unique(label)
    present = present[present > 0]
    if present.size == 0:
        return np.zeros_like(label)
    remap = np.zeros(int(present.max()) + 1, dtype=np.uint16)
    remap[present] = np.arange(1, present.size + 1, dtype=np.uint16)
    return remap[label]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source_cache",
        default="/home/jovyan/PGOT/data/coco_inst_mask_cache_coda512",
        help="Instance cache that fixes the image set, ordering and crop size.",
    )
    parser.add_argument(
        "--output_dir",
        default="/home/jovyan/PGOT/data/coco_cat_mask_cache_coda512",
    )
    parser.add_argument(
        "--instances_json",
        default="/home/jovyan/data/coco/annotations/instances_val2017.json",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    src = Path(args.source_cache)
    out = Path(args.output_dir)
    if out.exists() and not args.overwrite:
        raise SystemExit(f"Already exists (use --overwrite): {out}")
    out.mkdir(parents=True, exist_ok=True)

    image_ids = np.load(src / "image_ids.npy")
    with (src / "meta.json").open() as f:
        src_meta = json.load(f)
    size = int(src_meta["size"])
    n_images = len(image_ids)
    print(f"source cache: {src} | images={n_images} size={size}")

    coco = COCO(args.instances_json)

    masks_out = np.lib.format.open_memmap(
        out / "masks.npy", mode="w+", dtype=np.uint16, shape=(n_images, size, size)
    )
    overlap_out = np.lib.format.open_memmap(
        out / "overlap_masks.npy", mode="w+", dtype=np.uint8, shape=(n_images, size, size)
    )
    counts = np.zeros(n_images, dtype=np.int32)

    for idx, image_id in enumerate(tqdm(image_ids, desc="COCO category masks")):
        info = coco.loadImgs([int(image_id)])[0]
        height, width = int(info["height"]), int(info["width"])

        per_category: dict[int, np.ndarray] = {}
        ann_ids = coco.getAnnIds(imgIds=[int(image_id)], iscrowd=False)
        for ann in coco.loadAnns(ann_ids):
            if ann.get("ignore", False) or ann.get("area", 0) <= 0:
                continue
            cat_id = int(ann["category_id"])
            binary = coco.annToMask(ann).astype(bool)
            if cat_id in per_category:
                per_category[cat_id] |= binary
            else:
                per_category[cat_id] = binary

        label = np.zeros((height, width), dtype=np.uint16)
        coverage = np.zeros((height, width), dtype=np.uint8)
        for order, cat_id in enumerate(sorted(per_category)):
            binary = per_category[cat_id]
            label[binary] = order + 1
            coverage += binary.astype(np.uint8)

        label_cropped = relabel_contiguous(preprocess_coda(label, size))
        masks_out[idx] = label_cropped
        overlap_out[idx] = preprocess_coda((coverage > 1).astype(np.uint8), size)
        counts[idx] = int(np.unique(label_cropped[label_cropped > 0]).size)

    masks_out.flush()
    overlap_out.flush()
    np.save(out / "image_ids.npy", image_ids)
    np.save(out / "gt_counts.npy", counts)
    if (src / "file_names.json").exists():
        shutil.copy(src / "file_names.json", out / "file_names.json")

    meta = dict(src_meta)
    meta.update(
        {
            "gt_granularity": "category",
            "gt_source_annotations": str(Path(args.instances_json).name),
            "note": (
                "Thing-category masks: all instances of one COCO thing category "
                "share a label. Use for mBO^c / mIoU^c. Overlap pixels are those "
                "covered by two or more different categories."
            ),
        }
    )
    with (out / "meta.json").open("w") as f:
        json.dump(meta, f, indent=2)

    print(f"Saved: {out}")
    print(f"GT regions per image: mean={counts.mean():.2f} max={counts.max()}")


if __name__ == "__main__":
    main()
