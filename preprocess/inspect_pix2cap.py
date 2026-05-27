"""
Inspect Pix2Cap-COCO JSON structure to verify field names and sample format.
Run this FIRST before prepare_pgot_data.py to confirm assumptions.

Usage:
    python inspect_pix2cap.py
"""
import json
import os
from collections import Counter

PIX2CAP_TRAIN = "/home/jovyan/data/coco/pix2cap_coco_train.json"
PIX2CAP_VAL = "/home/jovyan/data/coco/pix2cap_coco_val.json"
PANOPTIC_JSON_TRAIN = "/home/jovyan/data/coco/annotations/panoptic_train2017.json"
PANOPTIC_DIR_TRAIN = "/home/jovyan/data/coco/annotations/panoptic_train2017"
IMAGE_DIR_TRAIN = "/home/jovyan/data/coco/train2017"


def inspect_json(path, name):
    print(f"\n{'='*60}")
    print(f"INSPECTING: {name}")
    print(f"PATH: {path}")
    print(f"SIZE: {os.path.getsize(path) / 1024 / 1024:.1f} MB")
    print('='*60)

    with open(path) as f:
        data = json.load(f)

    if isinstance(data, dict):
        print(f"\nTop-level keys: {list(data.keys())}")
        for k, v in data.items():
            if isinstance(v, list):
                print(f"  - {k}: list of {len(v)} items")
            elif isinstance(v, dict):
                print(f"  - {k}: dict with keys {list(v.keys())[:5]}")
            else:
                print(f"  - {k}: {type(v).__name__}")

        # Find the annotations-like list
        for candidate in ['annotations', 'data', 'images']:
            if candidate in data and isinstance(data[candidate], list) and len(data[candidate]) > 0:
                items = data[candidate]
                print(f"\n[Sample of '{candidate}'] - 1 entry:")
                sample = items[0]
                print(json.dumps(sample, indent=2)[:2000])
                if len(json.dumps(sample)) > 2000:
                    print("... (truncated)")

                # Stats
                print(f"\n[Stats over '{candidate}']:")
                print(f"  Total entries: {len(items)}")
                if 'segments_info' in sample:
                    seg_counts = [len(it.get('segments_info', [])) for it in items[:5000]]
                    print(f"  Avg segments per entry (first 5000): {sum(seg_counts)/len(seg_counts):.2f}")
                    print(f"  Max: {max(seg_counts)}, Min: {min(seg_counts)}")

                    # Categories
                    cats = Counter()
                    desc_lens = []
                    desc_empty = 0
                    for it in items[:1000]:
                        for s in it.get('segments_info', []):
                            cat = s.get('category', s.get('category_id'))
                            cats[cat] += 1
                            desc = s.get('description', '')
                            if desc:
                                desc_lens.append(len(desc))
                            else:
                                desc_empty += 1
                    print(f"  Top 10 categories (1000 entries): {cats.most_common(10)}")
                    if desc_lens:
                        print(f"  Description length: avg={sum(desc_lens)/len(desc_lens):.0f}, "
                              f"max={max(desc_lens)}, min={min(desc_lens)}")
                    print(f"  Empty descriptions: {desc_empty}")
                break
    elif isinstance(data, list):
        print(f"\nList of {len(data)} items.")
        print(f"\n[Sample - 1 entry]:")
        print(json.dumps(data[0], indent=2)[:2000])


def check_panoptic_alignment():
    """Verify pix2cap entries align with panoptic PNG masks."""
    print(f"\n{'='*60}")
    print("CHECKING PIX2CAP <-> PANOPTIC MASK ALIGNMENT")
    print('='*60)

    with open(PIX2CAP_TRAIN) as f:
        p2c = json.load(f)

    # Get annotation-like list
    items = None
    for k in ['annotations', 'data', 'images']:
        if k in p2c and isinstance(p2c[k], list):
            items = p2c[k]
            break
    if items is None:
        print("Could not find list field in pix2cap JSON")
        return

    sample = items[0]
    print(f"Pix2Cap sample fields: {list(sample.keys())}")

    # Try to find image_id / file_name
    image_id = sample.get('image_id')
    file_name = sample.get('file_name')
    print(f"  image_id: {image_id}")
    print(f"  file_name: {file_name}")

    # Check panoptic mask existence
    if file_name:
        mask_path = os.path.join(PANOPTIC_DIR_TRAIN, file_name)
        if not os.path.exists(mask_path) and image_id:
            mask_path = os.path.join(PANOPTIC_DIR_TRAIN, f"{image_id:012d}.png")
        print(f"  expected mask path: {mask_path}")
        print(f"  exists: {os.path.exists(mask_path)}")

        # Check image existence
        if image_id:
            img_path = os.path.join(IMAGE_DIR_TRAIN, f"{image_id:012d}.jpg")
            print(f"  expected image path: {img_path}")
            print(f"  exists: {os.path.exists(img_path)}")

    # Check first 20 entries for mask coverage
    print(f"\nChecking mask existence for 20 random entries...")
    import random
    random.seed(0)
    sampled = random.sample(items, min(20, len(items)))
    ok, miss = 0, 0
    for it in sampled:
        fn = it.get('file_name')
        if not fn:
            iid = it.get('image_id')
            fn = f"{iid:012d}.png" if iid else None
        if fn and os.path.exists(os.path.join(PANOPTIC_DIR_TRAIN, fn)):
            ok += 1
        else:
            miss += 1
    print(f"  Mask exists: {ok}/20, Missing: {miss}/20")


def check_panoptic_json():
    """Cross-check with COCO panoptic_train2017.json for segment_id <-> RGB mapping."""
    print(f"\n{'='*60}")
    print("INSPECTING COCO PANOPTIC ANNOTATIONS (for segment_id mapping)")
    print('='*60)
    with open(PANOPTIC_JSON_TRAIN) as f:
        pan = json.load(f)
    print(f"Top-level keys: {list(pan.keys())}")
    if 'annotations' in pan:
        sample = pan['annotations'][0]
        print(f"\nPanoptic annotation sample:")
        print(json.dumps(sample, indent=2)[:1500])
    if 'categories' in pan:
        print(f"\nCategories count: {len(pan['categories'])}")
        print(f"First 3 categories: {pan['categories'][:3]}")


if __name__ == '__main__':
    inspect_json(PIX2CAP_TRAIN, "Pix2Cap Train")
    inspect_json(PIX2CAP_VAL, "Pix2Cap Val")
    check_panoptic_alignment()
    check_panoptic_json()
    print("\nDone. If field names look right, proceed to prepare_pgot_data.py")
