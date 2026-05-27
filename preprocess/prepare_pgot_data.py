"""
Convert Pix2Cap-COCO annotations into PGOT training samples.

Each output line (JSONL) contains everything needed to construct a training sample:
- image_path
- panoptic_mask_path (PNG file with RGB-encoded segment IDs)
- segments: list of dicts {segment_id, category, description, area, bbox}
- pgot_caption: formatted caption with <ovt><ovt> per object + <scene_end>

Token-level <ovt> position extraction is deferred to the dataloader (depends on tokenizer).

Usage:
    # Smoke test (10 samples each):
    python prepare_pgot_data.py --smoke

    # Full preprocessing:
    python prepare_pgot_data.py
"""
import argparse
import json
import os
import random
from collections import defaultdict
from tqdm import tqdm

PIX2CAP_TRAIN = "/home/jovyan/data/coco/pix2cap_coco_train.json"
PIX2CAP_VAL = "/home/jovyan/data/coco/pix2cap_coco_val.json"
PANOPTIC_DIR_TRAIN = "/home/jovyan/data/coco/annotations/panoptic_train2017"
PANOPTIC_DIR_VAL = "/home/jovyan/data/coco/annotations/panoptic_val2017"
PANOPTIC_JSON_TRAIN = "/home/jovyan/data/coco/annotations/panoptic_train2017.json"
PANOPTIC_JSON_VAL = "/home/jovyan/data/coco/annotations/panoptic_val2017.json"
IMAGE_DIR_TRAIN = "/home/jovyan/data/coco/train2017"
IMAGE_DIR_VAL = "/home/jovyan/data/coco/val2017"
OUTPUT_DIR = "/home/jovyan/PGOT/data"

N_OVT_PER_OBJECT = 2
OVT_TOKEN = "<ovt>"
SCENE_END_TOKEN = "<scene_end>"
MIN_SEGMENT_AREA = 0  # 0 = no filtering. Set to e.g. 100 to drop tiny segments.
MAX_SEGMENTS_PER_IMAGE = 50  # cap to prevent runaway captions


def find_annotations(data):
    """Pix2Cap JSON might use different top-level keys; try common ones."""
    if isinstance(data, list):
        return data
    for k in ['annotations', 'data', 'images']:
        if k in data and isinstance(data[k], list) and len(data[k]) > 0:
            sample = data[k][0]
            if 'segments_info' in sample:
                return data[k]
    raise ValueError(f"Could not locate annotation list. Top keys: {list(data.keys())[:10] if isinstance(data, dict) else 'list'}")


def build_pgot_caption(segments_info, n_ovt=N_OVT_PER_OBJECT):
    """
    Build "Person 1: ... <ovt><ovt>. Giraffe 1: ... <ovt><ovt>. ... <scene_end>"

    Returns:
        caption: str
        used_segments: list of dicts in order they appear in the caption
                       (each: {segment_id, category, category_index, description, area, bbox})
    """
    # Sort by area descending so big objects come first (better gradient signal)
    valid = []
    for s in segments_info:
        desc = (s.get('description') or '').strip()
        if not desc:
            continue
        area = int(s.get('area', 0))
        if area < MIN_SEGMENT_AREA:
            continue
        valid.append({
            'segment_id': int(s['id']),
            'category': str(s.get('category', f"cat_{s.get('category_id')}")),
            'category_id': s.get('category_id'),
            'description': desc,
            'area': area,
            'bbox': s.get('bbox', []),
        })

    valid.sort(key=lambda x: -x['area'])
    valid = valid[:MAX_SEGMENTS_PER_IMAGE]

    if not valid:
        return None, []

    # Category counter for "Person 1", "Person 2", ...
    cat_counter = defaultdict(int)
    parts = []
    used = []
    ovt_str = OVT_TOKEN * n_ovt

    for v in valid:
        cat_counter[v['category']] += 1
        cat_idx = cat_counter[v['category']]
        cat_label = f"{v['category'].capitalize()} {cat_idx}"
        desc = v['description'].rstrip('.').rstrip()
        parts.append(f"{cat_label}: {desc}. {ovt_str}.")

        v['category_index'] = cat_idx
        used.append(v)

    caption = " ".join(parts) + f" {SCENE_END_TOKEN}"
    return caption, used


def process_split(pix2cap_json, panoptic_dir, image_dir, output_path, split_name, smoke=False):
    print(f"\n[{split_name}] Loading {pix2cap_json}...")
    with open(pix2cap_json) as f:
        data = json.load(f)
    items = find_annotations(data)
    print(f"[{split_name}] Loaded {len(items)} entries.")

    if smoke:
        random.seed(42)
        items = random.sample(items, min(10, len(items)))
        print(f"[{split_name}] SMOKE MODE: using {len(items)} samples.")

    out_samples = []
    skipped_no_img = 0
    skipped_no_mask = 0
    skipped_no_segs = 0
    skipped_no_desc = 0

    total_n_objs = 0
    cap_lens = []

    for ann in tqdm(items, desc=f"Processing {split_name}"):
        image_id = ann.get('image_id')
        file_name = ann.get('file_name')
        if file_name is None and image_id is not None:
            file_name = f"{image_id:012d}.png"

        if image_id is None or file_name is None:
            skipped_no_img += 1
            continue

        image_path = os.path.join(image_dir, f"{image_id:012d}.jpg")
        mask_path = os.path.join(panoptic_dir, file_name)

        if not os.path.exists(image_path):
            skipped_no_img += 1
            continue
        if not os.path.exists(mask_path):
            skipped_no_mask += 1
            continue

        segments_info = ann.get('segments_info', [])
        if not segments_info:
            skipped_no_segs += 1
            continue

        caption, used = build_pgot_caption(segments_info)
        if caption is None:
            skipped_no_desc += 1
            continue

        sample = {
            'image_id': image_id,
            'image_path': image_path,
            'panoptic_mask_path': mask_path,
            'caption': caption,
            'segments': used,
            'n_objects': len(used),
        }
        out_samples.append(sample)
        total_n_objs += len(used)
        cap_lens.append(len(caption))

    print(f"\n[{split_name}] DONE")
    print(f"  Output: {len(out_samples)} samples")
    print(f"  Skipped (no image):  {skipped_no_img}")
    print(f"  Skipped (no mask):   {skipped_no_mask}")
    print(f"  Skipped (no segs):   {skipped_no_segs}")
    print(f"  Skipped (no desc):   {skipped_no_desc}")
    if out_samples:
        print(f"  Avg objects/image:   {total_n_objs/len(out_samples):.2f}")
        print(f"  Caption length: avg={sum(cap_lens)/len(cap_lens):.0f}, "
              f"max={max(cap_lens)}, min={min(cap_lens)}")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        for s in out_samples:
            f.write(json.dumps(s, ensure_ascii=False) + '\n')
    print(f"  Wrote {output_path}")

    # Print first 3 samples for verification
    print(f"\n[{split_name}] First {min(3, len(out_samples))} samples:")
    for i, s in enumerate(out_samples[:3]):
        print(f"\n--- Sample {i} ---")
        print(f"  image_id: {s['image_id']}")
        print(f"  image_path: {s['image_path']}")
        print(f"  mask_path: {s['panoptic_mask_path']}")
        print(f"  n_objects: {s['n_objects']}")
        cap = s['caption']
        if len(cap) > 800:
            cap = cap[:800] + "...(truncated)"
        print(f"  caption: {cap}")
        if s['segments']:
            seg = s['segments'][0]
            print(f"  segment[0]: id={seg['segment_id']} cat={seg['category']} "
                  f"idx={seg['category_index']} area={seg['area']}")

    return out_samples


def verify_mask_loading(samples, panoptic_json_path):
    """Smoke check: load 1 panoptic PNG and verify segment IDs match annotation."""
    if not samples:
        return
    from PIL import Image
    import numpy as np

    print(f"\n{'='*60}")
    print("VERIFYING PANOPTIC MASK LOADING")
    print('='*60)

    # Build segment_id lookup from panoptic json (RGB -> ID encoding)
    s = samples[0]
    mask = np.array(Image.open(s['panoptic_mask_path']).convert('RGB'))
    # COCO panoptic: segment_id = R + G*256 + B*256^2
    rgb2id = mask[:, :, 0].astype(np.int64) + \
             mask[:, :, 1].astype(np.int64) * 256 + \
             mask[:, :, 2].astype(np.int64) * 256 * 256
    unique_ids = set(np.unique(rgb2id).tolist())
    expected_ids = {seg['segment_id'] for seg in s['segments']}

    print(f"Sample image_id: {s['image_id']}")
    print(f"Mask shape: {mask.shape}")
    print(f"Unique segment IDs in mask: {len(unique_ids)}")
    print(f"Expected segment IDs (from pix2cap): {len(expected_ids)}")
    overlap = unique_ids & expected_ids
    print(f"Overlap: {len(overlap)} / {len(expected_ids)} expected")

    if len(overlap) == len(expected_ids):
        print("✅ All expected segment IDs found in panoptic mask.")
    else:
        missing = expected_ids - unique_ids
        print(f"⚠️ Missing {len(missing)} segment IDs: {list(missing)[:5]}")
        print("    This might be ok if pix2cap filtered some segments.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--smoke', action='store_true',
                        help='Process only 10 samples per split (fast sanity check)')
    parser.add_argument('--skip_val', action='store_true')
    parser.add_argument('--skip_train', action='store_true')
    args = parser.parse_args()

    train_out = os.path.join(OUTPUT_DIR, "pgot_train_smoke.jsonl" if args.smoke else "pgot_train.jsonl")
    val_out = os.path.join(OUTPUT_DIR, "pgot_val_smoke.jsonl" if args.smoke else "pgot_val.jsonl")

    if not args.skip_train:
        train_samples = process_split(
            PIX2CAP_TRAIN, PANOPTIC_DIR_TRAIN, IMAGE_DIR_TRAIN,
            train_out, "train", smoke=args.smoke
        )
        verify_mask_loading(train_samples, PANOPTIC_JSON_TRAIN)

    if not args.skip_val:
        val_samples = process_split(
            PIX2CAP_VAL, PANOPTIC_DIR_VAL, IMAGE_DIR_VAL,
            val_out, "val", smoke=args.smoke
        )
        verify_mask_loading(val_samples, PANOPTIC_JSON_VAL)

    print(f"\n{'='*60}")
    print("ALL DONE")
    print('='*60)


if __name__ == '__main__':
    main()
