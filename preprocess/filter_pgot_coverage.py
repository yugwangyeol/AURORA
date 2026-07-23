#!/usr/bin/env python3
"""Create the optional no-void/coverage-filtered *ablation* manifest.

This is not used by the full-data main experiment.  The main comparison keeps
all samples and toggles a residual VOID token with ``PGOT_N_VOID=1/0``.

Coverage is measured after the same CODA resize-min-side + center crop used by
training.  A sample is retained only when the union of all selected Pix2Cap
thing/stuff segments covers at least ``--min_coverage`` of the cropped image.
The original segment order (area-descending) is preserved.
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm


def _decode_panoptic(path: str, crop_size: int) -> np.ndarray:
    image = Image.open(path).convert("RGB")
    width, height = image.size
    factor = max(float(crop_size) / float(height), float(crop_size) / float(width))
    resized = (int(round(width * factor)), int(round(height * factor)))
    resampling = getattr(Image, "Resampling", Image)
    image = image.resize(resized, resampling.NEAREST)
    left = max((resized[0] - crop_size) // 2, 0)
    top = max((resized[1] - crop_size) // 2, 0)
    rgb = np.asarray(image.crop((left, top, left + crop_size, top + crop_size)))
    return (
        rgb[..., 0].astype(np.int64)
        + 256 * rgb[..., 1].astype(np.int64)
        + 256 * 256 * rgb[..., 2].astype(np.int64)
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--min_coverage", type=float, default=0.95)
    parser.add_argument("--crop_size", type=int, default=512)
    parser.add_argument("--max_objects", type=int, default=50)
    parser.add_argument("--max_samples", type=int, default=None)
    args = parser.parse_args()

    if not 0.0 <= args.min_coverage <= 1.0:
        raise ValueError("--min_coverage must be in [0, 1]")
    input_path = Path(args.input).resolve()
    output_path = Path(args.output).resolve()
    if input_path == output_path:
        raise ValueError("Input and output manifests must be different files.")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")

    total = kept = 0
    coverage_sum = 0.0
    with input_path.open() as source, temporary_path.open("w") as target:
        for line in tqdm(source, desc=f"coverage >= {args.min_coverage:.3f}"):
            if args.max_samples is not None and total >= args.max_samples:
                break
            sample = json.loads(line)
            total += 1
            segments = sample.get("segments", [])[: args.max_objects]
            if not segments:
                continue
            id_map = _decode_panoptic(sample["panoptic_mask_path"], args.crop_size)
            segment_ids = np.asarray(
                [int(segment["segment_id"]) for segment in segments], dtype=np.int64
            )
            coverage = float(np.isin(id_map, segment_ids).mean())
            coverage_sum += coverage
            if coverage + 1e-12 < args.min_coverage:
                continue
            sample["segment_coverage_coda_crop"] = coverage
            target.write(json.dumps(sample, ensure_ascii=False) + "\n")
            kept += 1

    os.replace(temporary_path, output_path)
    mean_coverage = coverage_sum / max(total, 1)
    print(
        f"Wrote {kept}/{total} samples to {output_path} "
        f"(kept={kept / max(total, 1):.3%}, input_mean_coverage={mean_coverage:.4f})"
    )


if __name__ == "__main__":
    main()
