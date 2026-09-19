"""NaN-safe VOC metric aggregation by GT instance count (no GPU needed)."""

import argparse
import csv
import json
import math
from pathlib import Path


METRICS = ("fARI", "mBO", "mIoU", "sMBO", "sMIOU")
BUCKETS = ("0", "1-3", "4-6", "7-10", "11+")


def count_bucket(count):
    if count < 0:
        raise ValueError(f"Negative GT object count: {count}")
    return "0" if count == 0 else "1-3" if count <= 3 else "4-6" if count <= 6 else "7-10" if count <= 10 else "11+"


def _summarize(rows):
    result = {"n": len(rows)}
    for metric in METRICS:
        values = [row.get(metric) for row in rows]
        values = [value for value in values if isinstance(value, (int, float)) and math.isfinite(value)]
        result[metric] = sum(values) / len(values) if values else None
        result[f"{metric}_valid_n"] = len(values)
    if rows and all("kept_slots" in row for row in rows):
        result["kept_slots_mean"] = sum(row["kept_slots"] for row in rows) / len(rows)
    return result


def summarize_object_counts(records):
    """Bucket using cropped/ignore-filtered GT instances, not predicted slots."""
    grouped = {}
    exact = {}
    for row in records:
        count = int(row["gt_objects"])
        grouped.setdefault(count_bucket(count), []).append(row)
        exact.setdefault(count, []).append(row)
    return {
        "object_count_source": "GT instances after VOC center crop and 255 ignore filtering; background excluded",
        "by_object_count": {key: _summarize(grouped[key]) for key in BUCKETS if key in grouped},
        "by_exact_object_count": {str(key): _summarize(exact[key]) for key in sorted(exact)},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_root", default="/home/jovyan/PGOT/outputs/baselines_voc_zeroshot")
    parser.add_argument("--models", nargs="+", default=["coda", "spot", "metaslot", "adaslot"])
    args = parser.parse_args()
    root = Path(args.output_root)
    combined, flat = {}, []
    reference = None
    for name in args.models:
        path = root / name / "per_image.jsonl"
        if not path.is_file():
            raise FileNotFoundError(f"Missing per-image results: {path}; select only evaluated --models")
        records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        sample_counts = {row["id"]: int(row["gt_objects"]) for row in records}
        if len(sample_counts) != len(records):
            raise ValueError(f"Duplicate image IDs in {path}")
        if reference is not None and sample_counts != reference:
            raise ValueError("Models do not share identical image IDs/GT object counts; do not compare smoke to full results")
        reference = sample_counts
        combined[name] = summarize_object_counts(records)
        for group, scores in combined[name]["by_object_count"].items():
            flat.append({"model": name, "gt_objects": group, "n": scores["n"],
                         **{key: scores[key] for key in METRICS}})
    with (root / "object_count_comparison.json").open("w") as handle:
        json.dump(combined, handle, indent=2, allow_nan=False)
    with (root / "object_count_comparison.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["model", "gt_objects", "n", *METRICS])
        writer.writeheader()
        writer.writerows(flat)
    print(f"Saved {root / 'object_count_comparison.csv'} (metric scale 0..1)")


if __name__ == "__main__":
    main()
