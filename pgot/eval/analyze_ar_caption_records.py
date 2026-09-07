"""Offline diagnostics for autoregressive PGOT caption records.

This reads the JSONL emitted by ``run_eval --caption_mode autoregressive`` and
the aligned validation JSONL.  It does not load a model or rerun inference.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


OBJECT_RE = re.compile(
    r"<(?:thing|stuff)>\s*([^:\n]+?)\s*:\s*(.*?)(?=<thing>|<stuff>|<scene_end>|$)",
    flags=re.IGNORECASE | re.DOTALL,
)


def _normalize_category(value: str) -> str:
    value = re.sub(r"\s+\d+\s*$", "", value.strip().lower())
    return re.sub(r"\s+", " ", value)


def _normalize_description(value: str) -> str:
    value = value.split("<ovt>", 1)[0].lower()
    value = re.sub(r"[^a-z0-9 ]+", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def _parse_objects(text: str) -> list[tuple[str, str]]:
    # Count only structurally complete marker/description/<ovt> records.  A
    # repeated bare <ovt> therefore cannot masquerade as another object.
    return [
        (_normalize_category(name), _normalize_description(description))
        for name, description in OBJECT_RE.findall(text)
        if "<ovt>" in description.lower()
    ]


def _safe_mean(values: list[float]) -> float | None:
    return float(np.mean(values)) if values else None


def _quantiles(values: np.ndarray) -> dict[str, float]:
    return {
        key: float(np.quantile(values, quantile))
        for key, quantile in (("min", 0), ("p25", 0.25), ("median", 0.5), ("p75", 0.75), ("p90", 0.9), ("p95", 0.95), ("max", 1.0))
    }


def _bucket(count: int) -> str:
    if count <= 3:
        return "1-3"
    if count <= 6:
        return "4-6"
    if count <= 10:
        return "7-10"
    return "11+"


def _choose_examples(rows: list[dict]) -> list[dict]:
    candidates: list[tuple[str, dict]] = []

    def take(label: str, predicate, key, reverse: bool = True) -> None:
        options = [row for row in rows if predicate(row)]
        options.sort(key=key, reverse=reverse)
        used = {entry["index"] for _, entry in candidates}
        for row in options:
            if row["index"] not in used:
                selected = dict(row)
                selected["selection_reason"] = label
                candidates.append((label, selected))
                return

    take(
        "exact count",
        lambda r: r["difference"] == 0 and r["format_valid"] and r["gt_count"] <= 5,
        lambda r: (r["category_recall_exact"], -r["gt_count"]),
    )
    take(
        "mild over-count",
        lambda r: 1 <= r["difference"] <= 3 and r["format_valid"] and r["gt_count"] <= 6,
        lambda r: (r["category_recall_exact"], -r["difference"]),
    )
    take("severe repeated-category over-count", lambda r: r["duplicate_category_count"] >= 5, lambda r: r["duplicate_category_count"])
    take("severe unique-category over-count", lambda r: r["difference"] >= 8, lambda r: (r["extra_unique_category_count"], r["difference"]))
    take("under-count", lambda r: r["difference"] <= -2, lambda r: -r["difference"])
    take("generation cutoff", lambda r: not r["scene_end_generated"], lambda r: r["pred_count"])
    return [entry for _, entry in candidates]


def run(args: argparse.Namespace) -> dict:
    with open(args.generated_jsonl) as handle:
        generated = [json.loads(line) for line in handle]
    with open(args.val_jsonl) as handle:
        ground_truth = [json.loads(line) for line in handle]
    if len(generated) != len(ground_truth):
        raise ValueError(
            f"Aligned JSONL length mismatch: generated={len(generated)} gt={len(ground_truth)}"
        )

    metric_count_by_image: dict[int, int] = {}
    if args.coco_mask_cache:
        cache_dir = Path(args.coco_mask_cache)
        cache_ids = np.load(cache_dir / "image_ids.npy")
        cache_counts = np.load(cache_dir / "gt_counts.npy")
        metric_count_by_image = {
            int(image_id): int(count)
            for image_id, count in zip(cache_ids.tolist(), cache_counts.tolist())
        }

    rows = []
    per_gt_bucket: dict[str, list[dict]] = defaultdict(list)
    all_pred_categories = Counter()
    all_gt_categories = Counter()
    for index, (prediction, target) in enumerate(zip(generated, ground_truth)):
        parsed = _parse_objects(prediction["text"])
        pred_categories = [category for category, _ in parsed]
        descriptions = [description for _, description in parsed if description]
        gt_categories = [
            _normalize_category(str(segment.get("category", "")))
            for segment in target.get("segments", [])
        ]
        pred_counter = Counter(pred_categories)
        gt_counter = Counter(gt_categories)
        matched = sum((pred_counter & gt_counter).values())
        pred_count = int(prediction["object_count"])
        gt_count = int(target.get("n_objects", len(gt_categories)))
        row = {
            "index": index,
            "image_id": int(target["image_id"]),
            "gt_count": gt_count,
            "metric_gt_count": metric_count_by_image.get(int(target["image_id"])),
            "pred_count": pred_count,
            "difference": pred_count - gt_count,
            "format_valid": bool(prediction["format_valid"]),
            "scene_end_generated": bool(prediction["scene_end_generated"]),
            "token_count": int(prediction["token_count"]),
            "parsed_object_count": len(parsed),
            "duplicate_category_count": sum(max(value - 1, 0) for value in pred_counter.values()),
            "duplicate_description_count": sum(max(value - 1, 0) for value in Counter(descriptions).values()),
            "extra_unique_category_count": len(set(pred_categories) - set(gt_categories)),
            "matched_exact_category_instances": matched,
            "category_precision_exact": matched / max(len(parsed), 1),
            "category_precision_per_raw_ovt_exact": matched / max(pred_count, 1),
            "category_recall_exact": matched / max(gt_count, 1),
            "gt_categories": gt_categories,
            "pred_categories": pred_categories,
            "caption": prediction["text"],
        }
        rows.append(row)
        per_gt_bucket[_bucket(gt_count)].append(row)
        all_pred_categories.update(pred_categories)
        all_gt_categories.update(gt_categories)

    gt_counts = np.asarray([row["gt_count"] for row in rows], dtype=np.int64)
    pred_counts = np.asarray([row["pred_count"] for row in rows], dtype=np.int64)
    parsed_counts = np.asarray(
        [row["parsed_object_count"] for row in rows], dtype=np.int64
    )
    metric_gt_counts = np.asarray(
        [
            row["metric_gt_count"]
            if row["metric_gt_count"] is not None
            else row["gt_count"]
            for row in rows
        ],
        dtype=np.int64,
    )
    dangling_ovts = pred_counts - parsed_counts
    differences = pred_counts - gt_counts
    metric_differences = pred_counts - metric_gt_counts
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    max_count = int(max(gt_counts.max(), metric_gt_counts.max(), pred_counts.max()))
    bins = np.arange(-0.5, max_count + 1.5, 1.0)
    figure, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes[0, 0].hist(gt_counts, bins=bins, alpha=0.55, label="Caption GT records", color="#2878b5")
    axes[0, 0].hist(metric_gt_counts, bins=bins, histtype="step", linewidth=2.0, label="COCO metric GT instances", color="#555555")
    axes[0, 0].hist(pred_counts, bins=bins, alpha=0.50, label="All generated <OVT>", color="#e07a1f")
    axes[0, 0].hist(parsed_counts, bins=bins, histtype="step", linewidth=2.0, label="Complete caption records", color="#339966")
    axes[0, 0].set_xlabel("Count per image")
    axes[0, 0].set_ylabel("Number of images")
    axes[0, 0].set_title("Object-count distributions")
    axes[0, 0].legend()
    axes[0, 0].grid(axis="y", alpha=0.2)

    parsed_differences = parsed_counts - gt_counts
    parsed_metric_differences = parsed_counts - metric_gt_counts
    diff_min = min(
        int(differences.min()),
        int(parsed_differences.min()),
        int(parsed_metric_differences.min()),
    )
    diff_max = max(
        int(differences.max()),
        int(parsed_differences.max()),
        int(parsed_metric_differences.max()),
    )
    diff_bins = np.arange(diff_min - 0.5, diff_max + 1.5, 1.0)
    axes[0, 1].hist(differences, bins=diff_bins, color="#6f55a5", alpha=0.55, label="All <OVT>")
    axes[0, 1].hist(parsed_differences, bins=diff_bins, histtype="step", linewidth=2.0, color="#339966", label="Complete records")
    axes[0, 1].hist(parsed_metric_differences, bins=diff_bins, histtype="step", linewidth=1.7, linestyle="--", color="#222222", label="Complete records vs metric GT")
    axes[0, 1].axvline(0, color="black", linewidth=1)
    axes[0, 1].set_xlabel("Generated count - GT count")
    axes[0, 1].set_ylabel("Number of images")
    axes[0, 1].set_title("Count error")
    axes[0, 1].legend()
    axes[0, 1].grid(axis="y", alpha=0.2)

    clip = int(args.heatmap_clip)
    matrix = np.zeros((clip + 1, clip + 1), dtype=np.int64)
    for gt_count, pred_count in zip(metric_gt_counts, pred_counts):
        matrix[min(int(gt_count), clip), min(int(pred_count), clip)] += 1
    image = axes[1, 0].imshow(np.log1p(matrix), origin="lower", cmap="magma", aspect="auto")
    axes[1, 0].plot([0, clip], [0, clip], color="white", linewidth=1, alpha=0.8)
    axes[1, 0].set_xlabel(f"All generated <OVT> ({clip}={clip}+)")
    axes[1, 0].set_ylabel(f"COCO metric GT instances ({clip}={clip}+)")
    axes[1, 0].set_title("Raw token count vs metric GT")
    figure.colorbar(image, ax=axes[1, 0], label="log(1 + images)")

    parsed_matrix = np.zeros((clip + 1, clip + 1), dtype=np.int64)
    for gt_count, parsed_count in zip(metric_gt_counts, parsed_counts):
        parsed_matrix[min(int(gt_count), clip), min(int(parsed_count), clip)] += 1
    parsed_image = axes[1, 1].imshow(
        np.log1p(parsed_matrix), origin="lower", cmap="magma", aspect="auto"
    )
    axes[1, 1].plot([0, clip], [0, clip], color="white", linewidth=1, alpha=0.8)
    axes[1, 1].set_xlabel(f"Complete caption records ({clip}={clip}+)")
    axes[1, 1].set_ylabel(f"COCO metric GT instances ({clip}={clip}+)")
    axes[1, 1].set_title("Parsed object records vs metric GT")
    figure.colorbar(parsed_image, ax=axes[1, 1], label="log(1 + images)")
    figure.tight_layout()
    distribution_path = output_dir / "object_count_distribution.png"
    figure.savefig(distribution_path, dpi=180, bbox_inches="tight")
    plt.close(figure)

    over = differences > 0
    exact = differences == 0
    under = differences < 0
    selected_examples = _choose_examples(rows)
    summary = {
        "num_samples": len(rows),
        "gt_count": {"mean": float(gt_counts.mean()), **_quantiles(gt_counts)},
        "metric_gt_count": {
            "mean": float(metric_gt_counts.mean()),
            **_quantiles(metric_gt_counts),
        },
        "pred_count": {"mean": float(pred_counts.mean()), **_quantiles(pred_counts)},
        "parsed_marker_count": {
            "mean": float(parsed_counts.mean()),
            **_quantiles(parsed_counts),
        },
        "schema_count_gap": {
            "dangling_ovt_mean": float(dangling_ovts.mean()),
            "images_with_dangling_ovt_rate": float((dangling_ovts > 0).mean()),
            "fraction_of_counted_ovts_without_parsed_object_record": float(
                dangling_ovts.sum() / max(pred_counts.sum(), 1)
            ),
            "max_object_cap_rate": float((pred_counts >= 50).mean()),
            "note": (
                "run_eval counts every <ovt> as an object. parsed_marker_count "
                "counts complete '<thing> name: ... <ovt>' records, so this gap "
                "exposes malformed/repeated bare <ovt> tokens."
            ),
        },
        "difference": {"mean": float(differences.mean()), **_quantiles(differences)},
        "count_relation": {
            "over_rate": float(over.mean()),
            "exact_rate": float(exact.mean()),
            "under_rate": float(under.mean()),
            "pearson_correlation": float(np.corrcoef(gt_counts, pred_counts)[0, 1]),
        },
        "raw_ovt_vs_metric_gt_count_relation": {
            "mean_difference": float(metric_differences.mean()),
            "mae": float(np.abs(metric_differences).mean()),
            "over_rate": float((metric_differences > 0).mean()),
            "exact_rate": float((metric_differences == 0).mean()),
            "under_rate": float((metric_differences < 0).mean()),
            "pearson_correlation": float(
                np.corrcoef(metric_gt_counts, pred_counts)[0, 1]
            ),
        },
        "parsed_count_relation": {
            "mean_difference": float(parsed_differences.mean()),
            "mae": float(np.abs(parsed_differences).mean()),
            "over_rate": float((parsed_differences > 0).mean()),
            "exact_rate": float((parsed_differences == 0).mean()),
            "under_rate": float((parsed_differences < 0).mean()),
            "pearson_correlation": float(np.corrcoef(gt_counts, parsed_counts)[0, 1]),
        },
        "parsed_vs_metric_gt_count_relation": {
            "mean_difference": float(parsed_metric_differences.mean()),
            "mae": float(np.abs(parsed_metric_differences).mean()),
            "over_rate": float((parsed_metric_differences > 0).mean()),
            "exact_rate": float((parsed_metric_differences == 0).mean()),
            "under_rate": float((parsed_metric_differences < 0).mean()),
            "pearson_correlation": float(
                np.corrcoef(metric_gt_counts, parsed_counts)[0, 1]
            ),
        },
        "format": {
            "valid_rate": _safe_mean([float(row["format_valid"]) for row in rows]),
            "scene_end_rate": _safe_mean([float(row["scene_end_generated"]) for row in rows]),
            "token_cutoff_rate": _safe_mean([float(not row["scene_end_generated"]) for row in rows]),
            "strict_schema_valid_rate": float(
                np.mean(
                    [
                        row["scene_end_generated"]
                        and row["pred_count"] == row["parsed_object_count"]
                        and row["parsed_object_count"] > 0
                        for row in rows
                    ]
                )
            ),
        },
        "caption_repetition": {
            "images_with_duplicate_category_rate": _safe_mean([float(row["duplicate_category_count"] > 0) for row in rows]),
            "duplicate_category_count_mean": _safe_mean([row["duplicate_category_count"] for row in rows]),
            "images_with_duplicate_description_rate": _safe_mean([float(row["duplicate_description_count"] > 0) for row in rows]),
            "duplicate_description_count_mean": _safe_mean([row["duplicate_description_count"] for row in rows]),
        },
        "exact_category_proxy": {
            "per_image_precision_mean": _safe_mean([row["category_precision_exact"] for row in rows]),
            "per_image_precision_per_raw_ovt_mean": _safe_mean(
                [row["category_precision_per_raw_ovt_exact"] for row in rows]
            ),
            "per_image_recall_mean": _safe_mean([row["category_recall_exact"] for row in rows]),
            "note": (
                "Exact normalized category-name match over complete caption records; "
                "synonyms and valid unannotated objects count as mismatches. The raw-OVT "
                "precision additionally treats dangling <ovt> tokens as false positives."
            ),
        },
        "by_gt_count_bucket": {
            key: {
                "n": len(values),
                "gt_mean": _safe_mean([row["gt_count"] for row in values]),
                "pred_mean": _safe_mean([row["pred_count"] for row in values]),
                "mae": _safe_mean([abs(row["difference"]) for row in values]),
                "over_rate": _safe_mean([float(row["difference"] > 0) for row in values]),
                "format_valid_rate": _safe_mean([float(row["format_valid"]) for row in values]),
            }
            for key, values in sorted(per_gt_bucket.items())
        },
        "most_overpredicted": sorted(rows, key=lambda row: row["difference"], reverse=True)[:20],
        "most_underpredicted": sorted(rows, key=lambda row: row["difference"])[:20],
        "selected_visual_examples": selected_examples,
        "top_predicted_categories": all_pred_categories.most_common(25),
        "top_gt_categories": all_gt_categories.most_common(25),
        "distribution_plot": str(distribution_path),
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    (output_dir / "selected_examples.json").write_text(
        json.dumps(selected_examples, indent=2)
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generated_jsonl", required=True)
    parser.add_argument("--val_jsonl", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--heatmap_clip", type=int, default=20)
    parser.add_argument(
        "--coco_mask_cache",
        default="/home/jovyan/PGOT/data/coco_inst_mask_cache_coda512",
    )
    args = parser.parse_args()
    print(json.dumps(run(args), indent=2))


if __name__ == "__main__":
    main()
