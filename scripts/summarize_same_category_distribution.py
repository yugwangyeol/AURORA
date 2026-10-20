#!/usr/bin/env python3
"""Summarize audited same-category distractors for cached RefCOCO evaluation."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np


def load_jsonl(path: Path) -> list[dict]:
    with path.open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--per_expression", type=Path, required=True)
    parser.add_argument("--train_jsonl", type=Path, required=True)
    parser.add_argument("--output_prefix", type=Path, required=True)
    args = parser.parse_args()

    rows = load_jsonl(args.per_expression)
    train_ids = {int(row["image_id"]) for row in load_jsonl(args.train_jsonl)}
    subsets = {
        "full": rows,
        "nonoverlap": [row for row in rows if int(row["image_id"]) not in train_ids],
    }
    result = {
        "source": str(args.per_expression),
        "definition": (
            "Number of other valid GT instances in the same image with the target COCO "
            "category. The target annotation itself is excluded."
        ),
        "subsets": {},
    }
    csv_rows = []
    splits = ("val", "testA", "testB", "all")
    for subset_name, subset_rows in subsets.items():
        result["subsets"][subset_name] = {}
        for split in splits:
            selected = subset_rows if split == "all" else [row for row in subset_rows if row["split"] == split]
            if not selected:
                continue
            values = np.asarray(
                [int(row["same_category_gt_competitors"]) for row in selected], dtype=np.int64
            )
            if np.any(values < 0):
                raise AssertionError("Distractor counts must be non-negative")
            distribution = Counter(values.tolist())
            summary = {
                "num_images": len({int(row["image_id"]) for row in selected}),
                "num_expressions": len(selected),
                "expressions_with_distractor": int((values >= 1).sum()),
                "expressions_with_distractor_fraction": float((values >= 1).mean()),
                "mean_distractors": float(values.mean()),
                "median_distractors": float(np.median(values)),
                "max_distractors": int(values.max()),
                "distribution": {
                    str(count): {
                        "count": int(frequency),
                        "fraction": float(frequency / len(selected)),
                    }
                    for count, frequency in sorted(distribution.items())
                },
            }
            result["subsets"][subset_name][split] = summary
            for count, values_at_count in summary["distribution"].items():
                csv_rows.append({
                    "subset": subset_name,
                    "split": split,
                    "num_images": summary["num_images"],
                    "num_expressions": summary["num_expressions"],
                    "distractor_count": count,
                    "count": values_at_count["count"],
                    "fraction": values_at_count["fraction"],
                })

    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    json_path = args.output_prefix.with_suffix(".json")
    csv_path = args.output_prefix.with_suffix(".csv")
    markdown_path = args.output_prefix.with_suffix(".md")
    json_path.write_text(json.dumps(result, indent=2) + "\n")
    with csv_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=[
            "subset", "split", "num_images", "num_expressions",
            "distractor_count", "count", "fraction",
        ])
        writer.writeheader()
        writer.writerows(csv_rows)
    lines = [
        "| Subset | Split | Images | Expressions | With ≥1 distractor | Mean | Median | Max |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for subset_name, split_results in result["subsets"].items():
        for split, summary in split_results.items():
            lines.append(
                f"| {subset_name} | {split} | {summary['num_images']} | "
                f"{summary['num_expressions']} | "
                f"{100 * summary['expressions_with_distractor_fraction']:.2f}% "
                f"({summary['expressions_with_distractor']}) | "
                f"{summary['mean_distractors']:.2f} | {summary['median_distractors']:.1f} | "
                f"{summary['max_distractors']} |"
            )
    lines.extend([
        "",
        "The target annotation is excluded from every distractor count.",
    ])
    markdown_path.write_text("\n".join(lines) + "\n")
    print(markdown_path)


if __name__ == "__main__":
    main()
