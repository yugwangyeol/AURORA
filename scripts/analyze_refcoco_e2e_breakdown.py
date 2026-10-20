#!/usr/bin/env python3
"""Audit and re-aggregate cached RefCOCO object-selection results."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from pycocotools.coco import COCO


CONDITIONS = ("category", "description")


def load_jsonl(path: Path) -> list[dict]:
    with path.open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def load_train_image_ids(path: Path) -> set[int]:
    result: set[int] = set()
    for row in load_jsonl(path):
        image_id = row.get("image_id", row.get("coco_image_id"))
        if image_id is not None:
            result.add(int(image_id))
    return result


def clustered_bootstrap(
    records: list[dict], value_fn, *, seed: int, iterations: int
) -> list[float | None]:
    if not records:
        return [None, None]
    by_image: dict[int, list[float]] = defaultdict(list)
    for record in records:
        by_image[record["image_id"]].append(float(value_fn(record)))
    image_ids = np.asarray(sorted(by_image), dtype=np.int64)
    sums = np.asarray([sum(by_image[int(i)]) for i in image_ids], dtype=np.float64)
    counts = np.asarray([len(by_image[int(i)]) for i in image_ids], dtype=np.float64)
    if image_ids.size == 1:
        value = float(sums[0] / counts[0])
        return [value, value]
    rng = np.random.default_rng(seed)
    estimates = np.empty(iterations, dtype=np.float64)
    for index in range(iterations):
        sampled = rng.integers(0, image_ids.size, size=image_ids.size)
        estimates[index] = sums[sampled].sum() / counts[sampled].sum()
    low, high = np.quantile(estimates, [0.025, 0.975])
    return [float(low), float(high)]


def aggregate(records: list[dict], condition: str, seed: int, iterations: int) -> dict:
    count = len(records)
    if not count:
        return {}

    def covered(record: dict) -> bool:
        return record["max_candidate_iou"] >= 0.5

    def success(record: dict) -> bool:
        return record["conditions"][condition]["iou"] >= 0.5

    def oracle_id_hit(record: dict) -> bool:
        return record["conditions"][condition]["selected_id"] == record["max_candidate_id"]

    covered_records = [record for record in records if covered(record)]
    success_count = sum(success(record) for record in records)
    selection_fail_count = sum(covered(record) and not success(record) for record in records)
    not_covered_count = sum(not covered(record) for record in records)
    if any(success(record) and not covered(record) for record in records):
        raise AssertionError("Selected-mask success is impossible when max-candidate coverage fails")
    if not_covered_count + selection_fail_count + success_count != count:
        raise AssertionError("Failure decomposition must be mutually exclusive and exhaustive")

    selected_iou = lambda record: record["conditions"][condition]["iou"]
    result = {
        "num_images": len({record["image_id"] for record in records}),
        "num_expressions": count,
        "coverage_50": sum(covered(record) for record in records) / count,
        "coverage_50_ci95": clustered_bootstrap(
            records, covered, seed=seed + 1, iterations=iterations
        ),
        "selector_success_given_covered_50": (
            sum(success(record) for record in covered_records) / len(covered_records)
            if covered_records else None
        ),
        "oracle_id_hit_given_covered_50": (
            sum(oracle_id_hit(record) for record in covered_records) / len(covered_records)
            if covered_records else None
        ),
        "e2e_success_50": success_count / count,
        "e2e_success_50_ci95": clustered_bootstrap(
            records, success, seed=seed + 2, iterations=iterations
        ),
        "e2e_miou": sum(selected_iou(record) for record in records) / count,
        "e2e_miou_ci95": clustered_bootstrap(
            records, selected_iou, seed=seed + 3, iterations=iterations
        ),
        "not_covered_50": not_covered_count / count,
        "covered_but_selection_failed_50": selection_fail_count / count,
        "e2e_success_component_50": success_count / count,
    }
    terms = (
        result["not_covered_50"]
        + result["covered_but_selection_failed_50"]
        + result["e2e_success_component_50"]
    )
    if not np.isclose(terms, 1.0):
        raise AssertionError(f"Failure decomposition sums to {terms}, not 1")
    return result


def paired_delta(
    records: list[dict], metric: str, *, seed: int, iterations: int
) -> dict:
    if metric == "success_50":
        value = lambda record, condition: float(record["conditions"][condition]["iou"] >= 0.5)
    elif metric == "miou":
        value = lambda record, condition: float(record["conditions"][condition]["iou"])
    else:
        raise ValueError(metric)
    delta = lambda record: value(record, "description") - value(record, "category")
    return {
        "description_minus_category": float(np.mean([delta(record) for record in records])),
        "ci95": clustered_bootstrap(records, delta, seed=seed, iterations=iterations),
    }


def build_records(root: Path, annotations: Path) -> tuple[list[dict], list[str]]:
    manifest = json.loads((root / "manifest.json").read_text())
    samples = {int(sample["image_id"]): sample for sample in manifest["samples"]}
    grouped: dict[tuple[str, str], dict[str, dict]] = defaultdict(dict)
    for row in load_jsonl(root / "per_query.jsonl"):
        if row["condition"] in CONDITIONS:
            grouped[(str(row["split"]), str(row["query_id"]))][row["condition"]] = row
    missing = [key for key, rows in grouped.items() if any(c not in rows for c in CONDITIONS)]
    if missing:
        raise ValueError(f"{len(missing)} queries do not contain both selector conditions")

    coco = COCO(str(annotations))
    annotation_masks: dict[int, np.ndarray] = {}
    by_image_queries: dict[tuple[str, int], list[tuple[str, dict[str, dict]]]] = defaultdict(list)
    for (split, query_id), conditions in grouped.items():
        image_id = int(conditions["category"]["image_id"])
        by_image_queries[(split, image_id)].append((query_id, conditions))

    records: list[dict] = []
    for image_key, queries in by_image_queries.items():
        sample = samples[image_key[1]]
        prediction_path = root / "owners" / str(image_key[1]) / "prediction.npz"
        with np.load(prediction_path) as archive:
            prediction = np.asarray(archive["prediction"], dtype=np.int64)
        label_areas = np.bincount(prediction.ravel())
        candidates = sample["candidates"]
        for query_id, conditions in queries:
            annotation_id = int(conditions["category"]["annotation_id"])
            if annotation_id not in annotation_masks:
                annotation_masks[annotation_id] = coco.annToMask(coco.anns[annotation_id]).astype(bool)
            target = annotation_masks[annotation_id]
            if target.shape != prediction.shape:
                raise ValueError(
                    f"Mask shape mismatch for annotation {annotation_id}: "
                    f"GT={target.shape}, prediction={prediction.shape}"
                )
            intersections = np.bincount(prediction[target], minlength=label_areas.size)
            target_area = int(target.sum())
            candidate_ious: dict[str, float] = {}
            for candidate in candidates:
                candidate_id = str(candidate["id"])
                label = candidate.get("prediction_label")
                if label is None or int(label) >= label_areas.size:
                    candidate_ious[candidate_id] = 0.0
                    continue
                label = int(label)
                intersection = int(intersections[label])
                union = target_area + int(label_areas[label]) - intersection
                candidate_ious[candidate_id] = intersection / union if union else 0.0
            if candidate_ious:
                best_id, best_iou = max(candidate_ious.items(), key=lambda item: item[1])
            else:
                best_id, best_iou = None, 0.0
            for condition in CONDITIONS:
                selected_id = conditions[condition].get("selected_id")
                recomputed_iou = candidate_ious.get(str(selected_id), 0.0) if selected_id is not None else 0.0
                if not np.isclose(recomputed_iou, float(conditions[condition]["iou"]), atol=1e-9):
                    raise AssertionError(
                        f"Cached/recomputed IoU mismatch for query {query_id}, {condition}: "
                        f"{conditions[condition]['iou']} vs {recomputed_iou}"
                    )
            records.append(
                {
                    "split": image_key[0],
                    "query_id": query_id,
                    "image_id": image_key[1],
                    "annotation_id": annotation_id,
                    "max_candidate_id": best_id,
                    "max_candidate_iou": float(best_iou),
                    "conditions": {
                        condition: {
                            "selected_id": conditions[condition].get("selected_id"),
                            "iou": float(conditions[condition]["iou"]),
                        }
                        for condition in CONDITIONS
                    },
                }
            )
    return records, sorted({record["split"] for record in records})


def pct(value: float | None) -> str:
    return "—" if value is None else f"{100 * value:.2f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root", type=Path, nargs="+", required=True,
        help="One or more compatible split caches (for example val testA testB).",
    )
    parser.add_argument(
        "--annotations", type=Path,
        default=Path("/home/jovyan/data/coco/refcoco/instances.json"),
    )
    parser.add_argument(
        "--train_jsonl", type=Path,
        default=Path("/home/jovyan/PGOT/data/pgot_pix2cap_thing_train.jsonl"),
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--bootstrap_iterations", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()
    if args.output is None and len(args.root) != 1:
        parser.error("--output is required when multiple --root directories are supplied")
    output = args.output or (args.root[0] / "reviewer_breakdown_v2.json")

    records = []
    cached_splits_set: set[str] = set()
    for root in args.root:
        root_records, root_splits = build_records(root, args.annotations)
        records.extend(root_records)
        cached_splits_set.update(root_splits)
    query_keys = [(record["split"], record["query_id"]) for record in records]
    if len(query_keys) != len(set(query_keys)):
        raise ValueError("Duplicate split/query IDs across supplied cache roots")
    cached_splits = [split for split in ("val", "testA", "testB") if split in cached_splits_set]
    train_ids = load_train_image_ids(args.train_jsonl)
    subsets = {
        "full": records,
        "nonoverlap": [record for record in records if record["image_id"] not in train_ids],
    }
    result = {
        "source": [str(root) for root in args.root],
        "cached_splits": cached_splits,
        "warning": (
            "Only cached splits are reported; missing testA/testB require new DeOR inference."
            if set(cached_splits) != {"val", "testA", "testB"} else None
        ),
        "definitions": {
            "coverage_50": "At least one generated DeOR candidate mask reaches IoU >= 0.5.",
            "selector_success_given_covered_50": "Selected mask reaches IoU >= 0.5, conditional on coverage.",
            "oracle_id_hit_given_covered_50": "Selected candidate ID equals the maximum-IoU candidate ID, conditional on coverage.",
            "e2e_success_50": "Actually selected mask reaches IoU >= 0.5.",
            "failure_partition": "Not covered + covered but selected mask fails + E2E success = 1.",
            "bootstrap": "95% percentile CI from image-cluster bootstrap; expressions remain clustered by image.",
        },
        "subsets": {},
    }
    table_rows: list[dict] = []
    detailed_rows: list[dict] = []
    for subset_name, subset_records in subsets.items():
        result["subsets"][subset_name] = {}
        for split in [*cached_splits, "all"]:
            split_records = (
                subset_records if split == "all"
                else [record for record in subset_records if record["split"] == split]
            )
            if not split_records:
                continue
            split_result = {
                "conditions": {},
                "paired_deltas": {
                    "e2e_success_50": paired_delta(
                        split_records, "success_50", seed=args.seed + 20,
                        iterations=args.bootstrap_iterations,
                    ),
                    "e2e_miou": paired_delta(
                        split_records, "miou", seed=args.seed + 30,
                        iterations=args.bootstrap_iterations,
                    ),
                },
                "oracle_matched_representation": {
                    "coverage_50": float(np.mean([
                        record["max_candidate_iou"] >= 0.5 for record in split_records
                    ])),
                    "miou": float(np.mean([
                        record["max_candidate_iou"] for record in split_records
                    ])),
                },
            }
            coverages = []
            for condition in CONDITIONS:
                metrics = aggregate(
                    split_records, condition, args.seed,
                    args.bootstrap_iterations,
                )
                split_result["conditions"][condition] = metrics
                coverages.append(metrics["coverage_50"])
                table_rows.append({
                    "subset": subset_name,
                    "split": split,
                    "condition": condition,
                    **metrics,
                })
            if not np.isclose(coverages[0], coverages[1]):
                raise AssertionError("Selector conditions must share the identical candidate pool")
            result["subsets"][subset_name][split] = split_result

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    per_query_path = output.with_name(output.stem + "_per_query.jsonl")
    with per_query_path.open("w") as stream:
        for record in records:
            stream.write(json.dumps(record) + "\n")
    csv_path = output.with_suffix(".csv")
    csv_fields = [
        "subset", "split", "condition", "num_images", "num_expressions",
        "coverage_50", "selector_success_given_covered_50",
        "oracle_id_hit_given_covered_50", "e2e_success_50", "e2e_miou",
        "not_covered_50", "covered_but_selection_failed_50",
        "e2e_success_component_50",
    ]
    with csv_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=csv_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(table_rows)

    markdown_path = output.with_suffix(".md")
    lines = [
        "| Subset | Split | Selector input | Images | Expressions | Coverage@0.5 | Success \\| Covered | Oracle-ID Hit \\| Covered | E2E Success@0.5 | E2E mIoU | Not covered | Covered, selection failed |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in table_rows:
        lines.append(
            f"| {row['subset']} | {row['split']} | {row['condition']} | "
            f"{row['num_images']} | {row['num_expressions']} | {pct(row['coverage_50'])} | "
            f"{pct(row['selector_success_given_covered_50'])} | "
            f"{pct(row['oracle_id_hit_given_covered_50'])} | {pct(row['e2e_success_50'])} | "
            f"{pct(row['e2e_miou'])} | {pct(row['not_covered_50'])} | "
            f"{pct(row['covered_but_selection_failed_50'])} |"
        )
    lines.extend([
        "",
        "| Subset | Split | Δ E2E Success (description-category) | 95% CI | Δ mIoU | 95% CI | Oracle matched coverage | Oracle matched mIoU |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ])
    for subset_name, split_results in result["subsets"].items():
        for split, values in split_results.items():
            success = values["paired_deltas"]["e2e_success_50"]
            miou = values["paired_deltas"]["e2e_miou"]
            oracle = values["oracle_matched_representation"]
            lines.append(
                f"| {subset_name} | {split} | {pct(success['description_minus_category'])} | "
                f"[{pct(success['ci95'][0])}, {pct(success['ci95'][1])}] | "
                f"{pct(miou['description_minus_category'])} | "
                f"[{pct(miou['ci95'][0])}, {pct(miou['ci95'][1])}] | "
                f"{pct(oracle['coverage_50'])} | {pct(oracle['miou'])} |"
            )
    markdown_path.write_text("\n".join(lines) + "\n")
    print(markdown_path)


if __name__ == "__main__":
    main()
