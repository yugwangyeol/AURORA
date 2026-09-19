"""Summarize the three requested runs, including image-matched PGOT COCO rows."""
import argparse
import copy
import csv
import json
import math
from pathlib import Path

from pgot.eval.voc_object_counts import METRICS, summarize_object_counts


def matched_summary(summary, records, ids, manifest):
    selected = [row for row in records if int(row["id"]) in ids]
    if {int(row["id"]) for row in selected} != ids or len(selected) != len(ids):
        raise ValueError("Full run does not cover the comparison manifest exactly")
    result = copy.deepcopy(summary)
    result["num_samples"] = len(selected)
    result["comparison_manifest"] = str(manifest)
    result["sample_scope"] = "same COCO image IDs as PGOT generated-caption evaluation manifest"
    result["timing_scope"] = "parent full COCO5000 run, not subset inference"
    for key in METRICS:
        values = [row[key] for row in selected if math.isfinite(row[key])]
        result[key] = sum(values) / len(values) if values else None
        result[f"{key}_valid_n"] = len(values)
    result["mBO_i"], result["mIoU_i"] = result["mBO"], result["mIoU"]
    result["mBO_c"], result["mIoU_c"] = result["sMBO"], result["sMIOU"]
    result["pred_segments_mean"] = sum(row["pred_segments"] for row in selected) / len(selected)
    result.update(summarize_object_counts(selected))
    result["object_count_source"] = summary["object_count_source"]
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_root", default="/home/jovyan/PGOT/outputs/baselines_coco_checkpoint")
    parser.add_argument("--comparison_manifest", default="/home/jovyan/PGOT/data/pgot_pix2cap_generated_val5k.jsonl")
    args = parser.parse_args()
    root, manifest = Path(args.output_root), Path(args.comparison_manifest)
    items = [json.loads(line) for line in manifest.read_text().splitlines() if line.strip()]
    ids = {int(item["image_id"]) for item in items}
    if len(ids) != len(items):
        raise ValueError("Duplicate comparison manifest image IDs")
    combined, flat, count_rows = {}, [], []
    coco_reference = None
    for stage in ("spot_coco", "slotdiffusion_coco", "slotdiffusion_voc"):
        stage_root = root / stage
        summary = json.loads((stage_root / "summary.json").read_text())
        records = [json.loads(line) for line in (stage_root / "per_image.jsonl").read_text().splitlines() if line.strip()]
        if len({row["id"] for row in records}) != len(records) or len(records) != summary["num_samples"]:
            raise ValueError(f"Invalid per-image records in {stage_root}")
        summary.update({f"{key}_valid_n": sum(math.isfinite(row[key]) for row in records) for key in METRICS})
        (stage_root / "summary.json").write_text(json.dumps(summary, indent=2))
        combined[stage] = summary
        if summary["dataset"] == "coco":
            image_counts = {int(row["id"]): int(row["gt_objects"]) for row in records}
            if coco_reference is not None and image_counts != coco_reference:
                raise ValueError("COCO models do not share identical images/GT object counts")
            coco_reference = image_counts
            matched = matched_summary(summary, records, ids, manifest)
            (stage_root / "summary_pgot_subset.json").write_text(json.dumps(matched, indent=2))
            combined[stage + "_pgot_subset"] = matched
    for stage, summary in combined.items():
        flat.append({"run": stage, "model": summary["model"], "dataset": summary["dataset"], "n": summary["num_samples"],
                     **{key: summary[key] for key in METRICS}})
        for group, scores in summary["by_object_count"].items():
            count_rows.append({"run": stage, "model": summary["model"], "dataset": summary["dataset"],
                               "gt_objects": group, "n": scores["n"], **{key: scores[key] for key in METRICS}})
    (root / "comparison.json").write_text(json.dumps(combined, indent=2))
    with (root / "comparison.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("run", "model", "dataset", "n", *METRICS))
        writer.writeheader()
        writer.writerows(flat)
    with (root / "comparison_by_object_count.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("run", "model", "dataset", "gt_objects", "n", *METRICS))
        writer.writeheader()
        writer.writerows(count_rows)
    print(f"{'run':<38} {'n':>5} {'FG-ARI':>8} {'mBO^i':>8} {'mBO^c':>8} {'mIoU^i':>8} {'mIoU^c':>8}")
    for stage, summary in combined.items():
        values = [summary[key] for key in ("fARI", "mBO", "sMBO", "mIoU", "sMIOU")]
        metrics = ' '.join(f'{100 * value:8.2f}' if value is not None else f'{"-":>8}' for value in values)
        print(f'{stage:<38} {summary["num_samples"]:5d} {metrics}')


if __name__ == "__main__":
    main()
