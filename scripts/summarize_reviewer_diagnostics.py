#!/usr/bin/env python3
"""Validate matched editing pairs and write compact reviewer tables."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load(path: Path):
    return json.loads(path.read_text())


def main() -> None:
    full_edit_dir = ROOT / "outputs/reviewer_object_transfer_full"
    wors_edit_dir = ROOT / "outputs/reviewer_object_transfer_woreader"
    full_local = load(ROOT / "outputs/reviewer_writer_locality_full/summary.json")
    wog_local = load(ROOT / "outputs/reviewer_writer_locality_wogrounding/summary.json")
    full_edit = load(full_edit_dir / "summary.json")
    wors_edit = load(wors_edit_dir / "summary.json")
    identity = ("category_id", "target_sample_index", "target_object_index", "donor_sample_index", "donor_object_index", "diffusion_seed")
    def pair_ids(path):
        return [tuple(json.loads(line)[key] for key in identity) for line in path.open()]
    if pair_ids(full_edit_dir / "pairs.jsonl") != pair_ids(wors_edit_dir / "pairs.jsonl"):
        raise AssertionError("Full and w/o-RS editing runs did not use identical pairs/seeds")

    def mean(result, key):
        return result["metrics"][key]["mean"]
    lines = [
        "# Reviewer diagnostics", "",
        "## Controlled same-category visual-memory transfer", "",
        "| Method | Pairs | Donor gain ↑ | Donor closer ↑ | Outside-target LPIPS ↓ | Target change |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, result in (("Full DeOR", full_edit), ("w/o Reader supervision", wors_edit)):
        lines.append(
            f"| {name} | {result['num_pairs']} | {mean(result,'donor_gain'):.4f} | "
            f"{100*mean(result,'donor_closer'):.2f} | {mean(result,'outside_target_lpips'):.6f} | "
            f"{mean(result,'target_change'):.4f} |"
        )
    lines += ["", "## Writer locality", "",
        "| Method | Object inside mass ↑ | Enrichment ↑ | Other-object mass ↓ | Register annotated-object mass ↓ |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, result in (("Full DeOR", full_local), ("w/o grounding loss", wog_local)):
        lines.append(
            f"| {name} | {mean(result,'object_inside_mask_mass'):.4f} | "
            f"{mean(result,'object_area_normalized_enrichment'):.3f} | "
            f"{mean(result,'object_other_object_mass'):.4f} | "
            f"{mean(result,'register_annotated_object_mass'):.4f} |"
        )
    output = ROOT / "outputs/reviewer_diagnostics_summary.md"
    output.write_text("\n".join(lines) + "\n")
    print(output)


if __name__ == "__main__":
    main()
