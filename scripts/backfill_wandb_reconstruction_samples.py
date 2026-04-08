#!/usr/bin/env python3
"""Backfill legacy W&B reconstruction tables into step-indexable image samples.

This creates a separate W&B run that logs combined GT|Reconstruction images
under a single key such as `eval/samples`, which gives the media panel the
step/index slider UI.
"""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path

import wandb
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-run",
        required=True,
        help="Source run in entity/project/run_id format, e.g. durianpang/AURORA/9syjnpjb",
    )
    parser.add_argument(
        "--target-entity",
        default=None,
        help="Entity for the backfill run. Defaults to the source run entity.",
    )
    parser.add_argument(
        "--target-project",
        default=None,
        help="Project for the backfill run. Defaults to the source run project.",
    )
    parser.add_argument(
        "--target-run-name",
        default=None,
        help="Name for the backfill run. Defaults to <source_name>_samples_backfill.",
    )
    parser.add_argument(
        "--target-run-id",
        default=None,
        help="Optional explicit run id for the backfill run.",
    )
    parser.add_argument(
        "--target-key",
        default="eval/samples",
        help="Metric key to log image arrays to.",
    )
    parser.add_argument(
        "--step-source",
        choices=("wandb_step", "train_global_step"),
        default="wandb_step",
        help="Which historical step to reuse for the new media logs.",
    )
    parser.add_argument(
        "--separator-px",
        type=int,
        default=8,
        help="White separator width between GT and reconstruction.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Optional limit on the number of historical evaluation steps to backfill.",
    )
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="Backfill only the overlapping prefix if the source run is not fully synced yet.",
    )
    return parser.parse_args()


def version_key(artifact: wandb.Artifact) -> int:
    version = getattr(artifact, "version", "") or ""
    if not version.startswith("v"):
        return -1
    try:
        return int(version[1:])
    except ValueError:
        return -1


def combine_images(left: Image.Image, right: Image.Image, separator_px: int) -> Image.Image:
    left = left.convert("RGB")
    right = right.convert("RGB")
    if right.size != left.size:
        right = right.resize(left.size, Image.Resampling.BILINEAR)

    separator_px = max(separator_px, 0)
    combined = Image.new(
        "RGB",
        (left.width + separator_px + right.width, left.height),
        color=(255, 255, 255),
    )
    combined.paste(left, (0, 0))
    combined.paste(right, (left.width + separator_px, 0))
    return combined


def iter_reconstruction_rows(source_run: wandb.apis.public.Run) -> list[dict]:
    rows = source_run.history(
        keys=["_step", "train/global_step", "eval/reconstructions"],
        samples=100000,
        pandas=False,
    )
    rows = [row for row in rows if row.get("eval/reconstructions")]
    rows.sort(key=lambda row: int(row["_step"]))
    return rows


def iter_reconstruction_artifacts(source_run: wandb.apis.public.Run) -> list[wandb.Artifact]:
    artifacts = [
        artifact
        for artifact in source_run.logged_artifacts()
        if artifact.type == "run_table" and "evalreconstructions" in artifact.name
    ]
    artifacts.sort(key=version_key)
    return artifacts


def load_table_images(artifact: wandb.Artifact, separator_px: int) -> list[wandb.Image]:
    artifact_dir = Path(artifact.download(root=tempfile.mkdtemp(prefix="wandb_backfill_art_")))
    table_path = artifact_dir / "eval" / "reconstructions.table.json"
    table = json.loads(table_path.read_text())

    images: list[wandb.Image] = []
    for row in table["data"]:
        _, source_meta, recon_meta = row
        source_path = artifact_dir / source_meta["path"]
        recon_path = artifact_dir / recon_meta["path"]
        with Image.open(source_path) as source_img, Image.open(recon_path) as recon_img:
            combined = combine_images(source_img, recon_img, separator_px=separator_px)
            images.append(
                wandb.Image(
                    combined,
                    caption="GT (left) | Reconstruction (right)",
                )
            )
    shutil.rmtree(artifact_dir, ignore_errors=True)
    return images


def pick_step(row: dict, step_source: str) -> int:
    if step_source == "train_global_step":
        return int(row.get("train/global_step") or row["_step"])
    return int(row["_step"])


def main() -> None:
    args = parse_args()

    api = wandb.Api(timeout=60)
    source_run = api.run(args.source_run)

    history_rows_all = iter_reconstruction_rows(source_run)
    artifacts_all = iter_reconstruction_artifacts(source_run)

    if len(history_rows_all) != len(artifacts_all) and not args.allow_incomplete:
        raise RuntimeError(
            "Backfill aborted because the source run appears incompletely synced. "
            f"Found {len(history_rows_all)} reconstruction history rows but only {len(artifacts_all)} run_table artifacts. "
            "Please finish syncing the source run and try again, or rerun with --allow-incomplete."
        )

    overlap = min(len(history_rows_all), len(artifacts_all))
    history_rows = history_rows_all[:overlap]
    artifacts = artifacts_all[:overlap]

    if args.max_steps is not None:
        history_rows = history_rows[: args.max_steps]
        artifacts = artifacts[: args.max_steps]

    target_entity = args.target_entity or source_run.entity
    target_project = args.target_project or source_run.project
    target_run_name = args.target_run_name or f"{source_run.name}_samples_backfill"

    run = wandb.init(
        entity=target_entity,
        project=target_project,
        id=args.target_run_id,
        name=target_run_name,
        job_type="backfill",
        config={
            "source_run": args.source_run,
            "target_key": args.target_key,
            "step_source": args.step_source,
            "separator_px": args.separator_px,
            "backfilled_eval_steps": len(history_rows),
            "source_history_rows": len(history_rows_all),
            "source_artifacts": len(artifacts_all),
            "allow_incomplete": args.allow_incomplete,
        },
        tags=["backfill", "aurora", "reconstruction-samples"],
    )

    try:
        for row, artifact in zip(history_rows, artifacts):
            step = pick_step(row, args.step_source)
            images = load_table_images(artifact, separator_px=args.separator_px)
            wandb.log({args.target_key: images}, step=step)
    finally:
        run.finish()


if __name__ == "__main__":
    main()
