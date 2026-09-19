"""Prepare GQA and cache frozen DeOR object representations.

The pipeline deliberately separates question/answer annotations from the image-only
DeOR extractor.  ``prepare`` materialises the images and writes two independent
JSONL manifests.  ``extract`` reads only the image manifest, performs native AR
discovery followed by one own-caption replay, and stores compact object bundles.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import time
from pathlib import Path
from typing import Iterator


FORMAT_VERSION = "deor-gqa-object-cache-v1"
SPLITS = ("train", "val", "testdev")
SAFE_IMAGE_ID = re.compile(r"^[A-Za-z0-9_-]+$")
QA_COLUMNS = (
    "id", "imageId", "question", "answer", "fullAnswer", "isBalanced",
    "groups", "entailed", "equivalent", "types", "annotations", "semantic",
    "semanticStr",
)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(chunk_size):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def iter_jsonl(path: Path) -> Iterator[dict]:
    with path.open() as stream:
        for line_number, line in enumerate(stream, 1):
            if line.strip():
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(f"Expected object at {path}:{line_number}")
                yield value


def _parquet_files(source_root: Path, split: str, kind: str) -> list[Path]:
    paths = sorted((source_root / f"{split}_balanced_{kind}").glob("*.parquet"))
    if not paths:
        raise FileNotFoundError(f"No {split} {kind} parquet files under {source_root}")
    return paths


def _image_extension(image_format: str | None) -> str:
    value = (image_format or "").upper()
    return {"JPEG": ".jpg", "PNG": ".png", "WEBP": ".webp"}.get(value, ".img")


def prepare_split(args, split: str) -> dict:
    """Materialise one split and write annotation-separated JSONL manifests."""
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover - exercised by launcher preflight
        raise RuntimeError("prepare requires pyarrow; use the metaquery environment") from exc
    from PIL import Image
    import io

    if args.cpu_threads < 1:
        raise ValueError("cpu_threads must be positive")
    # PyArrow otherwise sees all host CPUs rather than this container's cgroup
    # quota and can create hundreds of contending workers.
    pa.set_cpu_count(args.cpu_threads)

    source_root = Path(args.source_root).resolve()
    prepared_root = Path(args.prepared_root).resolve()
    image_paths = _parquet_files(source_root, split, "images")
    question_paths = _parquet_files(source_root, split, "instructions")
    image_root = prepared_root / "images"
    image_root.mkdir(parents=True, exist_ok=True)
    split_root = prepared_root / "manifests"
    split_root.mkdir(parents=True, exist_ok=True)
    images_tmp = split_root / f"images_{split}.jsonl.tmp"
    questions_tmp = split_root / f"questions_{split}.jsonl.tmp"

    image_ids: set[str] = set()
    image_count = 0
    with images_tmp.open("w") as output:
        stop = False
        for parquet_path in image_paths:
            parquet = pq.ParquetFile(parquet_path)
            for batch in parquet.iter_batches(columns=["id", "image"], batch_size=64):
                for row in batch.to_pylist():
                    if args.max_images and image_count >= args.max_images:
                        stop = True
                        break
                    image_id = str(row["id"])
                    if not SAFE_IMAGE_ID.fullmatch(image_id):
                        raise ValueError(f"Unsafe GQA image id: {image_id!r}")
                    if image_id in image_ids:
                        raise ValueError(f"Duplicate image id inside {split}: {image_id}")
                    payload = row["image"]["bytes"]
                    if not isinstance(payload, bytes) or not payload:
                        raise ValueError(f"Missing embedded image bytes for {image_id}")
                    digest = sha256_bytes(payload)
                    with Image.open(io.BytesIO(payload)) as image:
                        image.verify()
                        width, height = image.size
                        extension = _image_extension(image.format)
                    image_path = image_root / f"{image_id}{extension}"
                    if image_path.exists():
                        if sha256_file(image_path) != digest:
                            raise ValueError(f"Image-id collision with different bytes: {image_id}")
                    else:
                        temporary = image_path.with_name(image_path.name + ".tmp")
                        temporary.write_bytes(payload)
                        temporary.replace(image_path)
                    record = {
                        "image_id": image_id,
                        "image_path": str(image_path),
                        "image_sha256": digest,
                        "width": int(width),
                        "height": int(height),
                        "split": split,
                    }
                    output.write(json.dumps(record, ensure_ascii=False) + "\n")
                    image_ids.add(image_id)
                    image_count += 1
                if stop:
                    break
            if stop:
                break
    images_path = split_root / f"images_{split}.jsonl"
    images_tmp.replace(images_path)

    question_count = 0
    with questions_tmp.open("w") as output:
        for parquet_path in question_paths:
            parquet = pq.ParquetFile(parquet_path)
            # ``ParquetSchema.names`` exposes flattened leaf names and therefore
            # omits struct parents such as groups/types/annotations.  The Arrow
            # schema retains the top-level names accepted by iter_batches.
            available = set(parquet.schema_arrow.names)
            missing = set(QA_COLUMNS) - available
            if missing:
                raise ValueError(f"Missing GQA columns in {parquet_path}: {sorted(missing)}")
            for batch in parquet.iter_batches(columns=list(QA_COLUMNS), batch_size=1024):
                for row in batch.to_pylist():
                    if str(row["imageId"]) not in image_ids:
                        if args.max_images:
                            continue
                        raise ValueError(f"Question {row['id']} refers to absent image {row['imageId']}")
                    row["split"] = split
                    output.write(json.dumps(row, ensure_ascii=False) + "\n")
                    question_count += 1
    questions_path = split_root / f"questions_{split}.jsonl"
    questions_tmp.replace(questions_path)
    result = {
        "format_version": FORMAT_VERSION,
        "split": split,
        "num_images": image_count,
        "num_questions": question_count,
        "images_manifest": str(images_path),
        "questions_manifest": str(questions_path),
        "extractor_input_fields": [
            "image_id", "image_path", "image_sha256", "width", "height", "split"
        ],
        "extractor_receives_questions_or_answers": False,
        "source_image_parquets": [str(path.resolve()) for path in image_paths],
        "source_question_parquets": [str(path.resolve()) for path in question_paths],
    }
    write_json(split_root / f"summary_{split}.json", result)
    return result


def checkpoint_fingerprint(model_path: Path) -> dict:
    files = []
    for name in ("config.json", "model.safetensors.index.json"):
        path = model_path / name
        if not path.is_file():
            raise FileNotFoundError(path)
        files.append({"name": name, "sha256": sha256_file(path), "size": path.stat().st_size})
    return {"path": str(model_path.resolve()), "files": files}


def compact_object_bundle(
    result: dict,
    caption_text: str,
    ovt_count: int,
    *,
    cache_dtype: str,
    batch_index: int = 0,
) -> dict:
    """Remove register owners and padded object owners from an E11 bundle."""
    import torch
    from pgot.eval.ar_binding_router import parse_ar_slots

    semantic = result["semantic_slots"]
    visual = result["visual_memory"]
    valid = result["ovt_object_valid"]
    object_probs = result["ovt_object_probs"]
    register_probs = result["ovt_void_probs"]
    if semantic.ndim != 3 or visual.ndim != 4 or valid.ndim != 2:
        raise ValueError("Unexpected DeOR object bundle rank")
    batch_size = semantic.shape[0]
    if not (
        visual.shape[0] == batch_size
        and valid.shape[0] == batch_size
        and object_probs.shape[0] == batch_size
        and register_probs.shape[0] == batch_size
    ):
        raise ValueError("DeOR bundle batch dimensions disagree")
    if not 0 <= batch_index < batch_size:
        raise IndexError(f"batch_index={batch_index} outside batch size {batch_size}")
    object_capacity = valid.shape[1]
    if semantic.shape[1] < object_capacity or visual.shape[1] < object_capacity:
        raise ValueError("Object capacity exceeds semantic/visual owners")
    if visual.shape[2] < 8:
        raise ValueError("Checkpoint exposes fewer than eight memories per object")
    keep = valid[batch_index].bool().nonzero(as_tuple=False).flatten()
    if len(keep) != int(ovt_count):
        raise ValueError(f"Valid object count {len(keep)} != generated OVT count {ovt_count}")
    slots = parse_ar_slots(caption_text, int(ovt_count))
    if len(slots) != len(keep):
        raise ValueError("Caption/object metadata count mismatch")
    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16}[cache_dtype]
    semantic_out = semantic[batch_index, keep].detach().to(device="cpu", dtype=dtype).contiguous()
    visual_out = visual[batch_index, keep, :8].detach().to(device="cpu", dtype=dtype).contiguous()
    owner_out = object_probs[batch_index, keep].detach().to(device="cpu", dtype=dtype).contiguous()
    background = register_probs[batch_index].sum(dim=0).detach().to(device="cpu", dtype=dtype).contiguous()
    competition = torch.cat([owner_out.float(), background.float().unsqueeze(0)], dim=0)
    grid = round(competition.shape[-1] ** 0.5)
    if grid * grid != competition.shape[-1]:
        raise ValueError("Ownership probabilities are not a square patch grid")
    labels = competition.argmax(dim=0).reshape(grid, grid).to(torch.int16)
    objects = []
    for compact_index, (source_index, slot) in enumerate(zip(keep.tolist(), slots)):
        objects.append({
            "object_index": compact_index,
            "source_slot_index": int(source_index),
            "category": slot["category"],
            "description": slot["description"],
            "is_thing": bool(slot["is_thing"]),
            "parsed": bool(slot["parsed"]),
        })
    tensors = (semantic_out, visual_out, owner_out, background)
    if not all(torch.isfinite(value.float()).all() for value in tensors):
        raise ValueError("Non-finite DeOR representation")
    return {
        "objects": objects,
        "semantic": semantic_out,
        "visual_memory": visual_out,
        "object_valid": torch.ones(len(objects), dtype=torch.bool),
        "owner_probabilities": owner_out,
        "background_probability": background,
        "ownership_labels": labels,
        "object_capacity_before_compaction": int(object_capacity),
        "background_register_owners_excluded": int(semantic.shape[1] - object_capacity),
        "register_memories_per_owner_excluded": int(visual.shape[2]),
        "visual_memories_per_object": 8,
    }


def cache_path(cache_root: Path, split: str, image_id: str) -> Path:
    # Two-character fanout prevents tens of thousands of files in one directory.
    fanout = hashlib.sha1(image_id.encode()).hexdigest()[:2]
    return cache_root / split / fanout / f"{image_id}.pt"


def extract(args) -> None:
    import torch
    from PIL import Image
    from pgot.eval.eval_recon_oracles import load_model_and_tokenizer
    from pgot.eval.pgot_inference import pgot_forward_eval
    from pgot.eval.run_eval import generate_pgot_caption_batch

    manifest_path = Path(args.images_manifest).resolve()
    rows = list(iter_jsonl(manifest_path))
    if args.max_images:
        rows = rows[: args.max_images]
    rows = [row for index, row in enumerate(rows) if index % args.num_shards == args.shard_index]
    cache_root = Path(args.cache_root).resolve()
    model_path = Path(args.model_path).resolve()
    provenance = {
        "format_version": FORMAT_VERSION,
        "extractor_code_sha256": sha256_file(Path(__file__)),
        "checkpoint": checkpoint_fingerprint(model_path),
        "images_manifest": str(manifest_path),
        "images_manifest_sha256": sha256_file(manifest_path),
        "model_dtype": args.dtype,
        "cache_dtype": args.cache_dtype,
        "eval_size": args.eval_size,
        "ar_max_new_tokens": args.ar_max_new_tokens,
        "max_objects": args.max_objects,
        "batch_size": args.batch_size,
        "object_representation": "one semantic token plus eight visual memories",
        "register_memories_included": False,
        "question_or_answer_input": False,
    }
    provenance_digest = sha256_bytes(json.dumps(provenance, sort_keys=True).encode())
    pending = []
    for row in rows:
        target = cache_path(cache_root, row["split"], row["image_id"])
        if target.is_file():
            cached = torch.load(target, map_location="cpu", weights_only=False)
            if cached.get("provenance_sha256") != provenance_digest:
                raise ValueError(f"Cache provenance mismatch: {target}")
            if cached.get("image_sha256") != row["image_sha256"]:
                raise ValueError(f"Cache image mismatch: {target}")
        else:
            pending.append(row)
    if not pending:
        print(f"Shard already complete: {len(rows)} images", flush=True)
        return
    if not torch.cuda.is_available():
        raise RuntimeError("DeOR extraction requires CUDA")
    torch.set_num_threads(args.cpu_threads)
    torch.manual_seed(args.seed)
    model, tokenizer, _, _ = load_model_and_tokenizer(args)
    checkpoint_index = json.loads((model_path / "model.safetensors.index.json").read_text())
    checkpoint_keys = set(checkpoint_index["weight_map"])
    named_parameters = dict(model.named_parameters(remove_duplicate=False))
    restored_parameter_ids = {
        id(named_parameters[key]) for key in checkpoint_keys if key in named_parameters
    }
    missing_parameters = [
        key for key, parameter in named_parameters.items()
        if key not in checkpoint_keys and id(parameter) not in restored_parameter_ids
    ]
    unexpected_keys = sorted(checkpoint_keys - set(model.state_dict()))
    if missing_parameters or unexpected_keys:
        raise ValueError(
            f"Checkpoint restore failure: missing={missing_parameters}, "
            f"unexpected={unexpected_keys}"
        )
    cfg = model.config
    expected = (
        int(getattr(cfg, "pgot_e11_object_memories_per_owner", -1)),
        int(getattr(cfg, "pgot_e11_register_memories_per_owner", -1)),
    )
    if expected != (8, 32):
        raise ValueError(f"Expected Object8/Register32 checkpoint, got {expected}")
    if bool(getattr(cfg, "pgot_one_shot_direct_rae_query", False)):
        raise ValueError("Expected the original MLLM RAE-query path")
    if bool(getattr(cfg, "pgot_one_shot_memory_contrastive_enable", False)):
        raise ValueError("Expected the non-contrastive checkpoint")
    for name, token in (("ovt", "<ovt>"), ("scene_end", "<scene_end>"),
                        ("thing", "<thing>"), ("stuff", "<stuff>")):
        setattr(model, f"pgot_{name}_token_id", tokenizer.convert_tokens_to_ids(token))
    towers = model.get_vision_tower_aux_list()
    target_processor = towers[1] if len(towers) > 1 else towers[0]
    started_all = time.perf_counter()
    processed = 0
    for batch_start in range(0, len(pending), args.batch_size):
        batch_rows = pending[batch_start : batch_start + args.batch_size]
        pixel_rows = []
        target_rows = []
        for row in batch_rows:
            with Image.open(row["image_path"]) as source:
                image = source.convert("RGB").resize(
                    (args.eval_size, args.eval_size), Image.Resampling.BILINEAR
                )
            pixel_rows.append(
                towers[0].image_processor.preprocess(image, return_tensors="pt")["pixel_values"]
            )
            target_rows.append(
                target_processor.image_processor.preprocess(image, return_tensors="pt")["pixel_values"]
            )
        pixels = torch.cat(pixel_rows, dim=0)
        target_pixels = torch.cat(target_rows, dim=0)
        started_batch = time.perf_counter()
        with torch.inference_mode():
            ar = generate_pgot_caption_batch(
                model, tokenizer, pixels, max_new_tokens=args.ar_max_new_tokens,
                max_objects=args.max_objects, n_ovt_per_object=1,
            )
            if ar["caption_input_ids"].shape[1] > args.max_caption_tokens:
                raise ValueError("Generated caption exceeds replay limit; do not truncate")
            result = pgot_forward_eval(
                model, images=pixels, target_images=target_pixels,
                **{key: ar[key] for key in (
                    "caption_input_ids", "caption_attention_mask",
                    "ovt_positions_in_caption", "ovt_valid_mask",
                )},
            )
        batch_seconds = time.perf_counter() - started_batch
        for batch_index, (row, record) in enumerate(zip(batch_rows, ar["records"])):
            target = cache_path(cache_root, row["split"], row["image_id"])
            target.parent.mkdir(parents=True, exist_ok=True)
            bundle = compact_object_bundle(
                result,
                record["text"],
                record["ovt_count"],
                cache_dtype=args.cache_dtype,
                batch_index=batch_index,
            )
            payload = {
                "format_version": FORMAT_VERSION,
                "provenance_sha256": provenance_digest,
                "image_id": row["image_id"],
                "split": row["split"],
                "image_sha256": row["image_sha256"],
                "caption": record["text"],
                "ar_record": record,
                "seconds": batch_seconds / len(batch_rows),
                "batch_seconds": batch_seconds,
                "batch_size_requested": args.batch_size,
                "batch_size_effective": len(batch_rows),
                "batch_position": batch_index,
                **bundle,
            }
            temporary = target.with_name(target.name + f".tmp.{os.getpid()}")
            torch.save(payload, temporary)
            temporary.replace(target)
            processed += 1
            elapsed = time.perf_counter() - started_all
            rate = processed / elapsed
            print(
                f"Extract {processed}/{len(pending)} image={row['image_id']} "
                f"objects={record['object_count']} batch={len(batch_rows)} "
                f"rate={rate:.3f} image/s",
                flush=True,
            )
            del payload, bundle
        del result, ar, pixels, target_pixels, pixel_rows, target_rows
    write_json(
        cache_root / f"run_{rows[0]['split']}_shard{args.shard_index:03d}.json",
        {**provenance, "provenance_sha256": provenance_digest,
         "num_shards": args.num_shards, "shard_index": args.shard_index,
         "assigned_images": len(rows), "completed_images": len(rows),
         "elapsed_seconds": time.perf_counter() - started_all},
    )


def verify(args) -> None:
    import torch

    rows = list(iter_jsonl(Path(args.images_manifest)))
    if args.max_images:
        rows = rows[: args.max_images]
    missing = []
    object_counts = []
    total_bytes = 0
    for row in rows:
        path = cache_path(Path(args.cache_root), row["split"], row["image_id"])
        if not path.is_file():
            missing.append(row["image_id"])
            continue
        value = torch.load(path, map_location="cpu", weights_only=False)
        n = len(value["objects"])
        if value["semantic"].shape != (n, 1536):
            raise ValueError(f"Bad semantic shape: {path} {tuple(value['semantic'].shape)}")
        if value["visual_memory"].shape != (n, 8, 1536):
            raise ValueError(f"Bad visual-memory shape: {path} {tuple(value['visual_memory'].shape)}")
        if value["owner_probabilities"].shape != (n, 1024):
            raise ValueError(f"Bad ownership shape: {path}")
        if value["background_register_owners_excluded"] != 4:
            raise ValueError(f"Registers leaked into object cache: {path}")
        if value["register_memories_per_owner_excluded"] != 32:
            raise ValueError(f"Unexpected register-memory capacity: {path}")
        object_counts.append(n)
        total_bytes += path.stat().st_size
    summary = {
        "format_version": FORMAT_VERSION,
        "num_manifest_images": len(rows),
        "num_cached_images": len(rows) - len(missing),
        "num_missing_images": len(missing),
        "first_missing_image_ids": missing[:20],
        "mean_generated_objects": sum(object_counts) / len(object_counts) if object_counts else None,
        "cache_bytes": total_bytes,
        "complete": not missing,
    }
    print(json.dumps(summary, indent=2))
    if args.summary_path:
        write_json(Path(args.summary_path), summary)
    if missing and args.require_complete:
        raise SystemExit(1)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="stage", required=True)
    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--source_root", required=True)
    prepare.add_argument("--prepared_root", required=True)
    prepare.add_argument("--splits", nargs="+", choices=SPLITS, default=list(SPLITS))
    prepare.add_argument("--max_images", type=int, default=0)
    prepare.add_argument("--cpu_threads", type=int, default=3)

    extract_parser = subparsers.add_parser("extract")
    extract_parser.add_argument("--images_manifest", required=True)
    extract_parser.add_argument("--cache_root", required=True)
    extract_parser.add_argument("--model_path", required=True)
    extract_parser.add_argument("--dtype", choices=("fp32", "bf16"), default="fp32")
    extract_parser.add_argument("--cache_dtype", choices=("float16", "bfloat16"), default="float16")
    extract_parser.add_argument("--eval_size", type=int, default=512)
    extract_parser.add_argument("--max_objects", type=int, default=50)
    extract_parser.add_argument("--max_caption_tokens", type=int, default=4096)
    extract_parser.add_argument("--ar_max_new_tokens", type=int, default=2048)
    extract_parser.add_argument("--diffusion_inference_steps", type=int, default=10)
    extract_parser.add_argument("--cpu_threads", type=int, default=3)
    extract_parser.add_argument("--seed", type=int, default=0)
    extract_parser.add_argument("--num_shards", type=int, default=1)
    extract_parser.add_argument("--shard_index", type=int, default=0)
    extract_parser.add_argument("--batch_size", type=int, default=8)
    extract_parser.add_argument("--max_images", type=int, default=0)

    verify_parser = subparsers.add_parser("verify")
    verify_parser.add_argument("--images_manifest", required=True)
    verify_parser.add_argument("--cache_root", required=True)
    verify_parser.add_argument("--max_images", type=int, default=0)
    verify_parser.add_argument("--summary_path")
    verify_parser.add_argument("--require_complete", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.stage == "prepare":
        summaries = [prepare_split(args, split) for split in args.splits]
        write_json(Path(args.prepared_root) / "manifests" / "summary.json", summaries)
        print(json.dumps(summaries, indent=2))
    elif args.stage == "extract":
        if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
            raise ValueError("Require num_shards >= 1 and 0 <= shard_index < num_shards")
        if args.batch_size < 1:
            raise ValueError("Require batch_size >= 1")
        extract(args)
    else:
        verify(args)


if __name__ == "__main__":
    main()
