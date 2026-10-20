"""Train/evaluate a RefCOCO selector over frozen DeOR object representations.

The DeOR checkpoint is never updated.  Image-only AR discovery is run once per
image and cached as immutable candidate masks plus semantic object tokens.  A
small selector learns to map an original RefCOCO expression to one cached
candidate (or NONE).  Ground-truth masks are used only to derive train labels
and to score held-out predictions; they are never model inputs at inference.
"""

from __future__ import annotations

import argparse
import json
import math
import pickle
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from mllm_refcoco_common import load_samples, save_prediction, write_json  # noqa: E402
from pgot.eval.natural_object_selection import candidates_from_caption, digest  # noqa: E402
from pgot.eval.run_natural_object_selection import checkpoint_signature  # noqa: E402
from pgot.eval.run_refcoco_eval import (  # noqa: E402
    file_hash,
    make_queries,
    resolve_image,
    selected_references,
)
from pgot.eval.run_refcoco_object_selection import owner_prediction  # noqa: E402


VERSION = "refcoco-frozen-deor-semantic-selector-v1"
GEOMETRY_DIM = 5


def load_json(path):
    return json.loads(Path(path).read_text())


def cache_folder(root, image_id):
    return Path(root) / "images" / str(int(image_id))


def cache_complete(root, manifest, image):
    folder = cache_folder(root, image["image_id"])
    required = (folder / "metadata.json", folder / "prediction.npz", folder / "features.npz")
    if not all(path.is_file() for path in required):
        return False
    metadata = load_json(folder / "metadata.json")
    if metadata.get("fingerprint") != manifest["fingerprint"]:
        raise ValueError(f"Cache fingerprint mismatch: {folder}")
    if metadata.get("image_sha256") != image["image_sha256"]:
        raise ValueError(f"Cached image provenance mismatch: {folder}")
    return True


def atomic_npz(path, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.stem + ".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    temporary.replace(path)


def geometry_for_labels(prediction, candidates):
    height, width = prediction.shape
    values = []
    for candidate in candidates:
        label = candidate.get("prediction_label")
        ys, xs = np.where(prediction == int(label)) if label is not None else (np.array([]), np.array([]))
        if len(xs) == 0:
            values.append([0.0] * GEOMETRY_DIM)
            continue
        x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
        values.append([
            ((x0 + x1) * 0.5) / width,
            ((y0 + y1) * 0.5) / height,
            (x1 - x0) / width,
            (y1 - y0) / height,
            len(xs) / float(height * width),
        ])
    return np.asarray(values, dtype=np.float32).reshape(len(candidates), GEOMETRY_DIM)


def parse_prepare(args):
    refs_path = Path(args.data_root) / "refcoco" / "refs(unc).p"
    instances_path = Path(args.data_root) / "refcoco" / "instances.json"
    eval_manifest_path = Path(args.eval_manifest).resolve()
    overlap_source_path = Path(args.train_overlap_source).resolve()
    model_path = Path(args.model_path).resolve()
    for path in (
        refs_path, instances_path, eval_manifest_path, overlap_source_path,
        model_path / "config.json",
    ):
        if not path.exists():
            raise FileNotFoundError(path)

    config = {
        "version": VERSION,
        "model_path": str(model_path),
        "checkpoint_signature": checkpoint_signature(model_path),
        "refs_sha256": file_hash(refs_path),
        "instances_sha256": file_hash(instances_path),
        "eval_manifest": str(eval_manifest_path),
        "eval_manifest_sha256": file_hash(eval_manifest_path),
        "train_overlap_source": str(overlap_source_path),
        "train_overlap_source_sha256": file_hash(overlap_source_path),
        "max_train_expressions": args.max_train_expressions,
        "max_eval_expressions": args.max_eval_expressions,
        "seed": args.seed,
        "eval_size": args.eval_size,
        "max_objects": args.max_objects,
        "ar_max_new_tokens": args.ar_max_new_tokens,
        "max_caption_tokens": args.max_caption_tokens,
        "dtype": args.dtype,
    }
    fingerprint = digest(json.dumps(config, sort_keys=True))
    root = Path(args.output)
    manifest_path = root / "manifest.json"
    if manifest_path.is_file():
        previous = load_json(manifest_path)
        if previous["fingerprint"] != fingerprint:
            raise ValueError("Existing output uses a different configuration; choose a fresh OUTPUT_ROOT")
        print(f"Matching manifest already exists: {manifest_path}")
        return

    with refs_path.open("rb") as stream:
        refs = pickle.load(stream, encoding="latin1")
    instances = load_json(instances_path)
    images = {int(item["id"]): item for item in instances["images"]}
    annotations = {int(item["id"]): item for item in instances["annotations"]}

    train_refs = selected_references(refs, ["train"])
    train_queries = make_queries(train_refs, args.max_train_expressions)
    eval_source, eval_queries = load_samples(
        eval_manifest_path, subset="nonoverlap", train_path=overlap_source_path,
        max_samples=args.max_eval_expressions, splits=("val", "testA", "testB"),
    )

    train_image_ids = {int(query["image_id"]) for query in train_queries}
    eval_image_ids = {int(query["image_id"]) for query in eval_queries}
    overlap = sorted(train_image_ids & eval_image_ids)
    if overlap:
        raise ValueError(f"Official train/eval image leakage detected: {overlap[:10]}")

    all_queries = train_queries + eval_queries
    image_records = {}
    for feature_index, query in enumerate(all_queries):
        query["feature_index"] = feature_index
        iid = int(query["image_id"])
        ann = annotations[int(query["annotation_id"])]
        if int(ann["image_id"]) != iid:
            raise ValueError("RefCOCO annotation/image mismatch")
        if iid not in image_records:
            info = images[iid]
            image_path = resolve_image(args.image_root, info)
            image_records[iid] = {
                "image_id": iid,
                "image_path": str(image_path),
                "width": int(info["width"]),
                "height": int(info["height"]),
                "image_sha256": file_hash(image_path),
            }
        for key in ("image_path", "width", "height", "image_sha256"):
            query[key] = image_records[iid][key]

    hidden_size = int(load_json(model_path / "config.json")["hidden_size"])
    protocol = {
        "training": "official RefCOCO train expressions only",
        "evaluation": "existing DeOR non-overlap val/testA/testB manifest",
        "deor": "frozen image-only AR object discovery and frozen semantic object tokens",
        "selector_input": "raw expression embedding + candidate semantic token + candidate mask geometry",
        "inference_gt_inputs": "none",
        "training_gt_use": "best candidate IoU label or NONE; GT never enters DeOR/selector inference",
        "train_eval_image_overlap": 0,
    }
    manifest = {
        "fingerprint": fingerprint,
        "config": config,
        "protocol": protocol,
        "hidden_size": hidden_size,
        "images": list(image_records.values()),
        "train_queries": train_queries,
        "eval_queries": eval_queries,
        "eval_source_fingerprint": eval_source["fingerprint"],
    }
    root.mkdir(parents=True, exist_ok=True)
    write_json(manifest_path, manifest)
    feature_path = root / "query_features.npy"
    done_path = root / "query_features_done.npy"
    shape = (len(all_queries), hidden_size)
    np.lib.format.open_memmap(feature_path, mode="w+", dtype=np.float16, shape=shape)[:] = 0
    np.lib.format.open_memmap(done_path, mode="w+", dtype=np.uint8, shape=(shape[0],))[:] = 0
    print(json.dumps({
        "train_expressions": len(train_queries),
        "eval_expressions": len(eval_queries),
        "images": len(image_records),
        "train_images": len(train_image_ids),
        "eval_images": len(eval_image_ids),
        "hidden_size": hidden_size,
    }, indent=2))


def load_manifest(args):
    path = Path(args.output) / "manifest.json"
    if not path.is_file():
        raise FileNotFoundError(f"Run --stage prepare first: {path}")
    return load_json(path)


def model_args(args):
    class Values:
        pass
    value = Values()
    value.model_path = args.model_path
    value.dtype = args.dtype
    value.diffusion_inference_steps = 10
    return value


def extract(args):
    import torch
    from PIL import Image
    from pgot.eval.eval_recon_oracles import load_model_and_tokenizer
    from pgot.eval.pgot_inference import pgot_forward_eval
    from pgot.eval.run_eval import generate_pgot_caption_batch

    manifest = load_manifest(args)
    images = sorted(manifest["images"], key=lambda item: int(item["image_id"]))
    assigned = [item for index, item in enumerate(images) if index % args.num_shards == args.shard_index]
    pending = [item for item in assigned if not cache_complete(args.output, manifest, item)]
    if args.max_extract_images:
        pending = pending[:args.max_extract_images]
    if not pending:
        print(f"Extraction shard {args.shard_index}/{args.num_shards} is complete ({len(assigned)} images).")
        return
    if not torch.cuda.is_available():
        raise RuntimeError("Extraction requires CUDA")
    torch.set_num_threads(args.cpu_threads)
    torch.manual_seed(args.seed)
    model, tokenizer, _, _ = load_model_and_tokenizer(model_args(args))
    cfg = model.config
    if (int(cfg.pgot_e11_object_memories_per_owner), int(cfg.pgot_e11_register_memories_per_owner)) != (8, 32):
        raise ValueError("Expected DeOR Object8/Register32 checkpoint")
    if bool(getattr(cfg, "pgot_one_shot_direct_rae_query", False)):
        raise ValueError("Expected original MLLM-query checkpoint")
    for name, token in (("ovt", "<ovt>"), ("scene_end", "<scene_end>"),
                        ("thing", "<thing>"), ("stuff", "<stuff>")):
        setattr(model, f"pgot_{name}_token_id", tokenizer.convert_tokens_to_ids(token))
    towers = model.get_vision_tower_aux_list()
    started = time.monotonic()
    target_processor = towers[1] if len(towers) > 1 else towers[0]
    processed = 0
    for batch_start in range(0, len(pending), args.extract_batch_size):
        batch = pending[batch_start:batch_start + args.extract_batch_size]
        pixel_rows, target_rows = [], []
        for sample in batch:
            # The manifest hash was fixed during preparation.  Avoid hashing a
            # multi-MB JPEG again in every resumed extraction pass.
            with Image.open(sample["image_path"]) as source:
                if source.size != (sample["width"], sample["height"]):
                    raise ValueError("Image dimensions changed after manifest preparation")
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
        target = torch.cat(target_rows, dim=0)
        batch_started = time.monotonic()
        with torch.inference_mode():
            ar = generate_pgot_caption_batch(
                model, tokenizer, pixels, max_new_tokens=args.ar_max_new_tokens,
                max_objects=args.max_objects, n_ovt_per_object=1,
            )
            if ar["caption_input_ids"].shape[1] > args.max_caption_tokens:
                raise ValueError("Generated caption exceeds replay limit")
            result = pgot_forward_eval(
                model, images=pixels, target_images=target,
                **{key: ar[key] for key in (
                    "caption_input_ids", "caption_attention_mask",
                    "ovt_positions_in_caption", "ovt_valid_mask",
                )},
            )
        batch_seconds = time.monotonic() - batch_started
        for batch_index, (sample, record) in enumerate(zip(batch, ar["records"])):
            parsed_candidates = candidates_from_caption(
                record["text"], record["ovt_count"], sample["image_id"], args.seed
            )
            # A generation stopped by the token limit may leave a trailing
            # textual object without an <ovt>.  It has no semantic slot or
            # ownership mask and therefore cannot be a DeOR representation
            # candidate for the learned selector.
            candidates = [
                candidate for candidate in parsed_candidates
                if candidate.get("slot_index") is not None
                and candidate.get("prediction_label") is not None
            ]
            hidden_size = int(manifest["hidden_size"])
            if record["ovt_count"]:
                prediction = owner_prediction(
                    result["ovt_object_probs"][batch_index:batch_index + 1],
                    result["ovt_void_probs"][batch_index:batch_index + 1],
                    ar["ovt_valid_mask"][batch_index:batch_index + 1],
                    candidates, sample["height"], sample["width"],
                )
                slot_indices = np.asarray([int(candidate["slot_index"]) for candidate in candidates], dtype=np.int64)
                semantic = result["semantic_slots"][batch_index, slot_indices].float().cpu().numpy().astype(np.float16)
            else:
                prediction = np.zeros((sample["height"], sample["width"]), dtype=np.uint16)
                slot_indices = np.zeros((0,), dtype=np.int64)
                semantic = np.zeros((0, hidden_size), dtype=np.float16)
            if semantic.shape != (len(candidates), int(manifest["hidden_size"])):
                raise ValueError("Unexpected semantic candidate feature shape")
            geometry = geometry_for_labels(prediction, candidates)
            labels = np.asarray([
                int(candidate["prediction_label"]) if candidate.get("prediction_label") is not None else 0
                for candidate in candidates
            ], dtype=np.int64)
            folder = cache_folder(args.output, sample["image_id"])
            folder.mkdir(parents=True, exist_ok=True)
            atomic_npz(folder / "prediction.npz", prediction=prediction)
            atomic_npz(
                folder / "features.npz", semantic=semantic, geometry=geometry,
                prediction_labels=labels, slot_indices=slot_indices,
            )
            write_json(folder / "metadata.json", {
                "fingerprint": manifest["fingerprint"],
                "image_sha256": sample["image_sha256"],
                "caption": record["text"],
                "caption_sha256": digest(record["text"]),
                "ovt_count": record["ovt_count"],
                "candidates": candidates,
                "dropped_unbound_candidates": len(parsed_candidates) - len(candidates),
                "scene_end_generated": record["scene_end_generated"],
                "batch_size_requested": args.extract_batch_size,
                "batch_size_effective": len(batch),
                "seconds": batch_seconds / len(batch),
                "inference_inputs": "image only; no RefCOCO expression or annotation",
            })
            processed += 1
            if processed % 25 == 0 or processed == len(pending):
                elapsed = time.monotonic() - started
                print(
                    f"extract shard={args.shard_index} {processed}/{len(pending)} "
                    f"image={sample['image_id']} owners={len(candidates)} batch={len(batch)} "
                    f"rate={processed/max(elapsed, 1e-6):.3f} image/s",
                    flush=True,
                )
        del result, ar, pixels, target, pixel_rows, target_rows


def encode_queries(args):
    import torch
    from pgot.eval.eval_recon_oracles import load_model_and_tokenizer

    manifest = load_manifest(args)
    queries = manifest["train_queries"] + manifest["eval_queries"]
    feature_path = Path(args.output) / "query_features.npy"
    done_path = Path(args.output) / "query_features_done.npy"
    features = np.lib.format.open_memmap(feature_path, mode="r+")
    done = np.lib.format.open_memmap(done_path, mode="r+")
    assigned = [
        query for query in queries
        if int(query["feature_index"]) % args.num_shards == args.shard_index
        and not bool(done[int(query["feature_index"])])
    ]
    if not assigned:
        print(f"Query encoding shard {args.shard_index}/{args.num_shards} is complete.")
        return
    if not torch.cuda.is_available():
        raise RuntimeError("Query encoding requires CUDA")
    torch.set_num_threads(args.cpu_threads)
    torch.manual_seed(args.seed)
    model, tokenizer, device, _ = load_model_and_tokenizer(model_args(args))
    model.eval()
    started = time.monotonic()
    for start in range(0, len(assigned), args.query_batch_size):
        block = assigned[start:start + args.query_batch_size]
        encoded = tokenizer(
            [query["shared_text"] for query in block], padding=True,
            add_special_tokens=True, return_tensors="pt",
        )
        input_ids = encoded["input_ids"].to(device)
        attention = encoded["attention_mask"].to(device)
        with torch.inference_mode():
            output = model(
                input_ids=input_ids, attention_mask=attention,
                output_hidden_states=True, return_dict=True, use_cache=False,
            )
            hidden = output.hidden_states[-1].float()
            weights = attention.unsqueeze(-1).float()
            pooled = (hidden * weights).sum(1) / weights.sum(1).clamp_min(1.0)
            pooled = torch.nn.functional.normalize(pooled, dim=-1)
        indices = np.asarray([int(query["feature_index"]) for query in block])
        features[indices] = pooled.cpu().numpy().astype(np.float16)
        features.flush()
        done[indices] = 1
        done.flush()
        complete = min(start + len(block), len(assigned))
        if complete % (args.query_batch_size * 20) == 0 or complete == len(assigned):
            elapsed = time.monotonic() - started
            print(
                f"encode shard={args.shard_index} {complete}/{len(assigned)} "
                f"rate={complete/max(elapsed, 1e-6):.1f} expression/s",
                flush=True,
            )


def verify(args):
    manifest = load_manifest(args)
    missing = [
        int(image["image_id"]) for image in manifest["images"]
        if not cache_complete(args.output, manifest, image)
    ]
    done = np.load(Path(args.output) / "query_features_done.npy", mmap_mode="r")
    incomplete_queries = int((done == 0).sum())
    status = {
        "images": len(manifest["images"]),
        "missing_image_caches": len(missing),
        "missing_image_ids_first_20": missing[:20],
        "query_features": len(done),
        "incomplete_query_features": incomplete_queries,
    }
    write_json(Path(args.output) / "cache_status.json", status)
    print(json.dumps(status, indent=2))
    if missing or incomplete_queries:
        raise RuntimeError("Cache is incomplete; resume extract/encode shards")


def build_training_records(args, manifest):
    import torch
    from pycocotools.coco import COCO

    path = Path(args.output) / "train_records.json"
    if path.is_file():
        value = load_json(path)
        if value["fingerprint"] != manifest["fingerprint"] or value["min_label_iou"] != args.min_label_iou:
            raise ValueError("Existing train labels use a different protocol")
        return value["records"], value["summary"]
    coco = COCO(args.instances)
    groups = defaultdict(list)
    for query in manifest["train_queries"]:
        groups[int(query["image_id"])].append(query)
    records = []
    coverage50 = 0
    nonnull = 0
    for number, (image_id, queries) in enumerate(sorted(groups.items()), 1):
        folder = cache_folder(args.output, image_id)
        prediction = np.load(folder / "prediction.npz")["prediction"]
        features = np.load(folder / "features.npz")
        labels = features["prediction_labels"].astype(np.int64)
        areas = np.asarray([(prediction == label).sum() for label in labels], dtype=np.int64)
        by_annotation = defaultdict(list)
        for query in queries:
            by_annotation[int(query["annotation_id"])].append(query)
        for annotation_id, sentences in by_annotation.items():
            target = coco.annToMask(coco.anns[annotation_id]).astype(bool)
            target_area = int(target.sum())
            if len(labels):
                counts = np.bincount(prediction[target].reshape(-1), minlength=int(labels.max()) + 1)
                intersections = np.asarray([counts[label] if label < len(counts) else 0 for label in labels])
                unions = areas + target_area - intersections
                ious = np.divide(
                    intersections, unions, out=np.zeros(len(labels), dtype=np.float64), where=unions > 0
                )
                best = int(np.argmax(ious))
                best_iou = float(ious[best])
            else:
                best, best_iou = 0, 0.0
            label_index = best if best_iou >= args.min_label_iou else len(labels)
            coverage50 += len(sentences) * int(best_iou >= 0.5)
            nonnull += len(sentences) * int(label_index < len(labels))
            for query in sentences:
                records.append({
                    "feature_index": int(query["feature_index"]),
                    "image_id": image_id,
                    "annotation_id": annotation_id,
                    "label": label_index,
                    "num_candidates": int(len(labels)),
                    "best_iou": best_iou,
                })
        if number % 1000 == 0:
            print(f"labeled {number}/{len(groups)} train images", flush=True)
    summary = {
        "expressions": len(records),
        "images": len(groups),
        "nonnull_labels": nonnull,
        "nonnull_rate": nonnull / len(records),
        "oracle_coverage_at_0.5": coverage50 / len(records),
        "min_label_iou": args.min_label_iou,
    }
    write_json(path, {"fingerprint": manifest["fingerprint"], "min_label_iou": args.min_label_iou,
                      "summary": summary, "records": records})
    return records, summary


def selector_class():
    import torch
    from torch import nn
    import torch.nn.functional as F

    class FrozenDeORSelector(nn.Module):
        def __init__(self, input_dim, projection_dim=384, dropout=0.1):
            super().__init__()
            self.input_dim = int(input_dim)
            self.projection_dim = int(projection_dim)
            self.query_norm = nn.LayerNorm(input_dim)
            self.candidate_norm = nn.LayerNorm(input_dim)
            self.query_projection = nn.Linear(input_dim, projection_dim)
            self.candidate_projection = nn.Linear(input_dim, projection_dim)
            self.geometry_projection = nn.Sequential(
                nn.Linear(GEOMETRY_DIM, projection_dim), nn.GELU(), nn.Linear(projection_dim, projection_dim)
            )
            self.pair_score = nn.Sequential(
                nn.Linear(projection_dim * 4 + GEOMETRY_DIM, projection_dim),
                nn.GELU(), nn.Dropout(dropout), nn.Linear(projection_dim, 1),
            )
            self.none_score = nn.Sequential(
                nn.Linear(projection_dim, projection_dim), nn.GELU(), nn.Linear(projection_dim, 1)
            )
            self.logit_scale = nn.Parameter(torch.tensor(math.log(10.0)))

        def forward(self, query, candidates, geometry, valid):
            q = F.normalize(self.query_projection(self.query_norm(query)), dim=-1)
            c = F.normalize(
                self.candidate_projection(self.candidate_norm(candidates)) + self.geometry_projection(geometry),
                dim=-1,
            )
            expanded = q[:, None, :].expand_as(c)
            pair = torch.cat((expanded, c, expanded * c, (expanded - c).abs(), geometry), dim=-1)
            scores = self.logit_scale.exp().clamp(max=100.0) * (expanded * c).sum(-1)
            scores = scores + self.pair_score(pair).squeeze(-1)
            scores = scores.masked_fill(~valid, torch.finfo(scores.dtype).min)
            return torch.cat((scores, self.none_score(q)), dim=1)

    return FrozenDeORSelector


def load_candidate_bank(args, image_ids):
    bank = {}
    for image_id in sorted(set(int(value) for value in image_ids)):
        folder = cache_folder(args.output, image_id)
        with np.load(folder / "features.npz") as saved:
            bank[image_id] = (
                saved["semantic"].astype(np.float32), saved["geometry"].astype(np.float32),
                saved["prediction_labels"].astype(np.int64),
            )
    return bank


def make_batches(records, query_features, bank, batch_size, shuffle, seed):
    order = list(range(len(records)))
    if shuffle:
        random.Random(seed).shuffle(order)
    for start in range(0, len(order), batch_size):
        chosen = [records[index] for index in order[start:start + batch_size]]
        maximum = max((record["num_candidates"] for record in chosen), default=0)
        maximum = max(maximum, 1)
        dim = query_features.shape[1]
        query = np.empty((len(chosen), dim), dtype=np.float32)
        candidate = np.zeros((len(chosen), maximum, dim), dtype=np.float32)
        geometry = np.zeros((len(chosen), maximum, GEOMETRY_DIM), dtype=np.float32)
        valid = np.zeros((len(chosen), maximum), dtype=bool)
        labels = np.empty((len(chosen),), dtype=np.int64)
        for row, record in enumerate(chosen):
            query[row] = query_features[record["feature_index"]]
            semantic, geom, _ = bank[record["image_id"]]
            count = len(semantic)
            candidate[row, :count] = semantic
            geometry[row, :count] = geom
            valid[row, :count] = True
            labels[row] = record["label"] if record["label"] < count else maximum
        yield query, candidate, geometry, valid, labels


def train(args):
    import torch
    import torch.nn.functional as F

    manifest = load_manifest(args)
    verify(args)
    records, label_summary = build_training_records(args, manifest)
    query_features = np.load(Path(args.output) / "query_features.npy", mmap_mode="r")
    bank = load_candidate_bank(args, [record["image_id"] for record in records])
    train_records = [record for record in records if int(record["image_id"]) % 20 != 0]
    validation_records = [record for record in records if int(record["image_id"]) % 20 == 0]
    if not validation_records:
        validation_records = train_records[: min(len(train_records), max(1, args.batch_size))]
    if not train_records:
        train_records = records
    device = torch.device("cuda")
    Selector = selector_class()
    model = Selector(manifest["hidden_size"], args.projection_dim, args.dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scaler = None
    best = None
    history = []
    torch.manual_seed(args.seed)
    for epoch in range(args.epochs):
        model.train()
        total_loss = total_correct = total = 0
        for arrays in make_batches(
            train_records, query_features, bank, args.batch_size, True, args.seed + epoch
        ):
            q, c, g, valid, labels = [torch.from_numpy(value).to(device) for value in arrays]
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(q, c, g, valid)
                loss = F.cross_entropy(logits.float(), labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.detach().item() * len(q)
            total_correct += int((logits.argmax(-1) == labels).sum())
            total += len(q)
        model.eval()
        val_loss = val_correct = val_total = 0
        with torch.inference_mode():
            for arrays in make_batches(validation_records, query_features, bank, args.batch_size, False, args.seed):
                q, c, g, valid, labels = [torch.from_numpy(value).to(device) for value in arrays]
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    logits = model(q, c, g, valid)
                loss = F.cross_entropy(logits.float(), labels)
                val_loss += loss.item() * len(q)
                val_correct += int((logits.argmax(-1) == labels).sum())
                val_total += len(q)
        row = {
            "epoch": epoch + 1,
            "train_loss": total_loss / total,
            "train_accuracy": total_correct / total,
            "validation_loss": val_loss / val_total,
            "validation_accuracy": val_correct / val_total,
        }
        history.append(row)
        print(json.dumps(row), flush=True)
        if best is None or row["validation_accuracy"] > best["validation_accuracy"]:
            best = row
            torch.save({
                "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
                "input_dim": manifest["hidden_size"],
                "projection_dim": args.projection_dim,
                "dropout": args.dropout,
                "fingerprint": manifest["fingerprint"],
                "best_epoch": epoch + 1,
                "validation_accuracy": row["validation_accuracy"],
            }, Path(args.output) / "selector.pt")
    summary = {
        "protocol": manifest["protocol"],
        "label_summary": label_summary,
        "train_expressions": len(train_records),
        "heldout_train_expressions": len(validation_records),
        "best": best,
        "history": history,
        "trainable_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "deor_parameters_updated": 0,
    }
    write_json(Path(args.output) / "train_summary.json", summary)


def evaluate(args):
    import torch

    manifest = load_manifest(args)
    verify(args)
    checkpoint = torch.load(Path(args.output) / "selector.pt", map_location="cpu", weights_only=True)
    if checkpoint["fingerprint"] != manifest["fingerprint"]:
        raise ValueError("Selector/manifest mismatch")
    Selector = selector_class()
    model = Selector(
        checkpoint["input_dim"], checkpoint["projection_dim"], checkpoint["dropout"]
    )
    model.load_state_dict(checkpoint["state_dict"])
    device = torch.device("cuda")
    model.to(device).eval()
    query_features = np.load(Path(args.output) / "query_features.npy", mmap_mode="r")
    queries = manifest["eval_queries"]
    bank = load_candidate_bank(args, [query["image_id"] for query in queries])
    records = [{
        "feature_index": int(query["feature_index"]), "image_id": int(query["image_id"]),
        "num_candidates": len(bank[int(query["image_id"])][0]), "label": 0,
    } for query in queries]
    selected = {}
    offset = 0
    with torch.inference_mode():
        for arrays in make_batches(records, query_features, bank, args.batch_size, False, args.seed):
            q, c, g, valid, _ = [torch.from_numpy(value).to(device) for value in arrays]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(q, c, g, valid)
            probabilities = logits.float().softmax(-1)
            choices = probabilities.argmax(-1).cpu().numpy()
            confidence = probabilities.max(-1).values.cpu().numpy()
            for local, choice in enumerate(choices):
                selected[offset + local] = (int(choice), float(confidence[local]))
            offset += len(choices)
    output = Path(args.output) / "predictions"
    by_image = {}
    empty = 0
    for index, query in enumerate(queries):
        iid = int(query["image_id"])
        if iid not in by_image:
            folder = cache_folder(args.output, iid)
            prediction = np.load(folder / "prediction.npz")["prediction"]
            metadata = load_json(folder / "metadata.json")
            by_image[iid] = (prediction, metadata)
        prediction, metadata = by_image[iid]
        choice, confidence = selected[index]
        candidates = metadata["candidates"]
        if choice >= len(candidates):
            mask = np.zeros(prediction.shape, dtype=bool)
            selected_id = None
            empty += 1
        else:
            label = candidates[choice].get("prediction_label")
            mask = prediction == int(label) if label is not None else np.zeros(prediction.shape, dtype=bool)
            selected_id = candidates[choice]["id"]
            empty += int(not mask.any())
        save_prediction(
            output, query, mask, "deor_refcoco_trained_selector",
            selected_candidate_index=None if selected_id is None else choice,
            selected_candidate_id=selected_id, confidence=confidence,
            selector_checkpoint=str((Path(args.output) / "selector.pt").resolve()),
            gt_inference_inputs=[],
        )
        if (index + 1) % 2000 == 0:
            print(f"saved {index+1}/{len(queries)} selector predictions", flush=True)
    write_json(Path(args.output) / "prediction_info.json", {
        "expressions": len(queries), "empty_masks": empty,
        "selector_checkpoint": str((Path(args.output) / "selector.pt").resolve()),
        "deor_frozen": True, "gt_inference_inputs": [],
    })


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("prepare", "extract", "encode", "verify", "train", "evaluate"), required=True)
    parser.add_argument("--output", default=str(PROJECT_ROOT / "outputs/refcoco_frozen_deor_selector"))
    parser.add_argument("--model-path", default=str(PROJECT_ROOT / "checkpoints/pgot_oneshot_memory8_register32_noprior/checkpoint-5000"))
    parser.add_argument("--eval-manifest", default=str(PROJECT_ROOT / "outputs/eval_deor_ctrlo_refcoco_direct/manifest.json"))
    parser.add_argument("--train-overlap-source", default=str(PROJECT_ROOT / "data/pgot_pix2cap_thing_train.jsonl"))
    parser.add_argument("--data-root", default="/home/jovyan/data/coco")
    parser.add_argument("--image-root", default="/home/jovyan/data/coco")
    parser.add_argument("--instances", default="/home/jovyan/data/coco/refcoco/instances.json")
    parser.add_argument("--dtype", choices=("fp32", "bf16"), default="fp32")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eval-size", type=int, default=512)
    parser.add_argument("--max-objects", type=int, default=50)
    parser.add_argument("--ar-max-new-tokens", type=int, default=2048)
    parser.add_argument("--max-caption-tokens", type=int, default=4096)
    parser.add_argument("--max-train-expressions", type=int, default=0)
    parser.add_argument("--max-eval-expressions", type=int, default=0)
    parser.add_argument("--max-extract-images", type=int, default=0,
                        help="Debug/resume aid: process at most this many pending images per shard")
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--query-batch-size", type=int, default=32)
    parser.add_argument("--extract-batch-size", type=int, default=8)
    parser.add_argument("--cpu-threads", type=int, default=3)
    parser.add_argument("--min-label-iou", type=float, default=0.1)
    parser.add_argument("--projection-dim", type=int, default=384)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        parser.error("invalid shard index/count")
    if min(args.eval_size, args.max_objects, args.ar_max_new_tokens, args.max_caption_tokens,
           args.query_batch_size, args.extract_batch_size, args.cpu_threads,
           args.projection_dim, args.epochs, args.batch_size) < 1:
        parser.error("sizes must be positive")
    if args.max_train_expressions < 0 or args.max_eval_expressions < 0 or args.max_extract_images < 0:
        parser.error("sample limits cannot be negative")
    if not 0 <= args.min_label_iou <= 1:
        parser.error("min-label-iou must be in [0,1]")
    return args


def main():
    args = parse_args()
    if args.stage == "prepare":
        parse_prepare(args)
    elif args.stage == "extract":
        extract(args)
    elif args.stage == "encode":
        encode_queries(args)
    elif args.stage == "verify":
        verify(args)
    elif args.stage == "train":
        train(args)
    elif args.stage == "evaluate":
        evaluate(args)


if __name__ == "__main__":
    main()
