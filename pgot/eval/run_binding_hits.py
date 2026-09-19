"""Evaluate annotation-derived COCO query binding; not GT-free AR discovery."""

import argparse
import hashlib
import json
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from pgot.eval.binding_coco_dataset import BindingCocoDataset, BindingCollator, QUERY_MODES
from pgot.eval.binding_hits import score_binding_hits
from pgot.eval.voc_object_counts import count_bucket


def aggregate(rows):
    queries = [query for row in rows for query in row["queries"]]
    n = len(queries)
    hits = sum(query["hit"] for query in queries)
    return {"num_images": len(rows), "num_images_with_queries": sum(bool(row["queries"]) for row in rows),
            "num_queries": n, "hits": hits, "binding_hits": hits / n if n else None,
            "binding_hits_percent": 100 * hits / n if n else None,
            "binding_hits_iou50": sum(q["hit_iou50"] for q in queries) / n if n else None,
            "positive_iou_binding_hits": sum(q["positive_iou_hit"] for q in queries) / n if n else None,
            "zero_iou_hits": sum(q["hit"] and q["target_iou"] == 0 for q in queries),
            "target_mIoU": sum(q["target_iou"] for q in queries) / n if n else None,
            "binding_hits_image_mean": (sum(sum(q["hit"] for q in r["queries"]) / len(r["queries"])
                                            for r in rows if r["queries"]) / sum(bool(r["queries"]) for r in rows)
                                        if n else None)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model_path", required=True)
    p.add_argument("--data_root", default="/home/jovyan/data/coco")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--query_mode", choices=QUERY_MODES, default="class_point")
    p.add_argument("--binding_route", choices=["direct", "ar_router"], default="direct")
    p.add_argument("--ar_max_new_tokens", type=int, default=512)
    p.add_argument("--center_source", choices=["mask", "bbox"], default="mask")
    p.add_argument("--max_bindings", type=int, default=7, help="0=all visible instances")
    p.add_argument("--max_objects", type=int, default=64)
    p.add_argument("--max_caption_tokens", type=int, default=4096)
    p.add_argument("--eval_size", type=int, default=512)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--max_samples", type=int)
    p.add_argument("--image_ids", type=int, nargs="+", help="Explicit IDs for smoke, no max_samples")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dtype", choices=["fp32", "bf16"], default="fp32")
    p.add_argument("--ignore_overlaps", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()
    if args.batch_size < 1 or args.num_workers < 0 or args.seed < 0 or args.eval_size < 1:
        p.error("Invalid batch/worker/seed/resolution")
    if not 0 <= args.max_bindings <= args.max_objects or args.max_objects < 1:
        p.error("Require 0 <= max_bindings <= max_objects and max_objects >= 1")
    if args.image_ids is not None and (args.max_samples is not None or len(set(args.image_ids)) != len(args.image_ids)):
        p.error("image_ids must be unique and cannot be combined with max_samples")
    if args.max_samples is not None and args.max_samples < 1:
        p.error("max_samples must be positive")
    if args.ar_max_new_tokens < 1:
        p.error("ar_max_new_tokens must be positive")
    output = Path(args.output_dir)
    if any((output / name).exists() for name in ["summary.json", "per_image.jsonl", "query_manifest.jsonl"]) and not args.overwrite:
        p.error("Output already contains evaluation artifacts; use a different directory or --overwrite")
    output.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    # Reuse the established loader (including adapter injection and full reload).
    from pgot.eval.eval_recon_oracles import load_model_and_tokenizer
    from pgot.eval.pgot_inference import pgot_forward_eval
    from pgot.model.pgot_utils import build_pred_mask_ovt_owner_eval
    if args.binding_route == "ar_router":
        from pgot.eval.run_eval import generate_pgot_caption_batch
        from pgot.eval.ar_binding_router import parse_ar_slots, route_ar_queries, routed_prediction
    args.diffusion_inference_steps = 50  # Loader compatibility only; no diffusion is run.
    start = time.time()
    model, tokenizer, device, dtype = load_model_and_tokenizer(args)
    towers = model.get_vision_tower_aux_list()
    dataset = BindingCocoDataset(args.data_root, tokenizer, towers[0].image_processor,
                                 towers[1].image_processor if len(towers) > 1 else towers[0].image_processor,
                                 eval_size=args.eval_size, max_bindings=args.max_bindings, max_objects=args.max_objects,
                                 max_caption_tokens=args.max_caption_tokens, seed=args.seed, query_mode=args.query_mode,
                                 ignore_overlaps=args.ignore_overlaps, center_source=args.center_source)
    if args.image_ids is not None:
        lookup = {iid: i for i, iid in enumerate(dataset.ids)}
        if any(iid not in lookup for iid in args.image_ids):
            raise ValueError("Smoke ID not in official val2017")
        dataset = Subset(dataset, [lookup[iid] for iid in args.image_ids])
    elif args.max_samples is not None:
        dataset = Subset(dataset, range(min(args.max_samples, len(dataset))))
    loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers,
                        collate_fn=BindingCollator(tokenizer.pad_token_id), shuffle=False)
    records = []
    all_ar_records = []
    query_identity_hash = hashlib.sha256()
    with (output / "per_image.jsonl").open("w") as per_image, (output / "query_manifest.jsonl").open("w") as manifest:
        for batch in tqdm(loader, desc="Binding Hits"):
            samples = batch["samples"]
            ar_batch = None
            # AR always sees images ONLY; query inputs go exclusively to the
            # CPU router AFTER generation, ownership readout, and hard masks.
            if args.binding_route == "ar_router":
                with torch.inference_mode():
                    ar_batch = generate_pgot_caption_batch(model, tokenizer, batch["images"],
                                max_new_tokens=args.ar_max_new_tokens, max_objects=args.max_objects, n_ovt_per_object=1)
                caption_ids = ar_batch["caption_input_ids"]
                caption_attention = ar_batch["caption_attention_mask"]
                ovt_positions = ar_batch["ovt_positions_in_caption"]
                ovt_valid = ar_batch["ovt_valid_mask"]
            else:
                caption_ids, caption_attention = batch["caption_input_ids"], batch["caption_attention_mask"]
                ovt_positions, ovt_valid = batch["ovt_positions_in_caption"], batch["ovt_valid_mask"]
            # An empty scene has no query: record it, but BH has no denominator.
            if bool(ovt_valid.any()):
                with torch.inference_mode():
                    out = pgot_forward_eval(model, images=batch["images"], target_images=batch["target_images"],
                                            caption_input_ids=caption_ids, caption_attention_mask=caption_attention,
                                            ovt_positions_in_caption=ovt_positions, ovt_valid_mask=ovt_valid)
                    pred = build_pred_mask_ovt_owner_eval(out["ovt_object_probs"], out["ovt_void_probs"],
                                                          ovt_valid.to(device), ovt_valid.to(device), args.eval_size, 1).cpu().numpy()
                del out
            else:
                pred = np.zeros((len(samples), args.eval_size, args.eval_size), dtype=np.int64)
            for b, sample in enumerate(samples):
                # Exclude text so prompt variants can verify identical GT targets,
                # exact point values, distractors, and query order.
                identity = {"image_id": sample["image_id"], "gt_instances": sample["objects"],
                            "selected": sample["selected"]}
                query_identity_hash.update((json.dumps(identity, sort_keys=True, separators=(",", ":")) + "\n").encode())
                ar_metadata, route_records = {}, None
                if ar_batch is not None:
                    ar_record = ar_batch["records"][b]
                    slots = parse_ar_slots(ar_record["text"], int(ovt_valid[b].sum()))
                    # Deliberately strip annotation IDs, GT indices, masks,
                    # category IDs, descriptions, and other scoring metadata.
                    query_inputs = [{"category": query["category"],
                                     "center_normalized_xy": [round(v, 4) for v in query["center_normalized_xy"]]}
                                    for query in sample["queries"]]
                    labels, route_records, candidates = route_ar_queries(pred[b], slots, query_inputs,
                                    center_source=args.center_source, use_point=args.query_mode != "class_only")
                    routed = routed_prediction(pred[b], labels)
                    query_labels = [i + 1 if label is not None else None for i, label in enumerate(labels)]
                    scored = score_binding_hits(routed, sample["targets"], sample["selected"], sample["ignore"],
                                                query_labels=query_labels)
                    all_scored = score_binding_hits(pred[b], sample["targets"], sample["selected"], sample["ignore"],
                                                    query_labels=labels, num_predictions=len(slots))
                    all_ar_records.append({"queries": all_scored})
                    ar_metadata = {"ar_caption": ar_record, "ar_slots": slots, "router_candidates": candidates,
                                   "router_query_inputs": query_inputs,
                                   "num_routed_queries": sum(label is not None for label in labels),
                                   "num_unmatched_queries": sum(label is None for label in labels)}
                else:
                    scored = score_binding_hits(pred[b], sample["targets"], sample["selected"], sample["ignore"])
                queries = []
                for query, score in zip(sample["queries"], scored):
                    matched = score["matched_gt_index"]
                    entry = dict(query, **{k: v for k, v in score.items() if k not in query},
                                 matched_annotation_id=sample["objects"][matched]["annotation_id"] if matched is not None else None)
                    if route_records is not None:
                        entry.update(route_records[score["query_index"]])
                        entry["all_ar_slots_score"] = all_scored[score["query_index"]]
                    queries.append(entry)
                row = {"image_id": sample["image_id"], "gt_objects": len(sample["objects"]),
                       "num_queries": len(queries), "queries": queries, **ar_metadata}
                records.append(row)
                per_image.write(json.dumps(row, allow_nan=False) + "\n")
                manifest.write(json.dumps({"image_id": sample["image_id"], "image_path": sample["image_path"],
                                            "caption": sample["caption"], "gt_instances": sample["objects"],
                                            "queries": sample["queries"], **ar_metadata}, allow_nan=False) + "\n")
            per_image.flush()
            manifest.flush()
    if len(records) != len(dataset) or len({r["image_id"] for r in records}) != len(records):
        raise RuntimeError("Did not evaluate each selected image exactly once")
    summary = aggregate(records)
    summary.update(dataset="coco", dataset_total_samples=5000, full_split_evaluated=len(records) == 5000,
                   checkpoint=str(Path(args.model_path).resolve()), query_mode=args.query_mode,
                   center_source=args.center_source, max_bindings=args.max_bindings, max_objects=args.max_objects,
                   seed=args.seed, batch_size=args.batch_size, dtype=args.dtype, device=str(device),
                   metric_resolution=args.eval_size, ignore_overlaps=args.ignore_overlaps,
                   evaluation_mode="query_conditioned_not_AR", pix2cap_used=False,
                   gt_mask_passed_to_model=False, gt_class_passed_to_model=True,
                   gt_point_passed_to_model=args.query_mode != "class_only", register_eval_route="unrestricted",
                   coordinate_convention="normalized cropped-image x left-to-right, y top-to-bottom",
                   query_template_version=1, query_order="image-ID-seeded random permutation",
                   natural_position_grid="fixed 3x3 thirds" if args.query_mode == "class_point_natural" else None,
                   query_identity_sha256=query_identity_hash.hexdigest(),
                   gt_instances="all noncrowd nonignored positive-area foreground instances visible after PIL CODA crop",
                   matching="Hungarian maximum total IoU over query predictions + aggregate register background vs all GT instances",
                   empty_scene_policy="record all images; zero-query images excluded from BH denominator",
                   zero_query_images=sum(not r["queries"] for r in records),
                   ctrlo_exact_reproduction=False,
                   comparison_note="CTRL-O-style scoring; coordinates encoded as PGOT text, not CTRL-O point embeddings; crop/query selection differ",
                   elapsed_sec=time.time() - start)
    ann_path = Path(args.data_root) / "annotations/instances_val2017.json"
    summary["annotation_sha256"] = hashlib.sha256(ann_path.read_bytes()).hexdigest()
    summary["by_object_count"] = {bucket: aggregate([r for r in records if count_bucket(r["gt_objects"]) == bucket])
                                  for bucket in ["0", "1-3", "4-6", "7-10", "11+"]
                                  if any(count_bucket(r["gt_objects"]) == bucket for r in records)}
    cfg = model.config
    summary["model_info"] = {key: getattr(cfg, key, None) for key in
                             ["pgot_e11_object_memories_per_owner", "pgot_e11_register_memories_per_owner",
                              "pgot_one_shot_writer_owner_prior", "pgot_n_register"]}
    summary["binding_route"] = args.binding_route
    if args.binding_route == "ar_router":
        summary.update(evaluation_mode="AR_discovery_then_class_point_routing", caption_mode="autoregressive",
                       gt_class_passed_to_model=False, gt_point_passed_to_model=False,
                       gt_class_passed_to_router=True, gt_point_passed_to_router=args.query_mode != "class_only",
                       gt_mask_passed_to_router=False, gt_annotation_id_passed_to_router=False,
                       ar_max_new_tokens=args.ar_max_new_tokens, ar_max_objects=args.max_objects,
                       router="fixed lexical class aliases + Hungarian one-to-one predicted-mask-center squared distance",
                       router_coordinate_precision_decimals=4,
                       router_uses_query_description=False,
                       matching="Hungarian IoU: routed query regions + residual background vs all visible GT instances",
                       background_policy="merge unselected AR regions and register background for primary Q+BG metric",
                       all_ar_slots_metric=aggregate(all_ar_records),
                       all_ar_slots_matching="secondary: all AR regions + register background vs all GT; query labels fixed by router",
                       num_routed_queries=sum(r["num_routed_queries"] for r in records),
                       num_unmatched_queries=sum(r["num_unmatched_queries"] for r in records),
                       ar_format_valid_rate=sum(r["ar_caption"]["format_valid"] for r in records) / len(records),
                       ar_scene_end_generated_rate=sum(r["ar_caption"]["scene_end_generated"] for r in records) / len(records),
                       ar_generated_object_count_mean=sum(r["ar_caption"]["object_count"] for r in records) / len(records),
                       empty_scene_policy="AR all images; zero-query images recorded, excluded only from BH denominator",
                       unmatched_query_policy="no candidate/one-to-one shortage is a miss, never a zero-IoU tie hit",
                       comparison_note="two-stage AR+router pipeline, not native CTRL-O-style query-conditioned extraction")
    (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False))
    print(json.dumps({key: summary[key] for key in ["num_images", "num_queries", "binding_hits_percent",
                                                   "binding_hits_iou50", "target_mIoU", "zero_iou_hits"]}, indent=2))
    print(f"Saved: {output / 'summary.json'}")


if __name__ == "__main__":
    main()
