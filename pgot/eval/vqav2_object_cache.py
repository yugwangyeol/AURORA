"""Prepare VQAv2 manifests and cache frozen DeOR object representations.

Question and answer annotations are written to a manifest that is independent
from the image-only manifest consumed by the DeOR extractor.  This separation
prevents the VQA question or answer from leaking into object discovery.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from PIL import Image

from pgot.eval import gqa_object_cache as shared


FORMAT_VERSION = "deor-vqav2-object-cache-v1"
SPLITS = ("train", "val")
EXPECTED_IMAGE_COUNTS = {"train": 82783, "val": 40504}
EXPECTED_QUESTION_COUNTS = {"train": 443757, "val": 214354}


def _source_paths(vqav2_root: Path, split: str) -> tuple[Path, Path, Path]:
    coco_split = f"{split}2014"
    questions = vqav2_root / f"v2_OpenEnded_mscoco_{coco_split}_questions.json"
    annotations = vqav2_root / f"v2_mscoco_{coco_split}_annotations.json"
    images = vqav2_root / coco_split
    for path in (questions, annotations, images):
        if not path.exists():
            raise FileNotFoundError(path)
    return questions, annotations, images


def _load_aligned_records(question_path: Path, annotation_path: Path) -> list[tuple[dict, dict]]:
    with question_path.open() as stream:
        questions = json.load(stream).get("questions")
    with annotation_path.open() as stream:
        annotations = json.load(stream).get("annotations")
    if not isinstance(questions, list) or not isinstance(annotations, list):
        raise ValueError("Invalid VQAv2 question/annotation JSON structure")
    if len(questions) != len(annotations):
        raise ValueError(
            f"Question/annotation length mismatch: {len(questions)} != {len(annotations)}"
        )
    aligned = []
    for question, annotation in zip(questions, annotations):
        question_id = int(question["question_id"])
        if question_id != int(annotation["question_id"]):
            raise ValueError(f"Question ordering mismatch at question_id={question_id}")
        if int(question["image_id"]) != int(annotation["image_id"]):
            raise ValueError(f"Image mismatch at question_id={question_id}")
        aligned.append((question, annotation))
    return aligned


def _image_name(split: str, image_id: int) -> str:
    return f"COCO_{split}2014_{image_id:012d}"


def prepare_split(args: argparse.Namespace, split: str) -> dict:
    vqav2_root = Path(args.vqav2_root).resolve()
    prepared_root = Path(args.prepared_root).resolve()
    question_path, annotation_path, image_root = _source_paths(vqav2_root, split)
    records = _load_aligned_records(question_path, annotation_path)

    all_image_ids = sorted({int(question["image_id"]) for question, _ in records})
    if args.max_images:
        all_image_ids = all_image_ids[: args.max_images]
    selected_image_ids = set(all_image_ids)
    if not selected_image_ids:
        raise ValueError(f"No images selected for VQAv2 {split}")

    manifest_root = prepared_root / "manifests"
    manifest_root.mkdir(parents=True, exist_ok=True)
    images_path = manifest_root / f"images_{split}.jsonl"
    questions_path = manifest_root / f"questions_{split}.jsonl"
    images_tmp = images_path.with_name(images_path.name + ".tmp")
    questions_tmp = questions_path.with_name(questions_path.name + ".tmp")

    with images_tmp.open("w") as output:
        for numeric_image_id in all_image_ids:
            image_id = _image_name(split, numeric_image_id)
            image_path = image_root / f"{image_id}.jpg"
            if not image_path.is_file():
                raise FileNotFoundError(image_path)
            with Image.open(image_path) as image:
                image.verify()
                width, height = image.size
            record = {
                "image_id": image_id,
                "coco_image_id": numeric_image_id,
                "image_path": str(image_path),
                "image_sha256": shared.sha256_file(image_path),
                "width": int(width),
                "height": int(height),
                "split": split,
            }
            output.write(json.dumps(record, ensure_ascii=False) + "\n")
    images_tmp.replace(images_path)

    question_count = 0
    with questions_tmp.open("w") as output:
        for question, annotation in records:
            numeric_image_id = int(question["image_id"])
            if numeric_image_id not in selected_image_ids:
                continue
            answers = [str(item["answer"]) for item in annotation.get("answers", [])]
            answer_counts = Counter(answers)
            record = {
                "question_id": int(question["question_id"]),
                "image_id": _image_name(split, numeric_image_id),
                "coco_image_id": numeric_image_id,
                "question": str(question["question"]),
                "multiple_choice_answer": str(annotation["multiple_choice_answer"]),
                "answers": answers,
                "answer_counts": dict(answer_counts),
                "question_type": str(annotation.get("question_type", "")),
                "answer_type": str(annotation.get("answer_type", "")),
                "split": split,
            }
            output.write(json.dumps(record, ensure_ascii=False) + "\n")
            question_count += 1
    questions_tmp.replace(questions_path)

    expected_images = EXPECTED_IMAGE_COUNTS[split] if not args.max_images else None
    expected_questions = EXPECTED_QUESTION_COUNTS[split] if not args.max_images else None
    if expected_images is not None and len(all_image_ids) != expected_images:
        raise ValueError(
            f"Unexpected {split} image count: {len(all_image_ids)} != {expected_images}"
        )
    if expected_questions is not None and question_count != expected_questions:
        raise ValueError(
            f"Unexpected {split} question count: {question_count} != {expected_questions}"
        )

    summary = {
        "format_version": FORMAT_VERSION,
        "split": split,
        "num_images": len(all_image_ids),
        "num_questions": question_count,
        "images_manifest": str(images_path),
        "questions_manifest": str(questions_path),
        "extractor_input_fields": [
            "image_id",
            "coco_image_id",
            "image_path",
            "image_sha256",
            "width",
            "height",
            "split",
        ],
        "extractor_receives_questions_or_answers": False,
        "source_questions": str(question_path),
        "source_annotations": str(annotation_path),
        "source_images": str(image_root),
    }
    shared.write_json(manifest_root / f"summary_{split}.json", summary)
    return summary


def _add_extract_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--images_manifest", required=True)
    parser.add_argument("--cache_root", required=True)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--dtype", choices=("fp32", "bf16"), default="fp32")
    parser.add_argument("--cache_dtype", choices=("float16", "bfloat16"), default="float16")
    parser.add_argument("--eval_size", type=int, default=512)
    parser.add_argument("--max_objects", type=int, default=50)
    parser.add_argument("--max_caption_tokens", type=int, default=4096)
    parser.add_argument("--ar_max_new_tokens", type=int, default=2048)
    parser.add_argument("--diffusion_inference_steps", type=int, default=10)
    parser.add_argument("--cpu_threads", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--shard_index", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_images", type=int, default=0)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="stage", required=True)

    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--vqav2_root", required=True)
    prepare.add_argument("--prepared_root", required=True)
    prepare.add_argument("--splits", nargs="+", choices=SPLITS, default=list(SPLITS))
    prepare.add_argument("--max_images", type=int, default=0)

    extract_parser = subparsers.add_parser("extract")
    _add_extract_arguments(extract_parser)

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
        shared.write_json(Path(args.prepared_root) / "manifests" / "summary.json", summaries)
        print(json.dumps(summaries, indent=2))
        return
    args.format_version = FORMAT_VERSION
    args.extractor_code_path = str(Path(__file__).resolve())
    if args.stage == "extract":
        if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
            raise ValueError("Require num_shards >= 1 and 0 <= shard_index < num_shards")
        if args.batch_size < 1:
            raise ValueError("Require batch_size >= 1")
        shared.extract(args)
    else:
        shared.verify(args)


if __name__ == "__main__":
    main()
