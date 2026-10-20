"""Evaluate the public Slot-MLLM-7B checkpoint on prepared balanced GQA.

This evaluator deliberately separates image tokenization from language-model
generation.  GQA testdev contains many questions per image, so the 32 x 4 RVQ
slot IDs are cached once per unique image and then reused by four independent
generation workers.

The public ``KU-AGI/Slot-MLLM-7B-instruct`` checkpoint is the discrete-token
release (32 slots x 4 RVQ IDs), not the later continuous-slot ICML/v3 model.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

import torch
import torch.distributed as dist
from PIL import Image
from torch.utils.data import DataLoader, Dataset


def read_jsonl(path: str | Path) -> list[dict]:
    with Path(path).open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_json(path: str | Path, value: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def normalize_answer(value: str) -> str:
    value = value.lower().strip()
    value = re.sub(r"[^a-z0-9 ]", "", value)
    return " ".join(value.split())


def image_path(image_root: Path, image_id: str) -> Path:
    for suffix in (".jpg", ".png", ".jpeg", ".webp"):
        candidate = image_root / f"{image_id}{suffix}"
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"No image found for GQA imageId={image_id} in {image_root}")


class ImageDataset(Dataset):
    def __init__(self, image_ids: list[str], image_root: Path, transform) -> None:
        self.image_ids = image_ids
        self.image_root = image_root
        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(self, index: int) -> tuple[str, torch.Tensor]:
        image_id = self.image_ids[index]
        with Image.open(image_path(self.image_root, image_id)) as image:
            tensor = self.transform(image.convert("RGB"))
        return image_id, tensor


def cache_slots(args: argparse.Namespace) -> None:
    os.environ["SLOT_MLLM_ENCODER_ONLY"] = "1"
    external = Path(args.slot_mllm_repo).resolve()
    sys.path.insert(0, str(external))
    os.chdir(external)

    import hydra
    from omegaconf import OmegaConf
    from modeling_slot_qformer import SlotQFormerModel

    rows = read_jsonl(args.questions)
    image_ids = sorted({str(row["imageId"]) for row in rows})
    if args.max_images:
        image_ids = image_ids[: args.max_images]
    transform_cfg = OmegaConf.load(external / "configs/transform/slot_transform.yaml")
    transform = hydra.utils.instantiate(transform_cfg)
    dataset = ImageDataset(image_ids, Path(args.image_root), transform)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    device = torch.device("cuda", 0)
    start = time.monotonic()
    model = SlotQFormerModel.from_pretrained(args.visual_tokenizer).wrapper.to(device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    cached: dict[str, torch.Tensor] = {}
    with torch.inference_mode():
        for step, (batch_ids, images) in enumerate(loader, start=1):
            images = images.to(device=device, dtype=torch.float16, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.float16):
                codes = model.forward_stage_1(images)
            codes = codes.to(device="cpu", dtype=torch.int16)
            for image_id, code in zip(batch_ids, codes):
                cached[str(image_id)] = code.contiguous()
            print(
                f"slot-cache batch={step}/{len(loader)} images={len(cached)}/{len(dataset)}",
                flush=True,
            )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "protocol": "Slot-MLLM public discrete release: 32 slots x 4 RVQ codes",
            "questions": str(Path(args.questions).resolve()),
            "visual_tokenizer": str(Path(args.visual_tokenizer).resolve()),
            "slot_codes": cached,
        },
        output,
    )
    write_json(
        output.with_suffix(".summary.json"),
        {
            "status": "complete",
            "num_images": len(cached),
            "slot_shape": list(next(iter(cached.values())).shape),
            "elapsed_seconds": time.monotonic() - start,
            "output": str(output.resolve()),
        },
    )


def prepare_input_ids(tokenizer, row: dict, slot_codes: torch.Tensor, model_vocab: int) -> list[int]:
    text_vocab_size = int(tokenizer.vocab_size)
    image_vocab_size = 8192
    boi_token = text_vocab_size + image_vocab_size
    eoi_token = boi_token + 1
    if model_vocab <= eoi_token:
        raise ValueError(
            f"Checkpoint vocab ({model_vocab}) does not contain Slot-MLLM image tokens through {eoi_token}"
        )
    prefix = tokenizer("USER: ", add_special_tokens=True).input_ids
    question = str(row["question"])
    suffix = tokenizer(
        f"{question} Please provide an accurate answer consisting of only one word or phrase.\nASSISTANT:",
        add_special_tokens=False,
    ).input_ids
    image_ids = slot_codes.to(dtype=torch.long).reshape(-1).tolist()
    image_ids = [token + text_vocab_size for token in image_ids]
    return prefix + [boi_token] + image_ids + [eoi_token] + suffix


def make_batch(
    tokenizer,
    rows: list[dict],
    cache: dict[str, torch.Tensor],
    model_vocab: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    sequences = [
        prepare_input_ids(tokenizer, row, cache[str(row["imageId"])], model_vocab) for row in rows
    ]
    width = max(map(len, sequences))
    pad = tokenizer.pad_token_id
    if pad is None:
        pad = tokenizer.eos_token_id
    input_ids = torch.full((len(rows), width), int(pad), dtype=torch.long, device=device)
    attention = torch.zeros((len(rows), width), dtype=torch.long, device=device)
    # Decoder-only generation expects left padding so every final prompt token
    # occupies the last column.
    for index, sequence in enumerate(sequences):
        values = torch.tensor(sequence, dtype=torch.long, device=device)
        input_ids[index, -len(sequence) :] = values
        attention[index, -len(sequence) :] = 1
    return input_ids, attention


def evaluate(args: argparse.Namespace) -> None:
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size > 1:
        dist.init_process_group("nccl")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    all_rows = read_jsonl(args.questions)
    indexed_rows = [(index, row) for index, row in enumerate(all_rows) if index % world_size == rank]
    cache_bundle = torch.load(args.slot_cache, map_location="cpu", weights_only=False)
    slot_cache = cache_bundle["slot_codes"]
    missing = sorted({str(row["imageId"]) for _, row in indexed_rows} - set(slot_cache))
    if missing:
        raise FileNotFoundError(f"Slot cache is missing {len(missing)} images, first={missing[0]}")

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    ).to(device)
    model.eval()
    model.config.use_cache = True

    outputs: list[dict] = []
    start = time.monotonic()
    for offset in range(0, len(indexed_rows), args.batch_size):
        group = indexed_rows[offset : offset + args.batch_size]
        rows = [row for _, row in group]
        input_ids, attention = make_batch(
            tokenizer, rows, slot_cache, int(model.config.vocab_size), device
        )
        with torch.inference_mode():
            generated = model.generate(
                input_ids=input_ids,
                attention_mask=attention,
                do_sample=False,
                num_beams=args.num_beams,
                max_new_tokens=args.max_new_tokens,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                use_cache=True,
            )
        answers = tokenizer.batch_decode(generated[:, input_ids.shape[1] :], skip_special_tokens=True)
        for (index, row), answer in zip(group, answers):
            prediction = answer.strip()
            gold = str(row["answer"])
            outputs.append(
                {
                    "index": index,
                    "id": str(row["id"]),
                    "image_id": str(row["imageId"]),
                    "question": str(row["question"]),
                    "answer": gold,
                    "prediction": prediction,
                    "strict_exact_match": prediction == gold,
                    "normalized_exact_match": normalize_answer(prediction) == normalize_answer(gold),
                }
            )
        print(
            f"rank={rank} generated={len(outputs)}/{len(indexed_rows)} elapsed={time.monotonic() - start:.1f}s",
            flush=True,
        )

    result_dir = Path(args.result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)
    shard = result_dir / f"predictions_rank{rank:02d}.jsonl"
    shard.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in outputs))
    if world_size > 1:
        dist.barrier()
    if rank == 0:
        merged = []
        for shard_rank in range(world_size):
            merged.extend(read_jsonl(result_dir / f"predictions_rank{shard_rank:02d}.jsonl"))
        merged.sort(key=lambda row: int(row["index"]))
        if len(merged) != len(all_rows):
            raise RuntimeError(f"Merged {len(merged)} predictions for {len(all_rows)} questions")
        (result_dir / "predictions.jsonl").write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in merged)
        )
        strict = sum(bool(row["strict_exact_match"]) for row in merged)
        normalized = sum(bool(row["normalized_exact_match"]) for row in merged)
        write_json(
            result_dir / "summary.json",
            {
                "status": "complete",
                "model": str(Path(args.model).resolve()),
                "release": "Slot-MLLM public v1/discrete checkpoint",
                "paper_score_association": "old public release reports GQA 58.1; latest v3 paper reports 57.3 for a different continuous-slot model",
                "split": "testdev",
                "balanced_only": all(bool(row.get("isBalanced", False)) for row in all_rows),
                "num_questions": len(merged),
                "num_unique_images": len({row["image_id"] for row in merged}),
                "strict_exact_match": strict / len(merged),
                "strict_correct": strict,
                "normalized_exact_match": normalized / len(merged),
                "normalized_correct": normalized,
                "normalization_added_matches": normalized - strict,
                "generation": {
                    "num_beams": args.num_beams,
                    "max_new_tokens": args.max_new_tokens,
                    "prompt": "official public Slot-MLLM VQA prompt",
                },
                "world_size": world_size,
                "elapsed_seconds_rank0": time.monotonic() - start,
            },
        )
    if world_size > 1:
        dist.barrier()
        dist.destroy_process_group()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    cache_parser = subparsers.add_parser("cache")
    cache_parser.add_argument("--questions", required=True)
    cache_parser.add_argument("--image_root", required=True)
    cache_parser.add_argument("--visual_tokenizer", required=True)
    cache_parser.add_argument("--slot_mllm_repo", required=True)
    cache_parser.add_argument("--output", required=True)
    cache_parser.add_argument("--batch_size", type=int, default=8)
    cache_parser.add_argument("--num_workers", type=int, default=4)
    cache_parser.add_argument("--max_images", type=int, default=0)

    eval_parser = subparsers.add_parser("eval")
    eval_parser.add_argument("--questions", required=True)
    eval_parser.add_argument("--slot_cache", required=True)
    eval_parser.add_argument("--model", required=True)
    eval_parser.add_argument("--result_dir", required=True)
    eval_parser.add_argument("--batch_size", type=int, default=32)
    eval_parser.add_argument("--max_new_tokens", type=int, default=12)
    eval_parser.add_argument("--num_beams", type=int, default=1)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "cache":
        cache_slots(args)
    else:
        evaluate(args)


if __name__ == "__main__":
    main()
