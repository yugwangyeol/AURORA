"""Train and evaluate a GQA answer model over cached DeOR object tokens.

The image is never reopened by this program.  Each GQA question receives one
frozen semantic vector per object from ``gqa_object_cache.py``.  A trainable
projector maps those vectors into the answer LLM embedding space, while LoRA is
the only trainable part of the LLM.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import time
from collections import OrderedDict
from pathlib import Path

import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Dataset, DistributedSampler


def cache_path(cache_root: Path, split: str, image_id: str) -> Path:
    fanout = hashlib.sha1(image_id.encode()).hexdigest()[:2]
    return cache_root / split / fanout / f"{image_id}.pt"


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    tmp.replace(path)


class GQACachedObjectDataset(Dataset):
    """Byte-offset JSONL reader with a small per-worker object-cache LRU."""

    def __init__(
        self,
        questions: str | Path,
        cache_root: str | Path,
        split: str,
        max_samples: int = 0,
        object_cache_size: int = 8,
    ) -> None:
        self.questions = Path(questions)
        self.cache_root = Path(cache_root)
        self.split = split
        self.object_cache_size = object_cache_size
        self.offsets: list[int] = []
        offset = 0
        with self.questions.open("rb") as stream:
            for line in stream:
                if line.strip():
                    self.offsets.append(offset)
                    if max_samples and len(self.offsets) >= max_samples:
                        break
                offset += len(line)
        if not self.offsets:
            raise ValueError(f"No questions found in {self.questions}")
        self._stream = None
        self._objects: OrderedDict[str, tuple[torch.Tensor, torch.Tensor]] = OrderedDict()

    def __len__(self) -> int:
        return len(self.offsets)

    def _record(self, index: int) -> dict:
        if self._stream is None:
            self._stream = self.questions.open("rb")
        self._stream.seek(self.offsets[index])
        return json.loads(self._stream.readline())

    def _object_tokens(self, image_id: str) -> tuple[torch.Tensor, torch.Tensor]:
        if image_id in self._objects:
            value = self._objects.pop(image_id)
            self._objects[image_id] = value
            return value
        path = cache_path(self.cache_root, self.split, image_id)
        if not path.is_file():
            raise FileNotFoundError(f"Missing DeOR object cache: {path}")
        bundle = torch.load(path, map_location="cpu", weights_only=False)
        semantic = bundle.get("semantic")
        visual = bundle.get("visual_memory")
        if not isinstance(semantic, torch.Tensor) or semantic.ndim != 2 or semantic.shape[1] != 1536:
            raise ValueError(f"Invalid semantic tensor in {path}: {getattr(semantic, 'shape', None)}")
        if (
            not isinstance(visual, torch.Tensor)
            or visual.ndim != 3
            or visual.shape[0] != semantic.shape[0]
            or visual.shape[1:] != (8, 1536)
        ):
            raise ValueError(f"Invalid visual-memory tensor in {path}: {getattr(visual, 'shape', None)}")
        value = (semantic.float().contiguous(), visual.float().contiguous())
        self._objects[image_id] = value
        while len(self._objects) > self.object_cache_size:
            self._objects.popitem(last=False)
        return value

    def __getitem__(self, index: int) -> dict:
        record = self._record(index)
        image_id = str(record["imageId"])
        answer = record.get("answer")
        if answer is None:
            raise ValueError(f"Question {record.get('id')} has no answer")
        semantic, visual_memory = self._object_tokens(image_id)
        return {
            "id": str(record["id"]),
            "image_id": image_id,
            "question": str(record["question"]),
            "answer": str(answer),
            "semantic": semantic,
            "visual_memory": visual_memory,
            "empty_discovery": semantic.shape[0] == 0,
        }

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_stream"] = None
        state["_objects"] = OrderedDict()
        return state


class DeORGQA(nn.Module):
    PREFIX = "USER: The image is represented by the following object tokens:\n"
    QUESTION = "\nAnswer the question with a short answer.\nQuestion: {question}\nASSISTANT:"

    REPRESENTATIONS = (
        "semantic",
        "visual_all",
        "semantic_visual_pooled",
        "semantic_visual_all",
    )

    def __init__(
        self,
        llm: nn.Module,
        tokenizer,
        projector_hidden: int = 2048,
        representation: str = "semantic",
    ) -> None:
        super().__init__()
        if representation not in self.REPRESENTATIONS:
            raise ValueError(f"Unknown representation: {representation}")
        self.llm = llm
        self.tokenizer = tokenizer
        self.representation = representation
        hidden = int(llm.config.hidden_size)
        needs_semantic = representation != "visual_all"
        needs_visual = representation != "semantic"

        def make_projector():
            return nn.Sequential(
                nn.LayerNorm(1536),
                nn.Linear(1536, projector_hidden),
                nn.GELU(),
                nn.Linear(projector_hidden, hidden),
            )

        self.semantic_projector = make_projector() if needs_semantic else None
        self.visual_projector = make_projector() if needs_visual else None
        self.semantic_type = nn.Parameter(torch.zeros(hidden)) if needs_semantic else None
        self.visual_type = nn.Parameter(torch.zeros(hidden)) if needs_visual else None
        for parameter in (self.semantic_type, self.visual_type):
            if parameter is not None:
                nn.init.normal_(parameter, std=0.02)

    @property
    def embedding(self):
        return self.llm.get_input_embeddings()

    def _token_ids(self, text: str) -> torch.Tensor:
        ids = self.tokenizer(text, add_special_tokens=False).input_ids
        return torch.tensor(ids, dtype=torch.long, device=self.embedding.weight.device)

    def _object_representation(self, sample: dict) -> torch.Tensor:
        device = self.embedding.weight.device
        dtype = self.embedding.weight.dtype
        semantic = sample["semantic"].to(device=device, dtype=torch.float32)
        visual = sample["visual_memory"].to(device=device, dtype=torch.float32)
        # AR discovery can legitimately return no objects.  Retain those GQA
        # questions and use one zero-valued pseudo-object rather than dropping
        # the sample.  Passing it through every active projector also keeps the
        # DDP parameter-use pattern identical on all ranks.
        if semantic.shape[0] == 0:
            semantic = torch.zeros((1, 1536), device=device, dtype=torch.float32)
            visual = torch.zeros((1, 8, 1536), device=device, dtype=torch.float32)
        semantic_tokens = None
        visual_tokens = None
        if self.semantic_projector is not None:
            semantic_tokens = self.semantic_projector(semantic) + self.semantic_type
        if self.visual_projector is not None:
            if self.representation == "semantic_visual_pooled":
                visual = visual.mean(dim=1, keepdim=True)
            visual_tokens = self.visual_projector(visual) + self.visual_type
        if self.representation == "semantic":
            result = semantic_tokens
        elif self.representation == "visual_all":
            result = visual_tokens.flatten(0, 1)
        elif self.representation in ("semantic_visual_pooled", "semantic_visual_all"):
            result = torch.cat([semantic_tokens[:, None], visual_tokens], dim=1).flatten(0, 1)
        else:  # guarded by __init__
            raise AssertionError(self.representation)
        return result.to(dtype=dtype)

    def _one(self, sample: dict, with_answer: bool) -> tuple[torch.Tensor, torch.Tensor]:
        device = self.embedding.weight.device
        dtype = self.embedding.weight.dtype
        bos = [] if self.tokenizer.bos_token_id is None else [self.tokenizer.bos_token_id]
        prefix_ids = torch.tensor(
            bos + self.tokenizer(self.PREFIX, add_special_tokens=False).input_ids,
            dtype=torch.long,
            device=device,
        )
        question_ids = self._token_ids(self.QUESTION.format(question=sample["question"]))
        prefix = self.embedding(prefix_ids)
        objects = self._object_representation(sample)
        question = self.embedding(question_ids)
        pieces = [prefix, objects, question]
        labels = [torch.full((sum(x.shape[0] for x in pieces),), -100, device=device, dtype=torch.long)]
        if with_answer:
            answer_ids = self.tokenizer(
                " " + sample["answer"], add_special_tokens=False
            ).input_ids
            if self.tokenizer.eos_token_id is not None:
                answer_ids.append(self.tokenizer.eos_token_id)
            answer_ids = torch.tensor(answer_ids, dtype=torch.long, device=device)
            pieces.append(self.embedding(answer_ids))
            labels.append(answer_ids)
        return torch.cat(pieces, dim=0), torch.cat(labels, dim=0)

    def build_batch(self, samples: list[dict], with_answer: bool = True):
        rows = [self._one(sample, with_answer) for sample in samples]
        max_length = max(emb.shape[0] for emb, _ in rows)
        hidden = rows[0][0].shape[1]
        dtype = rows[0][0].dtype
        device = rows[0][0].device
        embeddings = torch.zeros(len(rows), max_length, hidden, dtype=dtype, device=device)
        attention = torch.zeros(len(rows), max_length, dtype=torch.long, device=device)
        labels = torch.full((len(rows), max_length), -100, dtype=torch.long, device=device)
        lengths = []
        for index, (emb, lab) in enumerate(rows):
            length = emb.shape[0]
            embeddings[index, :length] = emb
            attention[index, :length] = 1
            labels[index, :length] = lab
            lengths.append(length)
        return embeddings, attention, labels, lengths

    def forward(self, samples: list[dict]):
        embeddings, attention, labels, _ = self.build_batch(samples, with_answer=True)
        return self.llm(inputs_embeds=embeddings, attention_mask=attention, labels=labels)

    @torch.inference_mode()
    def greedy_answer(self, sample: dict, max_new_tokens: int = 12) -> str:
        embeddings, attention, _, lengths = self.build_batch([sample], with_answer=False)
        output = self.llm(inputs_embeds=embeddings, attention_mask=attention, use_cache=True)
        next_token = output.logits[:, lengths[0] - 1].argmax(dim=-1)
        generated = []
        past = output.past_key_values
        for _ in range(max_new_tokens):
            token = int(next_token.item())
            if token == self.tokenizer.eos_token_id:
                break
            generated.append(token)
            attention = torch.cat(
                [attention, torch.ones((1, 1), dtype=attention.dtype, device=attention.device)], dim=1
            )
            output = self.llm(
                input_ids=next_token[:, None],
                attention_mask=attention,
                past_key_values=past,
                use_cache=True,
            )
            past = output.past_key_values
            next_token = output.logits[:, -1].argmax(dim=-1)
        return self.tokenizer.decode(generated, skip_special_tokens=True).strip()


def normalize_answer(value: str) -> str:
    value = value.lower().strip()
    value = re.sub(r"[^a-z0-9 ]", "", value)
    return " ".join(value.split())


def distributed_setup() -> tuple[int, int, int]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size > 1:
        dist.init_process_group("nccl")
    torch.cuda.set_device(local_rank)
    return rank, local_rank, world_size


def load_train_model(args, device: torch.device):
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=False)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    llm = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    llm.config.use_cache = False
    llm = get_peft_model(
        llm,
        LoraConfig(
            task_type="CAUSAL_LM",
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        ),
    )
    model = DeORGQA(llm, tokenizer, args.projector_hidden, args.representation).to(device)
    return model, tokenizer


def save_model(model: DeORGQA, tokenizer, output: Path, args) -> None:
    output.mkdir(parents=True, exist_ok=True)
    model.llm.save_pretrained(output / "adapter")
    tokenizer.save_pretrained(output / "tokenizer")
    torch.save(
        {
            "semantic_projector": (
                model.semantic_projector.state_dict() if model.semantic_projector is not None else None
            ),
            "visual_projector": (
                model.visual_projector.state_dict() if model.visual_projector is not None else None
            ),
            "semantic_type": (
                model.semantic_type.detach().cpu() if model.semantic_type is not None else None
            ),
            "visual_type": model.visual_type.detach().cpu() if model.visual_type is not None else None,
            "projector_hidden": args.projector_hidden,
            "semantic_dim": 1536,
            "representation": args.representation,
        },
        output / "deor_projector.pt",
    )
    write_json(
        output / "run_config.json",
        {
            "model_path": str(Path(args.model_path).resolve()),
            "cache_root": str(Path(args.cache_root).resolve()),
            "representation": "frozen DeOR object tokens projected into the answer LLM",
            "representation_mode": args.representation,
            "visual_memory_used": args.representation != "semantic",
            "register_tokens_used": False,
            "projector_hidden": args.projector_hidden,
            "lora_rank": args.lora_rank,
        },
    )


def train(args) -> None:
    rank, local_rank, world_size = distributed_setup()
    seed = args.seed + rank
    random.seed(seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda", local_rank)
    dataset = GQACachedObjectDataset(
        args.questions, args.cache_root, args.split, args.max_samples, args.object_cache_size
    )
    sampler = DistributedSampler(dataset, world_size, rank, shuffle=True, seed=args.seed) if world_size > 1 else None
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        shuffle=sampler is None,
        num_workers=args.num_workers,
        collate_fn=lambda rows: rows,
        pin_memory=True,
    )
    model, tokenizer = load_train_model(args, device)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    trainable_count = sum(parameter.numel() for parameter in trainable)
    optimizer = torch.optim.AdamW(trainable, lr=args.learning_rate, weight_decay=args.weight_decay)
    from transformers import get_cosine_schedule_with_warmup

    optimizer_steps = (args.max_steps + args.gradient_accumulation_steps - 1) // args.gradient_accumulation_steps
    warmup_steps = int(optimizer_steps * args.warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, optimizer_steps)
    wrapped = DistributedDataParallel(model, device_ids=[local_rank], find_unused_parameters=False) if world_size > 1 else model
    wrapped.train()
    start = time.monotonic()
    losses = []
    step = 0
    optimizer.zero_grad(set_to_none=True)
    while step < args.max_steps:
        if sampler is not None:
            sampler.set_epoch(step)
        for samples in loader:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = wrapped(samples).loss / args.gradient_accumulation_steps
            loss.backward()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(trainable, args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            reduced = loss.detach() * args.gradient_accumulation_steps
            if world_size > 1:
                dist.all_reduce(reduced, op=dist.ReduceOp.AVG)
            losses.append(float(reduced))
            step += 1
            if rank == 0:
                print(f"step={step}/{args.max_steps} loss={losses[-1]:.6f}", flush=True)
            if step >= args.max_steps:
                break
    if step % args.gradient_accumulation_steps:
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
    if world_size > 1:
        dist.barrier()
    if rank == 0:
        base = wrapped.module if isinstance(wrapped, DistributedDataParallel) else wrapped
        elapsed = time.monotonic() - start
        save_model(base, tokenizer, Path(args.output_dir), args)
        write_json(
            Path(args.output_dir) / "train_summary.json",
            {
                "status": "complete",
                "world_size": world_size,
                "representation": args.representation,
                "steps": step,
                "losses": losses,
                "initial_loss": losses[0],
                "final_loss": losses[-1],
                "mean_last_10_loss": sum(losses[-10:]) / min(10, len(losses)),
                "elapsed_seconds": elapsed,
                "steps_per_second": step / elapsed,
                "trainable_parameters": trainable_count,
                "optimizer_steps": optimizer_steps,
                "warmup_steps": warmup_steps,
                "scheduler": "cosine",
                "learning_rate": args.learning_rate,
                "peak_gpu_memory_gib": torch.cuda.max_memory_allocated(device) / 2**30,
                "num_questions_visible": len(dataset),
            },
        )
    if world_size > 1:
        dist.barrier()
        dist.destroy_process_group()


def load_saved_model(args, device: torch.device):
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    output = Path(args.output_dir)
    config = json.loads((output / "run_config.json").read_text())
    tokenizer = AutoTokenizer.from_pretrained(output / "tokenizer", use_fast=False)
    base = AutoModelForCausalLM.from_pretrained(
        config["model_path"], torch_dtype=torch.bfloat16, low_cpu_mem_usage=True
    )
    llm = PeftModel.from_pretrained(base, output / "adapter", is_trainable=False)
    model = DeORGQA(
        llm,
        tokenizer,
        int(config["projector_hidden"]),
        config.get("representation_mode", "semantic"),
    )
    state = torch.load(output / "deor_projector.pt", map_location="cpu", weights_only=False)
    if model.semantic_projector is not None:
        model.semantic_projector.load_state_dict(state["semantic_projector"])
        model.semantic_type.data.copy_(state["semantic_type"])
    if model.visual_projector is not None:
        model.visual_projector.load_state_dict(state["visual_projector"])
        model.visual_type.data.copy_(state["visual_type"])
    model.to(device).eval()
    model.llm.config.use_cache = True
    return model


def evaluate(args) -> None:
    torch.cuda.set_device(0)
    device = torch.device("cuda", 0)
    model = load_saved_model(args, device)
    dataset = GQACachedObjectDataset(
        args.questions, args.cache_root, args.split, args.max_samples, args.object_cache_size
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=lambda rows: rows,
    )
    config = json.loads((Path(args.output_dir) / "run_config.json").read_text())
    model.llm.config.use_cache = False
    loss_sum = 0.0
    loss_batches = 0
    with torch.inference_mode():
        for samples in loader:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = model(samples).loss
            loss_sum += float(loss)
            loss_batches += 1
    model.llm.config.use_cache = True
    predictions = []
    correct = 0
    start = time.monotonic()
    generation_count = min(len(dataset), args.generation_samples)
    visual_token_multiplier = {
        "semantic": 1,
        "visual_all": 8,
        "semantic_visual_pooled": 2,
        "semantic_visual_all": 9,
    }[config.get("representation_mode", "semantic")]
    token_counts = []
    for index in range(generation_count):
        sample = dataset[index]
        prediction = model.greedy_answer(sample, args.max_new_tokens)
        hit = normalize_answer(prediction) == normalize_answer(sample["answer"])
        correct += int(hit)
        predictions.append(
            {
                "id": sample["id"],
                "image_id": sample["image_id"],
                "question": sample["question"],
                "answer": sample["answer"],
                "prediction": prediction,
                "exact_match": hit,
                "num_objects": int(sample["semantic"].shape[0]),
                "num_visual_input_tokens": max(1, int(sample["semantic"].shape[0]))
                * visual_token_multiplier,
                "empty_discovery": bool(sample["empty_discovery"]),
            }
        )
        token_counts.append(max(1, int(sample["semantic"].shape[0])) * visual_token_multiplier)
        print(f"generate={index + 1}/{generation_count} prediction={prediction!r}", flush=True)
    result_dir = Path(args.result_dir) if args.result_dir else Path(args.output_dir)
    result_dir.mkdir(parents=True, exist_ok=True)
    path = result_dir / "predictions.jsonl"
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in predictions))
    write_json(
        result_dir / "eval_summary.json",
        {
            "status": "complete",
            "representation": config.get("representation_mode", "semantic"),
            "num_loss_samples": len(dataset),
            "validation_loss": loss_sum / loss_batches,
            "num_generation_samples": generation_count,
            "exact_match": correct / generation_count,
            "mean_visual_input_tokens": sum(token_counts) / len(token_counts),
            "elapsed_seconds": time.monotonic() - start,
            "checkpoint_reload_verified": True,
        },
    )


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser()
    sub = root.add_subparsers(dest="command", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--questions", required=True)
    common.add_argument("--cache_root", required=True)
    common.add_argument("--split", required=True, choices=("train", "val", "testdev"))
    common.add_argument("--output_dir", required=True)
    common.add_argument("--max_samples", type=int, default=0)
    common.add_argument("--object_cache_size", type=int, default=8)
    train_parser = sub.add_parser("train", parents=[common])
    train_parser.add_argument("--model_path", required=True)
    train_parser.add_argument("--batch_size", type=int, default=1)
    train_parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    train_parser.add_argument("--max_steps", type=int, default=2)
    train_parser.add_argument("--learning_rate", type=float, default=2e-4)
    train_parser.add_argument("--warmup_ratio", type=float, default=0.03)
    train_parser.add_argument("--weight_decay", type=float, default=0.0)
    train_parser.add_argument("--max_grad_norm", type=float, default=1.0)
    train_parser.add_argument("--lora_rank", type=int, default=16)
    train_parser.add_argument("--lora_alpha", type=int, default=32)
    train_parser.add_argument("--lora_dropout", type=float, default=0.05)
    train_parser.add_argument("--projector_hidden", type=int, default=2048)
    train_parser.add_argument(
        "--representation", choices=DeORGQA.REPRESENTATIONS, default="semantic"
    )
    train_parser.add_argument("--num_workers", type=int, default=0)
    train_parser.add_argument("--seed", type=int, default=42)
    eval_parser = sub.add_parser("eval", parents=[common])
    eval_parser.add_argument("--max_new_tokens", type=int, default=12)
    eval_parser.add_argument("--batch_size", type=int, default=8)
    eval_parser.add_argument("--generation_samples", type=int, default=32)
    eval_parser.add_argument("--result_dir")
    return root


def main() -> None:
    args = parser().parse_args()
    if args.command == "train":
        train(args)
    else:
        evaluate(args)


if __name__ == "__main__":
    main()
