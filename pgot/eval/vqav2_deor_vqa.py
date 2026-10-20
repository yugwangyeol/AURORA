"""CTRL-O-style VQAv2 probing over frozen DeOR object caches.

The DeOR cache was produced without seeing questions or answers.  This module
first aligns spaCy noun chunks to DeOR's generated object descriptions with a
frozen CLIP text encoder.  It then compares two otherwise matched VQA models:

* ``no_coupling``: selected object tokens enter only the output transformer.
* ``coupling``: each selected object token is inserted after its noun phrase
  before the T5 encoder.

Both modes use one fused semantic + mean-pooled visual-memory token per noun
chunk.  The answer head is a top-k VQAv2 classifier trained with soft targets.
"""

from __future__ import annotations

import argparse
import heapq
import hashlib
import json
import math
import os
import random
import re
import time
from collections import Counter, OrderedDict
from pathlib import Path
from typing import Iterable

import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Dataset


FORMAT_VERSION = "deor-vqav2-ctrl-o-style-v1"


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def cache_path(cache_root: Path, split: str, image_id: str) -> Path:
    fanout = hashlib.sha1(image_id.encode()).hexdigest()[:2]
    return cache_root / split / fanout / f"{image_id}.pt"


_CONTRACTIONS = {
    "aint": "ain't", "arent": "aren't", "cant": "can't", "couldve": "could've",
    "couldnt": "couldn't", "didnt": "didn't", "doesnt": "doesn't", "dont": "don't",
    "hadnt": "hadn't", "hasnt": "hasn't", "havent": "haven't", "hed": "he'd",
    "hes": "he's", "howd": "how'd", "howll": "how'll", "hows": "how's",
    "Idve": "I'd've", "Im": "I'm", "Ive": "I've", "isnt": "isn't",
    "itd": "it'd", "itll": "it'll", "lets": "let's", "mightnt": "mightn't",
    "mustnt": "mustn't", "shant": "shan't", "shed": "she'd", "shell": "she'll",
    "shes": "she's", "shouldnt": "shouldn't", "somebodyd": "somebody'd",
    "somebodys": "somebody's", "someoned": "someone'd", "someones": "someone's",
    "thatll": "that'll", "thats": "that's", "thered": "there'd", "therere": "there're",
    "theres": "there's", "theyd": "they'd", "theyll": "they'll", "theyre": "they're",
    "theyve": "they've", "wasnt": "wasn't", "wed": "we'd", "well": "we'll",
    "were": "we're", "werent": "weren't", "weve": "we've", "whatll": "what'll",
    "whatre": "what're", "whats": "what's", "whatve": "what've", "whens": "when's",
    "whered": "where'd", "wheres": "where's", "whereve": "where've",
    "whod": "who'd", "wholl": "who'll", "whos": "who's", "whove": "who've",
    "whyll": "why'll", "whys": "why's", "wont": "won't", "wouldnt": "wouldn't",
    "youd": "you'd", "youll": "you'll", "youre": "you're", "youve": "you've",
}
_NUMBER_WORDS = {
    "none": "0", "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
    "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10",
}
_ARTICLES = {"a", "an", "the"}


def normalize_answer(answer: str) -> str:
    """VQAv2-compatible answer normalization for vocabulary and scoring."""
    value = answer.lower().replace("\n", " ").replace("\t", " ").strip()
    value = re.sub(r"(?<!\d)[\.,](?!\d)", " ", value)
    value = re.sub(r"[;\/\[\]\"{}()=+\\_<>@`?!]", " ", value)
    words = []
    for word in value.split():
        word = _NUMBER_WORDS.get(word, word)
        if word in _ARTICLES:
            continue
        words.append(_CONTRACTIONS.get(word, word))
    return " ".join(words)


def iter_jsonl(path: Path, max_samples: int = 0) -> Iterable[dict]:
    count = 0
    with path.open() as stream:
        for line in stream:
            if not line.strip():
                continue
            yield json.loads(line)
            count += 1
            if max_samples and count >= max_samples:
                break


def build_answer_vocab(train_questions: Path, output: Path, size: int) -> list[str]:
    counts: Counter[str] = Counter()
    num_questions = 0
    for record in iter_jsonl(train_questions):
        answer = normalize_answer(str(record.get("multiple_choice_answer", "")))
        if answer:
            counts[answer] += 1
        num_questions += 1
    answers = [answer for answer, _ in counts.most_common(size)]
    if len(answers) != size:
        raise ValueError(f"Requested {size} answers but found only {len(answers)}")
    write_json(
        output,
        {
            "format_version": FORMAT_VERSION,
            "source": str(train_questions.resolve()),
            "num_train_questions": num_questions,
            "size": size,
            "answers": answers,
            "counts": {answer: counts[answer] for answer in answers},
        },
    )
    return answers


def object_texts(bundle: dict) -> list[str]:
    result = []
    for index, obj in enumerate(bundle.get("objects", [])):
        category = str(obj.get("category", "object")).strip()
        description = str(obj.get("description", "")).strip()
        result.append(f"{category}: {description}" if description else category or f"object {index}")
    return result


def noun_chunks(nlp, question: str, max_mentions: int) -> list[dict]:
    doc = nlp(question)
    chunks = []
    seen = set()
    for chunk in doc.noun_chunks:
        text = chunk.text.strip()
        key = (chunk.start_char, chunk.end_char)
        if not text or key in seen:
            continue
        seen.add(key)
        chunks.append({"text": text, "start": chunk.start_char, "end": chunk.end_char})
        if len(chunks) >= max_mentions:
            break
    if not chunks:
        chunks.append({"text": question.strip(), "start": 0, "end": len(question)})
    return chunks


class CLIPMatcher:
    def __init__(self, model_path: str, device: torch.device, batch_size: int = 64) -> None:
        from transformers import AutoTokenizer, CLIPTextModelWithProjection

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = CLIPTextModelWithProjection.from_pretrained(model_path).to(device).eval()
        self.device = device
        self.batch_size = batch_size

    @torch.inference_mode()
    def encode(self, texts: list[str]) -> torch.Tensor:
        rows = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            tokens = self.tokenizer(
                batch, padding=True, truncation=True, max_length=77, return_tensors="pt"
            ).to(self.device)
            with torch.autocast(self.device.type, dtype=torch.bfloat16, enabled=self.device.type == "cuda"):
                embedding = self.model(**tokens).text_embeds
            rows.append(nn.functional.normalize(embedding.float(), dim=-1).cpu())
        return torch.cat(rows) if rows else torch.empty(0, 512)


def prepare_alignments(args) -> None:
    import spacy

    questions = Path(args.questions)
    cache_root = Path(args.cache_root)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    summary_path = output.with_suffix(".summary.json")
    if output.is_file() and summary_path.is_file():
        prior = json.loads(summary_path.read_text())
        expected_max = int(args.max_samples)
        if (
            prior.get("status") == "complete"
            and prior.get("format_version") == FORMAT_VERSION
            and prior.get("split") == args.split
            and int(prior.get("num_shards", -1)) == args.num_shards
            and int(prior.get("shard_index", -1)) == args.shard_index
            and int(prior.get("max_samples", -1)) == expected_max
        ):
            print(f"Alignment shard already complete: {output}", flush=True)
            return
    nlp = spacy.load(args.spacy_model, disable=["ner", "textcat"])
    device = torch.device(args.device)
    matcher = CLIPMatcher(args.matcher_model, device, args.matcher_batch_size)
    object_embedding_cache: OrderedDict[str, tuple[list[str], torch.Tensor]] = OrderedDict()
    temporary = output.with_suffix(output.suffix + ".tmp")
    processed = 0
    missing_objects = 0
    start_time = time.monotonic()
    with temporary.open("w") as stream:
        for source_index, record in enumerate(iter_jsonl(questions, args.max_samples)):
            if source_index % args.num_shards != args.shard_index:
                continue
            image_id = str(record["image_id"])
            if image_id in object_embedding_cache:
                texts, embeddings = object_embedding_cache.pop(image_id)
                object_embedding_cache[image_id] = (texts, embeddings)
            else:
                path = cache_path(cache_root, args.split, image_id)
                if not path.is_file():
                    raise FileNotFoundError(f"Missing DeOR object cache: {path}")
                bundle = torch.load(path, map_location="cpu", weights_only=False)
                texts = object_texts(bundle)
                embeddings = matcher.encode(texts) if texts else torch.empty(0, 512)
                object_embedding_cache[image_id] = (texts, embeddings)
                while len(object_embedding_cache) > args.object_text_cache_size:
                    object_embedding_cache.popitem(last=False)
            chunks = noun_chunks(nlp, str(record["question"]), args.max_mentions)
            if texts:
                chunk_embeddings = matcher.encode([chunk["text"] for chunk in chunks])
                similarities = chunk_embeddings @ embeddings.T
                indices = similarities.argmax(dim=1).tolist()
                scores = similarities.max(dim=1).values.tolist()
                mentions = [
                    {
                        **chunk,
                        "object_index": int(index),
                        "match_score": float(score),
                        "object_text": texts[index],
                    }
                    for chunk, index, score in zip(chunks, indices, scores)
                ]
            else:
                mentions = []
                missing_objects += 1
            aligned = dict(record)
            aligned["mentions"] = mentions
            aligned["alignment_format"] = FORMAT_VERSION
            aligned["alignment_source_index"] = source_index
            stream.write(json.dumps(aligned, ensure_ascii=False) + "\n")
            processed += 1
            if processed == 1 or processed % args.log_every == 0:
                rate = processed / max(time.monotonic() - start_time, 1e-9)
                print(
                    f"align={processed} image={image_id} mentions={len(mentions)} rate={rate:.2f} q/s",
                    flush=True,
                )
    temporary.replace(output)
    write_json(
        summary_path,
        {
            "status": "complete",
            "format_version": FORMAT_VERSION,
            "split": args.split,
            "questions": processed,
            "questions_without_discovered_objects": missing_objects,
            "matcher_model": args.matcher_model,
            "spacy_model": args.spacy_model,
            "max_mentions": args.max_mentions,
            "num_shards": args.num_shards,
            "shard_index": args.shard_index,
            "max_samples": args.max_samples,
            "elapsed_seconds": time.monotonic() - start_time,
            "extractor_receives_questions_or_answers": False,
        },
    )


def merge_alignments(args) -> None:
    inputs = [Path(value) for value in args.inputs]
    output = Path(args.output)
    summary_path = output.with_suffix(".summary.json")
    if output.is_file() and summary_path.is_file():
        prior = json.loads(summary_path.read_text())
        if (
            prior.get("status") == "complete"
            and prior.get("format_version") == FORMAT_VERSION
            and int(prior.get("questions", -1)) == args.expected_questions
        ):
            print(f"Merged alignment already complete: {output}", flush=True)
            return
    for path in inputs:
        if not path.is_file() or not path.with_suffix(".summary.json").is_file():
            raise FileNotFoundError(f"Missing completed alignment shard: {path}")
    output.parent.mkdir(parents=True, exist_ok=True)
    streams = [path.open() for path in inputs]
    heap = []
    try:
        for shard, stream in enumerate(streams):
            line = stream.readline()
            if line:
                record = json.loads(line)
                heapq.heappush(heap, (int(record["alignment_source_index"]), shard, record))
        temporary = output.with_suffix(output.suffix + ".tmp")
        count = 0
        with temporary.open("w") as destination:
            while heap:
                source_index, shard, record = heapq.heappop(heap)
                if source_index != count:
                    raise ValueError(
                        f"Alignment merge is incomplete or duplicated: expected index {count}, got {source_index}"
                    )
                destination.write(json.dumps(record, ensure_ascii=False) + "\n")
                count += 1
                line = streams[shard].readline()
                if line:
                    next_record = json.loads(line)
                    heapq.heappush(
                        heap,
                        (int(next_record["alignment_source_index"]), shard, next_record),
                    )
        if count != args.expected_questions:
            raise ValueError(f"Expected {args.expected_questions} merged questions, found {count}")
        temporary.replace(output)
    finally:
        for stream in streams:
            stream.close()
    write_json(
        summary_path,
        {
            "status": "complete",
            "format_version": FORMAT_VERSION,
            "questions": count,
            "num_shards": len(inputs),
            "inputs": [str(path.resolve()) for path in inputs],
        },
    )
    print(f"Merged {count} aligned questions into {output}", flush=True)


class VQACachedDataset(Dataset):
    def __init__(
        self,
        questions: str | Path,
        cache_root: str | Path,
        split: str,
        answer_vocab: str | Path,
        max_samples: int = 0,
        object_cache_size: int = 8,
        require_answer_in_vocab: bool = False,
    ) -> None:
        self.questions = Path(questions)
        self.cache_root = Path(cache_root)
        self.split = split
        vocab = json.loads(Path(answer_vocab).read_text())
        if vocab.get("format_version") != FORMAT_VERSION:
            raise ValueError(f"Unexpected answer vocabulary format: {answer_vocab}")
        self.answers = list(vocab["answers"])
        self.answer_to_index = {answer: index for index, answer in enumerate(self.answers)}
        self.object_cache_size = object_cache_size
        self.offsets = []
        offset = 0
        with self.questions.open("rb") as stream:
            for line in stream:
                if line.strip():
                    if require_answer_in_vocab:
                        record = json.loads(line)
                        answer = normalize_answer(str(record.get("multiple_choice_answer", "")))
                        if answer not in self.answer_to_index:
                            offset += len(line)
                            continue
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
        bundle = torch.load(path, map_location="cpu", weights_only=False)
        semantic = bundle.get("semantic")
        visual = bundle.get("visual_memory")
        if not isinstance(semantic, torch.Tensor) or semantic.ndim != 2 or semantic.shape[1] != 1536:
            raise ValueError(f"Invalid semantic tensor in {path}: {getattr(semantic, 'shape', None)}")
        if not isinstance(visual, torch.Tensor) or visual.shape != (semantic.shape[0], 8, 1536):
            raise ValueError(f"Invalid visual-memory tensor in {path}: {getattr(visual, 'shape', None)}")
        value = (semantic.float().contiguous(), visual.float().contiguous())
        self._objects[image_id] = value
        while len(self._objects) > self.object_cache_size:
            self._objects.popitem(last=False)
        return value

    def __getitem__(self, index: int) -> dict:
        record = self._record(index)
        if record.get("alignment_format") != FORMAT_VERSION:
            raise ValueError(f"Question {record.get('question_id')} lacks verified alignment")
        image_id = str(record["image_id"])
        semantic, visual = self._object_tokens(image_id)
        scores: dict[int, float] = {}
        for answer in record.get("answers", []):
            normalized = normalize_answer(str(answer))
            answer_index = self.answer_to_index.get(normalized)
            if answer_index is not None:
                scores[answer_index] = scores.get(answer_index, 0.0) + 1.0
        scores = {key: min(value / 3.0, 1.0) for key, value in scores.items()}
        multiple_choice_answer = normalize_answer(str(record["multiple_choice_answer"]))
        return {
            "question_id": int(record["question_id"]),
            "image_id": image_id,
            "question": str(record["question"]),
            "multiple_choice_answer": multiple_choice_answer,
            "target_index": self.answer_to_index.get(multiple_choice_answer, -1),
            "answers": [normalize_answer(str(answer)) for answer in record.get("answers", [])],
            "mentions": list(record.get("mentions", [])),
            "semantic": semantic,
            "visual_memory": visual,
            "soft_scores": scores,
        }

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_stream"] = None
        state["_objects"] = OrderedDict()
        return state


class DeORVQAModel(nn.Module):
    MODES = ("no_coupling", "coupling")

    def __init__(
        self,
        text_encoder: nn.Module,
        tokenizer,
        answer_count: int,
        mode: str,
        output_layers: int = 2,
        output_heads: int = 64,
        output_dim: int = 128,
        projected_dim: int = 126,
    ) -> None:
        super().__init__()
        if mode not in self.MODES:
            raise ValueError(f"Unknown coupling mode: {mode}")
        self.text_encoder = text_encoder
        self.tokenizer = tokenizer
        self.mode = mode
        text_dim = int(text_encoder.config.d_model)
        if output_dim != projected_dim + 2:
            raise ValueError("CTRL-O-style output_dim must equal projected_dim + two type dimensions")
        if output_dim % output_heads:
            raise ValueError(f"output_dim={output_dim} is not divisible by output_heads={output_heads}")
        self.output_dim = output_dim
        self.projected_dim = projected_dim
        # CTRL-O inserts a linearly projected visual token into T5 for coupling.
        self.coupling_projector = (
            nn.Linear(3072, text_dim) if mode == "coupling" else None
        )
        # The downstream VQA setup projects image and T5 representations to
        # 126 dimensions and appends a fixed two-dimensional modality code.
        self.image_output_projector = (
            nn.Sequential(nn.Linear(3072, projected_dim), nn.Dropout(0.1))
            if mode == "no_coupling"
            else None
        )
        self.text_output_projector = nn.Sequential(
            nn.Linear(text_dim, projected_dim), nn.Dropout(0.1)
        )
        self.cls_token = nn.Parameter(torch.empty(output_dim))
        nn.init.normal_(self.cls_token, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=output_dim,
            nhead=output_heads,
            dim_feedforward=output_dim,
            dropout=0.1,
            activation="relu",
            batch_first=True,
            norm_first=False,
        )
        self.output_transformer = nn.TransformerEncoder(layer, num_layers=output_layers)
        self.classifier = nn.Sequential(
            nn.Linear(output_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.Linear(output_dim, answer_count),
        )

    def _fused_mentions(self, sample: dict, device: torch.device) -> torch.Tensor:
        indices = [int(mention["object_index"]) for mention in sample["mentions"]]
        valid = [index for index in indices if 0 <= index < sample["semantic"].shape[0]]
        if not valid:
            return torch.empty((0, 3072), device=device, dtype=torch.float32)
        semantic = sample["semantic"][valid].to(device=device, dtype=torch.float32)
        visual = sample["visual_memory"][valid].float().mean(dim=1).to(device)
        return torch.cat([semantic, visual], dim=-1)

    @staticmethod
    def _positional_encoding(length: int, width: int, device, dtype) -> torch.Tensor:
        positions = torch.arange(length, device=device, dtype=torch.float32)[:, None]
        frequencies = torch.exp(
            torch.arange(0, width, 2, device=device, dtype=torch.float32)
            * (-math.log(10000.0) / width)
        )
        result = torch.zeros((length, width), device=device, dtype=torch.float32)
        result[:, 0::2] = torch.sin(positions * frequencies)
        result[:, 1::2] = torch.cos(positions * frequencies)
        return result.to(dtype=dtype)

    def _add_type(self, values: torch.Tensor, image: bool) -> torch.Tensor:
        code = values.new_zeros((values.shape[0], 2))
        code[:, 0 if image else 1] = 1
        return torch.cat([values, code], dim=-1)

    def _tokenize(self, question: str):
        return self.tokenizer(
            question,
            add_special_tokens=True,
            truncation=True,
            max_length=128,
            return_offsets_mapping=True,
        )

    def _encoder_input_one(
        self, sample: dict
    ) -> tuple[torch.Tensor, torch.Tensor, int]:
        embedding = self.text_encoder.get_input_embeddings()
        device, dtype = embedding.weight.device, embedding.weight.dtype
        encoded = self._tokenize(sample["question"])
        ids = torch.tensor(encoded["input_ids"], dtype=torch.long, device=device)
        text_embeddings = embedding(ids)
        fused_objects = self._fused_mentions(sample, device)
        coupled_objects = (
            self.coupling_projector(fused_objects).to(dtype=dtype)
            if self.coupling_projector is not None
            else None
        )
        if self.mode == "coupling" and fused_objects.shape[0]:
            insertions: dict[int, list[torch.Tensor]] = {}
            offsets = encoded["offset_mapping"]
            object_cursor = 0
            for mention in sample["mentions"]:
                object_index = int(mention["object_index"])
                if object_index < 0 or object_index >= sample["semantic"].shape[0]:
                    continue
                end = int(mention["end"])
                position = 0
                for token_index, (_, token_end) in enumerate(offsets):
                    if token_end and token_end <= end:
                        position = token_index + 1
                insertions.setdefault(position, []).append(coupled_objects[object_cursor])
                object_cursor += 1
            pieces = []
            for position in range(text_embeddings.shape[0] + 1):
                if position in insertions:
                    pieces.extend(vector[None] for vector in insertions[position])
                if position < text_embeddings.shape[0]:
                    pieces.append(text_embeddings[position : position + 1])
            encoder_input = torch.cat(pieces, dim=0)
            return encoder_input, fused_objects, int(fused_objects.shape[0])
        return text_embeddings, fused_objects, int(fused_objects.shape[0])

    def forward(self, samples: list[dict]) -> tuple[torch.Tensor, list[int]]:
        encoder_rows = [self._encoder_input_one(sample) for sample in samples]
        encoder_max = max(row.shape[0] for row, _, _ in encoder_rows)
        text_dim = self.text_encoder.config.d_model
        device = self.cls_token.device
        encoder_dtype = self.text_encoder.get_input_embeddings().weight.dtype
        encoder_inputs = torch.zeros(
            (len(samples), encoder_max, text_dim), device=device, dtype=encoder_dtype
        )
        encoder_attention = torch.zeros(
            (len(samples), encoder_max), device=device, dtype=torch.long
        )
        for index, (row, _, _) in enumerate(encoder_rows):
            encoder_inputs[index, : row.shape[0]] = row.to(dtype=encoder_dtype)
            encoder_attention[index, : row.shape[0]] = 1
        encoded = self.text_encoder(
            inputs_embeds=encoder_inputs, attention_mask=encoder_attention
        ).last_hidden_state
        rows = []
        counts = []
        for index, (encoder_row, fused_objects, count) in enumerate(encoder_rows):
            text = self.text_output_projector(encoded[index, : encoder_row.shape[0]].float())
            text = text + self._positional_encoding(
                text.shape[0], text.shape[1], text.device, text.dtype
            )
            text = self._add_type(text, image=False)
            if self.mode == "no_coupling" and fused_objects.shape[0]:
                objects = self.image_output_projector(fused_objects.float())
                objects = self._add_type(objects, image=True)
                row = torch.cat([objects, text], dim=0)
            else:
                row = text
            rows.append(row)
            counts.append(count)
        max_length = max(row.shape[0] for row in rows) + 1
        device, dtype = self.cls_token.device, self.cls_token.dtype
        inputs = torch.zeros((len(rows), max_length, self.output_dim), device=device, dtype=dtype)
        padding = torch.ones((len(rows), max_length), device=device, dtype=torch.bool)
        for index, row in enumerate(rows):
            length = row.shape[0]
            inputs[index, 0] = self.cls_token
            inputs[index, 1 : length + 1] = row.to(dtype=dtype)
            padding[index, : length + 1] = False
        output = self.output_transformer(inputs, src_key_padding_mask=padding)
        return self.classifier(output[:, 0]), counts


def soft_targets(samples: list[dict], answer_count: int, device: torch.device) -> torch.Tensor:
    result = torch.zeros((len(samples), answer_count), device=device)
    for row, sample in enumerate(samples):
        for index, value in sample["soft_scores"].items():
            result[row, int(index)] = float(value)
    return result


def load_base_model(model_path: str, device: torch.device):
    from transformers import AutoConfig, AutoModel, AutoTokenizer, T5EncoderModel

    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    if not getattr(tokenizer, "is_fast", False):
        raise ValueError("A fast tokenizer is required for noun-phrase offset mapping")
    config = AutoConfig.from_pretrained(model_path)
    if config.model_type == "t5":
        # AutoModel resolves T5 checkpoints to the encoder-decoder T5Model,
        # which requires decoder_input_ids.  Coupling needs the encoder stack
        # only, just as in CTRL-O's VQA protocol.
        encoder = T5EncoderModel.from_pretrained(model_path, torch_dtype=torch.float32)
    else:
        encoder = AutoModel.from_pretrained(model_path, torch_dtype=torch.float32)
    return encoder.to(device), tokenizer


def make_model(args, answer_count: int, device: torch.device) -> DeORVQAModel:
    encoder, tokenizer = load_base_model(args.model_path, device)
    model = DeORVQAModel(
        encoder,
        tokenizer,
        answer_count,
        args.mode,
        args.output_layers,
        args.output_heads,
        args.output_dim,
        args.projected_dim,
    )
    return model.to(device)


def save_model(model: DeORVQAModel, output: Path, args, answers: list[str]) -> None:
    output.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output / "model.pt")
    write_json(
        output / "run_config.json",
        {
            "format_version": FORMAT_VERSION,
            "model_path": args.model_path,
            "mode": args.mode,
            "output_layers": args.output_layers,
            "output_heads": args.output_heads,
            "output_dim": args.output_dim,
            "projected_dim": args.projected_dim,
            "answers": answers,
            "representation": "one fused semantic + mean-pooled visual-memory token per noun chunk",
            "deor_frozen": True,
            "text_matcher_frozen": True,
            "text_encoder_finetuned": True,
            "objective": "cross_entropy over the top-3000 normalized answers",
            "optimizer": "Adam",
            "global_batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "max_steps": args.max_steps,
            "protocol_source": "CTRL-O Table 6 and its cited T-2 VQAv2 downstream setup",
        },
    )


def distributed_setup() -> tuple[int, int, int]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size > 1:
        dist.init_process_group("nccl")
    torch.cuda.set_device(local_rank)
    return rank, local_rank, world_size


def save_training_state(
    model: DeORVQAModel,
    optimizer: torch.optim.Optimizer,
    output: Path,
    step: int,
    initial_loss: float,
    args,
) -> None:
    output.mkdir(parents=True, exist_ok=True)
    path = output / "training_state.pt"
    temporary = output / "training_state.pt.tmp"
    torch.save(
        {
            "format_version": FORMAT_VERSION,
            "step": step,
            "initial_loss": initial_loss,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "mode": args.mode,
            "max_steps": args.max_steps,
            "global_batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
        },
        temporary,
    )
    temporary.replace(path)


def train(args) -> None:
    rank, local_rank, world_size = distributed_setup()
    random.seed(args.seed + rank)
    torch.manual_seed(args.seed + rank)
    torch.cuda.manual_seed(args.seed + rank)
    device = torch.device("cuda", local_rank)
    if args.batch_size % world_size:
        raise ValueError(
            f"Global batch size {args.batch_size} must be divisible by world size {world_size}"
        )
    local_batch_size = args.batch_size // world_size
    vocab = json.loads(Path(args.answer_vocab).read_text())
    answers = list(vocab["answers"])
    dataset = VQACachedDataset(
        args.questions,
        args.cache_root,
        args.split,
        args.answer_vocab,
        args.max_samples,
        args.object_cache_size,
        require_answer_in_vocab=True,
    )
    full_batches_per_epoch = len(dataset) // args.batch_size
    if full_batches_per_epoch < 1:
        raise ValueError("Training set is smaller than one global batch")
    model = make_model(args, len(answers), device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = nn.CrossEntropyLoss()
    output = Path(args.output_dir)
    step = 0
    initial_loss = float("nan")
    checkpoint = output / "training_state.pt"
    if args.resume and checkpoint.is_file():
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if state.get("format_version") != FORMAT_VERSION or state.get("mode") != args.mode:
            raise ValueError(f"Incompatible training checkpoint: {checkpoint}")
        if int(state.get("global_batch_size", -1)) != args.batch_size:
            raise ValueError("Cannot resume with a different global batch size")
        model.load_state_dict(state["model"], strict=True)
        optimizer.load_state_dict(state["optimizer"])
        step = int(state["step"])
        initial_loss = float(state["initial_loss"])
        if rank == 0:
            print(f"Resuming mode={args.mode} from step={step}: {checkpoint}", flush=True)
    start_step = step
    wrapped = (
        DistributedDataParallel(model, device_ids=[local_rank], find_unused_parameters=False)
        if world_size > 1
        else model
    )
    wrapped.train()
    start_time = time.monotonic()
    optimizer.zero_grad(set_to_none=True)
    permutation_epoch = -1
    permutation = None
    final_loss = float("nan")
    while step < args.max_steps:
        epoch = step // full_batches_per_epoch
        batch_in_epoch = step % full_batches_per_epoch
        if epoch != permutation_epoch:
            generator = torch.Generator().manual_seed(args.seed + epoch)
            permutation = torch.randperm(len(dataset), generator=generator)
            permutation_epoch = epoch
        start = batch_in_epoch * args.batch_size
        global_indices = permutation[start : start + args.batch_size].tolist()
        local_indices = global_indices[rank::world_size]
        if len(local_indices) != local_batch_size:
            raise AssertionError("Uneven distributed batch")
        samples = [dataset[index] for index in local_indices]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits, _ = wrapped(samples)
            targets = torch.tensor(
                [sample["target_index"] for sample in samples], dtype=torch.long, device=device
            )
            loss = criterion(logits, targets) / args.gradient_accumulation_steps
        loss.backward()
        if (step + 1) % args.gradient_accumulation_steps == 0:
            nn.utils.clip_grad_norm_(wrapped.parameters(), args.max_grad_norm)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        reduced = loss.detach() * args.gradient_accumulation_steps
        if world_size > 1:
            dist.all_reduce(reduced, op=dist.ReduceOp.AVG)
        final_loss = float(reduced)
        if math.isnan(initial_loss):
            initial_loss = final_loss
        step += 1
        if rank == 0 and (step == 1 or step % args.log_every == 0 or step == args.max_steps):
            elapsed = time.monotonic() - start_time
            print(
                f"mode={args.mode} step={step}/{args.max_steps} loss={final_loss:.6f} "
                f"step/s={max(step - start_step, 1) / elapsed:.3f}",
                flush=True,
            )
        if args.checkpoint_every and step % args.checkpoint_every == 0:
            if world_size > 1:
                dist.barrier()
            if rank == 0:
                base = wrapped.module if isinstance(wrapped, DistributedDataParallel) else wrapped
                save_training_state(base, optimizer, output, step, initial_loss, args)
                print(f"Saved resumable checkpoint at step={step}", flush=True)
            if world_size > 1:
                dist.barrier()
    if step % args.gradient_accumulation_steps:
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    elapsed = time.monotonic() - start_time
    if world_size > 1:
        dist.barrier()
    if rank == 0:
        base = wrapped.module if isinstance(wrapped, DistributedDataParallel) else wrapped
        if not args.benchmark_only:
            save_training_state(base, optimizer, output, step, initial_loss, args)
            save_model(base, output, args, answers)
        write_json(
            output / "train_summary.json",
            {
                "status": "complete",
                "mode": args.mode,
                "objective": "cross_entropy",
                "steps": step,
                "initial_loss": initial_loss,
                "final_loss": final_loss,
                "elapsed_seconds_this_invocation": elapsed,
                "elapsed_seconds": elapsed,
                "steps_per_second_this_invocation": max(step - start_step, 1) / elapsed,
                "steps_per_second": max(step - start_step, 1) / elapsed,
                "num_train_samples_in_top3000": len(dataset),
                "world_size": world_size,
                "global_batch_size": args.batch_size,
                "local_batch_size": local_batch_size,
                "peak_gpu_memory_gib_rank0": torch.cuda.max_memory_allocated(device) / 2**30,
                "peak_gpu_memory_gib": torch.cuda.max_memory_allocated(device) / 2**30,
                "benchmark_only": bool(args.benchmark_only),
            },
        )
    if world_size > 1:
        dist.barrier()
        dist.destroy_process_group()


def load_saved_model(args, answer_count: int, device: torch.device) -> tuple[DeORVQAModel, dict]:
    config = json.loads((Path(args.output_dir) / "run_config.json").read_text())
    args.model_path = config["model_path"]
    args.mode = config["mode"]
    args.output_layers = int(config["output_layers"])
    args.output_heads = int(config["output_heads"])
    args.output_dim = int(config.get("output_dim", 128))
    args.projected_dim = int(config.get("projected_dim", 126))
    model = make_model(args, answer_count, device)
    state = torch.load(Path(args.output_dir) / "model.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(state, strict=True)
    return model.eval(), config


def evaluate(args) -> None:
    torch.cuda.set_device(0)
    device = torch.device("cuda", 0)
    vocab = json.loads(Path(args.answer_vocab).read_text())
    answers = list(vocab["answers"])
    dataset = VQACachedDataset(
        args.questions,
        args.cache_root,
        args.split,
        args.answer_vocab,
        args.max_samples,
        args.object_cache_size,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=lambda x: x)
    model, config = load_saved_model(args, len(answers), device)
    rows = []
    score_sum = 0.0
    exact_sum = 0
    object_count = 0
    start_time = time.monotonic()
    with torch.inference_mode():
        for samples in loader:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits, counts = model(samples)
            predictions = logits.argmax(dim=-1).tolist()
            for sample, prediction_index, count in zip(samples, predictions, counts):
                prediction = answers[prediction_index]
                answer_list = sample["answers"]
                per_annotator = []
                for annotator, answer in enumerate(answer_list):
                    other_matches = sum(
                        prediction == other
                        for index, other in enumerate(answer_list)
                        if index != annotator
                    )
                    per_annotator.append(min(other_matches / 3.0, 1.0))
                soft_accuracy = (
                    sum(per_annotator) / len(per_annotator) if per_annotator else 0.0
                )
                exact = prediction == sample["multiple_choice_answer"]
                score_sum += soft_accuracy
                exact_sum += int(exact)
                object_count += count
                rows.append(
                    {
                        "question_id": sample["question_id"],
                        "image_id": sample["image_id"],
                        "question": sample["question"],
                        "answer": sample["multiple_choice_answer"],
                        "prediction": prediction,
                        "vqa_official_accuracy": soft_accuracy,
                        "classification_accuracy": exact,
                        "num_coupled_object_tokens": count,
                    }
                )
    result = Path(args.result_dir) if args.result_dir else Path(args.output_dir) / "eval"
    result.mkdir(parents=True, exist_ok=True)
    (result / "predictions.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    )
    write_json(
        result / "summary.json",
        {
            "status": "complete",
            "protocol": FORMAT_VERSION,
            "mode": config["mode"],
            "num_questions": len(rows),
            "vqa_official_accuracy": score_sum / len(rows),
            "vqa_soft_accuracy": score_sum / len(rows),
            "classification_accuracy": exact_sum / len(rows),
            "normalized_exact_match": exact_sum / len(rows),
            "mean_selected_object_tokens": object_count / len(rows),
            "elapsed_seconds": time.monotonic() - start_time,
            "checkpoint_reload_verified": True,
        },
    )


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser()
    sub = root.add_subparsers(dest="command", required=True)

    vocab = sub.add_parser("vocab")
    vocab.add_argument("--train_questions", required=True)
    vocab.add_argument("--output", required=True)
    vocab.add_argument("--size", type=int, default=3000)

    align = sub.add_parser("align")
    align.add_argument("--questions", required=True)
    align.add_argument("--cache_root", required=True)
    align.add_argument("--split", choices=("train", "val"), required=True)
    align.add_argument("--output", required=True)
    align.add_argument("--max_samples", type=int, default=0)
    align.add_argument("--max_mentions", type=int, default=4)
    align.add_argument("--matcher_model", default="openai/clip-vit-base-patch32")
    align.add_argument("--matcher_batch_size", type=int, default=64)
    align.add_argument("--spacy_model", default="en_core_web_sm")
    align.add_argument("--device", default="cuda")
    align.add_argument("--object_text_cache_size", type=int, default=64)
    align.add_argument("--log_every", type=int, default=100)
    align.add_argument("--num_shards", type=int, default=1)
    align.add_argument("--shard_index", type=int, default=0)

    merge = sub.add_parser("merge")
    merge.add_argument("--inputs", nargs="+", required=True)
    merge.add_argument("--output", required=True)
    merge.add_argument("--expected_questions", type=int, required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--questions", required=True)
    common.add_argument("--cache_root", required=True)
    common.add_argument("--split", choices=("train", "val"), required=True)
    common.add_argument("--answer_vocab", required=True)
    common.add_argument("--output_dir", required=True)
    common.add_argument("--max_samples", type=int, default=0)
    common.add_argument("--batch_size", type=int, default=8)
    common.add_argument("--object_cache_size", type=int, default=8)

    train_parser = sub.add_parser("train", parents=[common])
    train_parser.add_argument("--model_path", default="google-t5/t5-base")
    train_parser.add_argument("--mode", choices=DeORVQAModel.MODES, required=True)
    train_parser.add_argument("--max_steps", type=int, default=2)
    train_parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    train_parser.add_argument("--learning_rate", type=float, default=1e-4)
    train_parser.add_argument("--max_grad_norm", type=float, default=1.0)
    train_parser.add_argument("--output_layers", type=int, default=2)
    train_parser.add_argument("--output_heads", type=int, default=64)
    train_parser.add_argument("--output_dim", type=int, default=128)
    train_parser.add_argument("--projected_dim", type=int, default=126)
    train_parser.add_argument("--checkpoint_every", type=int, default=0)
    train_parser.add_argument("--log_every", type=int, default=100)
    train_parser.add_argument("--resume", action="store_true")
    train_parser.add_argument("--benchmark_only", action="store_true")
    train_parser.add_argument("--seed", type=int, default=42)

    eval_parser = sub.add_parser("eval", parents=[common])
    eval_parser.add_argument("--result_dir")
    return root


def main() -> None:
    args = parser().parse_args()
    if args.command == "vocab":
        build_answer_vocab(Path(args.train_questions), Path(args.output), args.size)
    elif args.command == "align":
        if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
            raise ValueError("Require num_shards >= 1 and 0 <= shard_index < num_shards")
        prepare_alignments(args)
    elif args.command == "merge":
        merge_alignments(args)
    elif args.command == "train":
        train(args)
    else:
        evaluate(args)


if __name__ == "__main__":
    main()
