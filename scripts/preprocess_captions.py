"""
Preprocess COCO captions with spaCy noun chunk extraction.

Usage:
    python scripts/preprocess_captions.py \
        --caption_json /home/jovyan/data/coco/annotations/captions_train2017.json \
        --tokenizer_path /home/jovyan/data/Scale-RAE-Qwen1.5B_DiT2.4B \
        --output_path /home/jovyan/data/coco/captions_with_chunks_train2017.json

    python scripts/preprocess_captions.py \
        --caption_json /home/jovyan/data/coco/annotations/captions_val2017.json \
        --tokenizer_path /home/jovyan/data/Scale-RAE-Qwen1.5B_DiT2.4B \
        --output_path /home/jovyan/data/coco/captions_with_chunks_val2017.json
"""

import json
import argparse

import spacy
from transformers import AutoTokenizer


def find_token_span(offsets, char_start, char_end):
    """Map character span to token span using offset mapping."""
    tok_start = None
    tok_end = None
    for i, (cs, ce) in enumerate(offsets):
        if cs == 0 and ce == 0 and i > 0:
            continue
        if cs < char_end and ce > char_start:
            if tok_start is None:
                tok_start = i
            tok_end = i + 1
    return tok_start, tok_end


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--caption_json", type=str, required=True,
                        help="Path to COCO captions JSON (e.g. captions_train2017.json)")
    parser.add_argument("--tokenizer_path", type=str, required=True,
                        help="Path to Qwen2 tokenizer (e.g. Scale-RAE model dir)")
    parser.add_argument("--output_path", type=str, required=True,
                        help="Output JSON path")
    parser.add_argument("--max_caption_tokens", type=int, default=64,
                        help="Maximum caption token length")
    args = parser.parse_args()

    print("Loading spaCy model...")
    nlp = spacy.load("en_core_web_sm")

    print(f"Loading tokenizer from {args.tokenizer_path}...")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, use_fast=True)

    print(f"Loading captions from {args.caption_json}...")
    with open(args.caption_json) as f:
        coco = json.load(f)

    # Group captions by image_id, take the first one per image
    image_captions = {}
    for ann in coco["annotations"]:
        img_id = str(ann["image_id"])
        if img_id not in image_captions:
            image_captions[img_id] = ann["caption"].strip()

    print(f"Processing {len(image_captions)} images...")
    results = {}
    skipped = 0

    for idx, (img_id, caption) in enumerate(image_captions.items()):
        doc = nlp(caption)

        encoding = tokenizer(
            caption,
            add_special_tokens=False,
            return_offsets_mapping=True,
            max_length=args.max_caption_tokens,
            truncation=True,
        )
        token_ids = encoding.input_ids
        offsets = encoding.offset_mapping

        noun_chunks = []
        for chunk in doc.noun_chunks:
            tok_s, tok_e = find_token_span(offsets, chunk.start_char, chunk.end_char)
            if tok_s is not None and tok_e is not None and tok_e <= len(token_ids):
                noun_chunks.append({
                    "text": chunk.text,
                    "token_start": tok_s,
                    "token_end": tok_e,
                })

        if len(noun_chunks) == 0:
            skipped += 1
            continue

        results[img_id] = {
            "caption": caption,
            "token_ids": token_ids,
            "noun_chunks": noun_chunks,
        }

        if (idx + 1) % 10000 == 0:
            print(f"  Processed {idx + 1}/{len(image_captions)} images...")

    with open(args.output_path, "w") as f:
        json.dump(results, f)

    print(f"\nDone! Saved {len(results)} entries to {args.output_path}")
    print(f"Skipped {skipped} images with no noun chunks")

    # Stats
    n_chunks = [len(r["noun_chunks"]) for r in results.values()]
    n_tokens = [len(r["token_ids"]) for r in results.values()]
    print(f"Noun chunks per image: mean={sum(n_chunks)/len(n_chunks):.1f}, "
          f"min={min(n_chunks)}, max={max(n_chunks)}")
    print(f"Caption tokens: mean={sum(n_tokens)/len(n_tokens):.1f}, "
          f"min={min(n_tokens)}, max={max(n_tokens)}")


if __name__ == "__main__":
    main()
