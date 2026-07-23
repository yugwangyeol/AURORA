#!/usr/bin/env python
"""Filter a JSONL manifest to image IDs present in a reference JSONL."""

import argparse
import json
from pathlib import Path


def load_image_ids(path: Path) -> set[int]:
    image_ids: set[int] = set()
    with path.open() as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            if "image_id" not in record:
                raise ValueError(f"Missing image_id in {path}:{line_number}")
            image_ids.add(int(record["image_id"]))
    return image_ids


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    input_path = Path(args.input)
    reference_path = Path(args.reference)
    output_path = Path(args.output)
    reference_ids = load_image_ids(reference_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(output_path.name + ".tmp")
    input_count = kept_count = 0
    kept_ids: set[int] = set()
    with input_path.open() as source, temporary_path.open("w") as destination:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            input_count += 1
            record = json.loads(line)
            if "image_id" not in record:
                raise ValueError(f"Missing image_id in {input_path}:{line_number}")
            image_id = int(record["image_id"])
            if image_id not in reference_ids:
                continue
            destination.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
            kept_count += 1
            kept_ids.add(image_id)

    temporary_path.replace(output_path)
    print(
        json.dumps(
            {
                "input": str(input_path.resolve()),
                "reference": str(reference_path.resolve()),
                "output": str(output_path.resolve()),
                "input_records": input_count,
                "reference_image_ids": len(reference_ids),
                "records_written": kept_count,
                "reference_ids_missing_from_input": len(reference_ids - kept_ids),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
