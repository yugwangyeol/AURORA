#!/usr/bin/env python3
"""Fail-fast audit for the matched fixed-query checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from safetensors import safe_open


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--count", type=int, default=16)
    args = parser.parse_args()

    config_path = args.checkpoint / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(config_path)
    config = json.loads(config_path.read_text())
    expected = {
        "pgot_fixed_query_owner_enable": True,
        "pgot_fixed_query_owner_count": args.count,
        "pgot_one_shot_reader_enable": True,
        "pgot_one_shot_readout_mode": "memory_content",
        "pgot_one_shot_writer_owner_prior": False,
        "pgot_e11_object_memories_per_owner": 8,
        "pgot_e11_register_memories_per_owner": 32,
        "pgot_n_register": 4,
        "pgot_e8_reader_supervision_mode": "writer",
    }
    for key, value in expected.items():
        if config.get(key) != value:
            raise AssertionError(f"{key}: expected {value!r}, got {config.get(key)!r}")

    shards = sorted(args.checkpoint.glob("*.safetensors"))
    if not shards:
        raise FileNotFoundError(f"No safetensors checkpoint in {args.checkpoint}")
    found_shape = None
    suffix = "pgot_fixed_object_queries"
    for shard in shards:
        with safe_open(str(shard), framework="pt", device="cpu") as handle:
            for key in handle.keys():
                if key.endswith(suffix):
                    found_shape = tuple(handle.get_slice(key).get_shape())
                    break
        if found_shape is not None:
            break
    expected_shape = (args.count, int(config["hidden_size"]))
    if found_shape != expected_shape:
        raise AssertionError(
            f"fixed query tensor: expected {expected_shape}, got {found_shape}"
        )
    print(
        f"PASS: fixed-query K={args.count}, O8/R32, R4, no-prior, "
        f"writer-supervised checkpoint: {args.checkpoint}"
    )


if __name__ == "__main__":
    main()
