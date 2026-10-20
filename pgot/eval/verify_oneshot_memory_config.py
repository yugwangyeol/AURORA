"""Validate a DeOR one-shot-memory checkpoint before an ablation evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def verify(
    folder: str,
    *,
    model_size: str,
    object_memories: int,
    register_memories: int,
    require_scratch_init: bool = False,
    caption_init: bool | None = None,
    own_caption_attention: bool | None = None,
) -> None:
    if model_size not in {"1.5b", "7b"}:
        raise ValueError(f"Unknown model size: {model_size}")
    if object_memories < 1 or register_memories < 1:
        raise ValueError("Memory counts must be positive")

    config_path = Path(folder) / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"Missing checkpoint config: {config_path}")
    config = json.loads(config_path.read_text())

    expected = {
        "pgot_e8_owner_weight": 1.0,
        "pgot_e8_reader_supervision_mode": "writer",
        "pgot_e8_reader_object_weight": 0.5,
        "pgot_e8_reader_background_weight": 0.25,
        "pgot_e11_memories_per_owner": max(object_memories, register_memories),
        "pgot_e11_object_memories_per_owner": object_memories,
        "pgot_e11_register_memories_per_owner": register_memories,
        "pgot_n_register": 4,
        "pgot_one_shot_reader_enable": True,
        "pgot_one_shot_readout_mode": "memory_content",
        "pgot_one_shot_writer_softmax_axis": "patch",
        "pgot_one_shot_writer_owner_prior": False,
        "pgot_lm_loss_weight": 1.0,
        "pgot_recon_loss_weight": 1.5,
        "hidden_size": 1536 if model_size == "1.5b" else 3584,
    }
    if model_size == "7b":
        expected.update(
            diffusion_model_hidden_size=4096,
            diffusion_model_z_channels=4096,
        )
    if require_scratch_init:
        expected["pgot_one_shot_scratch_init"] = True
    if caption_init is not None:
        expected["pgot_ovt_caption_init"] = caption_init
    if own_caption_attention is not None:
        expected["pgot_ovt_attends_own_caption"] = own_caption_attention

    errors = {
        key: {"expected": value, "actual": config.get(key)}
        for key, value in expected.items()
        if config.get(key) != value
    }
    for key in (
        "pgot_one_shot_direct_rae_query",
        "pgot_one_shot_rae_query_adapter_enable",
        "pgot_one_shot_memory_contrastive_enable",
    ):
        if config.get(key, False):
            errors[key] = {"expected": False, "actual": config[key]}
    if config.get("pgot_one_shot_memory_value_source", "siglip") != "siglip":
        errors["pgot_one_shot_memory_value_source"] = {
            "expected": "siglip",
            "actual": config.get("pgot_one_shot_memory_value_source"),
        }

    if errors:
        raise ValueError(f"Checkpoint configuration mismatch: {errors}")
    print(
        "PASS checkpoint: "
        f"DeOR {model_size}, O{object_memories}/R{register_memories}, no-prior, "
        f"scratch_required={require_scratch_init}; path={folder}",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--model_size", choices=("1.5b", "7b"), required=True)
    parser.add_argument("--object_memories", type=int, required=True)
    parser.add_argument("--register_memories", type=int, required=True)
    parser.add_argument("--require_scratch_init", action="store_true")
    parser.add_argument(
        "--caption_init",
        choices=("true", "false"),
        default=None,
        help="Optionally require the description-average OVT initialization path.",
    )
    parser.add_argument(
        "--own_caption_attention",
        choices=("true", "false"),
        default=None,
        help="Optionally require OVT attention to its own description span.",
    )
    args = parser.parse_args()
    verify(
        args.model_path,
        model_size=args.model_size,
        object_memories=args.object_memories,
        register_memories=args.register_memories,
        require_scratch_init=args.require_scratch_init,
        caption_init=None if args.caption_init is None else args.caption_init == "true",
        own_caption_attention=(
            None
            if args.own_caption_attention is None
            else args.own_caption_attention == "true"
        ),
    )


if __name__ == "__main__":
    main()
