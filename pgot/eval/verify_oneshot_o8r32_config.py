"""Fail fast unless a checkpoint is the standard DeOR O8/R32 no-prior model."""

import argparse
import json
from pathlib import Path


def verify(folder: str, model_size: str, require_scratch_init: bool = False) -> None:
    if model_size not in {"1.5b", "7b"}:
        raise ValueError(f"Unknown model size: {model_size}")
    config_path = Path(folder) / "config.json"
    config = json.loads(config_path.read_text())
    expected = {
        "pgot_e8_owner_weight": 1.0,
        "pgot_e8_reader_supervision_mode": "writer",
        "pgot_e8_reader_object_weight": 0.5,
        "pgot_e8_reader_background_weight": 0.25,
        "pgot_e11_object_memories_per_owner": 8,
        "pgot_e11_register_memories_per_owner": 32,
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
        raise ValueError(f"Not the standard DeOR O8/R32 no-prior checkpoint: {errors}")
    print(
        "PASS checkpoint: standard DeOR O8/R32 no-prior; "
        f"model={model_size}; Reader supervision=writer; path={folder}",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--model_size", choices=("1.5b", "7b"), required=True)
    parser.add_argument("--require_scratch_init", action="store_true")
    args = parser.parse_args()
    verify(args.model_path, args.model_size, args.require_scratch_init)


if __name__ == "__main__":
    main()
