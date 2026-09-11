"""Verify the owner-gradient + feature-loss smoke artifacts."""
import argparse
import json
import math
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--stage", choices=["train", "eval"], required=True)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--softmax-axis", choices=["patch", "memory"], default="patch")
    args = parser.parse_args()

    if args.stage == "eval":
        for branch in ("tf", "ar"):
            summary = json.loads((args.root / "eval" / branch / "summary.json").read_text())
            assert summary["one_shot_memory_enabled"] is True
            assert summary["memory_reader_key_mode"] == "content"
            assert summary["e11_object_memories_per_owner"] == 4
            assert summary["memory_writer_softmax_axis"] == args.softmax_axis
            assert summary["teacher_forced_caption"] is (branch == "tf")
            assert summary["num_samples"] == 2
            for key in ("recon_mse", "recon_psnr", "rFID"):
                assert math.isfinite(summary[key]), (branch, key, summary.get(key))
        print("TF/AR checkpoint reload + reconstruction PASS")
        return

    ckpt = args.root / "train" / f"checkpoint-{args.steps}"
    cfg = json.loads((ckpt / "config.json").read_text())
    assert cfg["pgot_one_shot_readout_mode"] == "memory_content"
    assert cfg["pgot_one_shot_detach_owner_routing"] is False
    assert cfg["pgot_one_shot_owner_gradient_ramp_steps"] == 2
    assert cfg["pgot_one_shot_writer_softmax_axis"] == args.softmax_axis
    assert cfg["pgot_latent_distill_enable"] is True
    assert cfg["pgot_latent_distill_weight"] == 0.5
    assert cfg["pgot_latent_distill_ramp_steps"] == 2

    state = json.loads((ckpt / "trainer_state.json").read_text())
    assert state["global_step"] == args.steps
    histories = state["log_history"]
    required = (
        "loss_recon", "loss_e8_owner", "loss_e8_reader",
        "loss_latent_distill", "latent_distill_mse", "latent_distill_cos",
        "latent_distill_weight_effective", "owner_gradient_scale",
        "eval_loss", "eval_loss_recon", "eval_loss_latent_distill",
        "eval_latent_distill_weight_effective", "eval_owner_gradient_scale",
    )
    for key in required:
        values = [row[key] for row in histories if key in row]
        assert values and all(math.isfinite(x) for x in values), (key, values)
    if args.softmax_axis == "memory":
        allocation_metrics = (
            "memory_object_allocation_entropy",
            "memory_object_allocation_max_share",
            "memory_register_allocation_entropy",
            "memory_register_allocation_max_share",
            "eval_memory_object_allocation_entropy",
            "eval_memory_object_allocation_max_share",
            "eval_memory_register_allocation_entropy",
            "eval_memory_register_allocation_max_share",
        )
        for key in allocation_metrics:
            values = [row[key] for row in histories if key in row]
            assert values and all(math.isfinite(x) for x in values), (key, values)
    assert max(row.get("owner_gradient_scale", 0.0) for row in histories) == 1.0
    assert max(row.get("latent_distill_weight_effective", 0.0) for row in histories) == 0.5

    banned = (
        "loss_contrastive", "loss_e8_causal", "e8_write_gate_mean",
        "one_shot_hard_outside_mass", "e9_gru", "e12_centroid",
        "dit_soft_routing", "latent_distill_l1",
    )
    for row in histories:
        assert not any(any(b in key for b in banned) for key in row), row

    from safetensors import safe_open
    keys = set()
    final_weight = None
    for shard in ckpt.glob("*.safetensors"):
        with safe_open(shard, framework="pt", device="cpu") as handle:
            keys.update(handle.keys())
            name = "pgot_latent_head.3.weight"
            if name in handle.keys():
                final_weight = handle.get_tensor(name)
    assert any(key.startswith("pgot_latent_head.") for key in keys)
    assert final_weight is not None and final_weight.abs().max().item() > 0

    logs = (args.root / "train.log").read_text()
    assert "train_runtime" in logs
    assert "recon image logging failed" not in logs
    tables = list((args.root / "wandb").glob("**/files/media/table/**/*.table.json"))
    assert tables, "W&B eval image table missing"
    table_text = " ".join(path.read_text() for path in tables)
    assert "our_recon" in table_text and "ovt_owner" in table_text
    print("train + in-train eval + save + active metrics + W&B table PASS")


if __name__ == "__main__":
    main()
