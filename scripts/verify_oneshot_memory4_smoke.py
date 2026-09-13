"""Verify actual checkpoint, train/eval metrics, W&B media, and TF/AR outputs."""
import argparse
import json
import math
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("mode", choices=["content", "id"])
    parser.add_argument("--stage", choices=["train", "eval"], required=True)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument(
        "--owner-prior", choices=["enabled", "disabled"], default="enabled"
    )
    parser.add_argument(
        "--direct-rae-query", choices=["enabled", "disabled"], default="disabled"
    )
    parser.add_argument("--expect-kid", action="store_true")
    parser.add_argument("--expect-class-metrics", action="store_true")
    parser.add_argument("--object-memories", type=int, default=4)
    parser.add_argument("--register-memories", type=int, default=16)
    parser.add_argument("--semantic-registers", type=int, default=4)
    args = parser.parse_args()
    run_root = args.root / args.mode
    if args.stage == "eval":
        for branch in ("tf", "ar"):
            summary = json.loads((run_root / "eval" / branch / "summary.json").read_text())
            assert summary["one_shot_memory_enabled"] is True
            assert summary["e11_object_memories_per_owner"] == args.object_memories
            assert summary["background_visual_memories"] == (
                args.register_memories * args.semantic_registers
            )
            assert summary["memory_reader_key_mode"] == args.mode
            assert summary["memory_writer_softmax_axis"] == "patch"
            assert summary["memory_writer_owner_prior"] is (
                args.owner_prior == "enabled"
            )
            assert summary["one_shot_direct_rae_query"] is (
                args.direct_rae_query == "enabled"
            )
            assert summary["mllm_rae_query_tokens"] == (
                0 if args.direct_rae_query == "enabled" else 256
            )
            assert summary["teacher_forced_caption"] is (branch == "tf")
            assert summary["num_samples"] == 2
            for key in ("recon_mse", "recon_psnr", "rFID"):
                assert math.isfinite(summary[key]), (branch, key, summary.get(key))
            if args.expect_kid:
                for key in ("KID_x1000", "KID_std_x1000"):
                    assert math.isfinite(summary[key]), (
                        branch, key, summary.get(key)
                    )
            if args.expect_class_metrics:
                assert summary["class_metrics_enabled"] is True
                assert summary["class_gt_granularity"] == "category"
                for key in ("mBO_i", "mIoU_i", "mBO_c", "mIoU_c"):
                    assert math.isfinite(summary[key]), (
                        branch, key, summary.get(key)
                    )
                assert summary["mBO_c_num_samples"] == 2
                assert summary["mIoU_c_num_samples"] == 2
            if branch == "ar":
                assert math.isfinite(summary["ar_object_count_mae"])
            print(
                f"{args.mode} {branch}: reload + reconstruction + rFID/KID"
                " PASS (2-image smoke only)"
            )
        return

    ckpt = run_root / "train" / f"checkpoint-{args.steps}"
    cfg = json.loads((ckpt / "config.json").read_text())
    assert cfg["pgot_one_shot_readout_mode"] == f"memory_{args.mode}"
    assert cfg["pgot_e11_object_memories_per_owner"] == args.object_memories
    assert cfg["pgot_e11_register_memories_per_owner"] == args.register_memories
    assert cfg["pgot_n_register"] == args.semantic_registers
    assert cfg["pgot_n_ovt_per_object"] == 1
    assert cfg["pgot_one_shot_writer_softmax_axis"] == "patch"
    assert cfg["pgot_one_shot_writer_owner_prior"] is (
        args.owner_prior == "enabled"
    )
    assert cfg["pgot_one_shot_direct_rae_query"] is (
        args.direct_rae_query == "enabled"
    )
    state = json.loads((ckpt / "trainer_state.json").read_text())
    assert state["global_step"] == args.steps
    histories = state["log_history"]
    for key in ("loss_recon", "loss_e8_owner", "loss_e8_reader", "memory_write_entropy",
                "memory_object_pair_cosine", "memory_reader_entropy", "eval_loss",
                "eval_loss_recon", "eval_memory_reader_entropy",
                "one_shot_direct_rae_query_enabled", "mllm_rae_query_tokens"):
        values = [row[key] for row in histories if key in row]
        assert values and all(math.isfinite(x) for x in values), (key, values)
    expected_direct = 1.0 if args.direct_rae_query == "enabled" else 0.0
    expected_mllm_tokens = 0.0 if args.direct_rae_query == "enabled" else 256.0
    assert all(
        row.get("one_shot_direct_rae_query_enabled", expected_direct) == expected_direct
        for row in histories
    )
    assert all(
        row.get("mllm_rae_query_tokens", expected_mllm_tokens) == expected_mllm_tokens
        for row in histories
    )
    banned = (
        "loss_contrastive", "loss_e8_causal", "e8_write_gate_mean",
        "one_shot_hard_outside_mass", "e9_gru", "e12_centroid",
        "dit_soft_routing", "latent_distill", "owner_gradient_scale",
        "memory_object_allocation", "memory_register_allocation",
    )
    for row in histories:
        assert not any(any(b in key for b in banned) for key in row), row
    from safetensors import safe_open
    weights = {}
    for shard in ckpt.glob("*.safetensors"):
        with safe_open(shard, framework="pt", device="cpu") as handle:
            for key in handle.keys():
                if key.startswith(("pgot_e8_writer.", "pgot_e8_reader.")):
                    weights[key] = handle.get_tensor(key)
    ids = weights["pgot_e8_writer.memory_id_embeddings"]
    expected_id_count = max(args.object_memories, args.register_memories)
    assert ids.shape[0] == expected_id_count and ids[:args.object_memories].std().item() > 0.01
    assert not any("gate" in key or "memory_to_query" in key for key in weights)
    if args.mode == "content":
        assert "pgot_e8_reader.content_key.weight" in weights
    else:
        assert "pgot_e8_reader.memory_key_embeddings" in weights
    # Compare with source IDs: optimizer actually updated the new Writer.
    source_path = Path((run_root / "source.txt").read_text().strip())
    for shard in source_path.glob("*.safetensors"):
        with safe_open(shard, framework="pt", device="cpu") as handle:
            key = "pgot_e8_writer.memory_id_embeddings"
            if key in handle.keys():
                source_ids = handle.get_tensor(key)
                shared_rows = min(ids.shape[0], source_ids.shape[0])
                delta = (ids[:shared_rows] - source_ids[:shared_rows]).abs().max().item()
                assert delta > 0, "Writer did not update"
                print(f"{args.mode}: Writer ID max update = {delta:.7g}")
                break
    else:
        raise AssertionError("source E11 Writer IDs missing")
    logs = (run_root / "train.log").read_text()
    assert "train_runtime" in logs
    assert "recon image logging failed" not in logs
    tables = list((run_root / "wandb").glob("**/files/media/table/**/*.table.json"))
    assert tables, "W&B eval image table missing"
    table_text = " ".join(path.read_text() for path in tables)
    assert "our_recon" in table_text and "ovt_owner" in table_text
    print(f"{args.mode}: train + in-train eval + model save + filtered metrics + W&B image table PASS")


if __name__ == "__main__":
    main()
