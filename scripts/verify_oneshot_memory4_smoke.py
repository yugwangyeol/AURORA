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
    parser.add_argument(
        "--rae-query-adapter", choices=["enabled", "disabled"], default="disabled"
    )
    parser.add_argument("--adapter-bottleneck", type=int, default=384)
    parser.add_argument(
        "--contrastive", choices=["enabled", "disabled"], default="disabled"
    )
    parser.add_argument("--contrastive-lambda", type=float, default=0.03)
    parser.add_argument("--contrastive-sampling-rate", type=float, default=0.5)
    parser.add_argument("--contrastive-warmup", type=int, default=200)
    parser.add_argument(
        "--memory-value-source", choices=["siglip", "dinov2"], default="siglip"
    )
    parser.add_argument("--memory-value-dim", type=int, default=1152)
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
            assert summary["one_shot_rae_query_adapter_enabled"] is (
                args.rae_query_adapter == "enabled"
            )
            assert summary["rae_query_adapter_bottleneck"] == (
                args.adapter_bottleneck
                if args.rae_query_adapter == "enabled"
                else 0
            )
            assert summary["one_shot_memory_contrastive_enabled"] is (
                args.contrastive == "enabled"
            )
            assert math.isclose(
                summary["one_shot_memory_contrastive_lambda"],
                args.contrastive_lambda if args.contrastive == "enabled" else 0.0,
            )
            assert math.isclose(
                summary["one_shot_memory_contrastive_sampling_rate"],
                args.contrastive_sampling_rate,
            )
            assert summary["one_shot_memory_contrastive_warmup_steps"] == (
                args.contrastive_warmup if args.contrastive == "enabled" else 0
            )
            assert summary["one_shot_memory_value_source"] == args.memory_value_source
            assert summary["one_shot_memory_value_dim"] == args.memory_value_dim
            expected_value_name = (
                "DINOv2" if args.memory_value_source == "dinov2" else "raw SigLIP"
            )
            assert expected_value_name in summary["visual_memory_value_source"]
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
    assert cfg["pgot_one_shot_rae_query_adapter_enable"] is (
        args.rae_query_adapter == "enabled"
    )
    assert cfg["pgot_one_shot_rae_query_adapter_bottleneck"] == (
        args.adapter_bottleneck
    )
    assert cfg["pgot_one_shot_memory_contrastive_enable"] is (
        args.contrastive == "enabled"
    )
    assert math.isclose(
        cfg["pgot_one_shot_memory_contrastive_target_weight"],
        args.contrastive_lambda if args.contrastive == "enabled" else 0.0,
    )
    assert math.isclose(
        cfg["pgot_contrastive_sampling_rate"], args.contrastive_sampling_rate
    )
    assert cfg["pgot_one_shot_memory_contrastive_warmup_steps"] == (
        args.contrastive_warmup if args.contrastive == "enabled" else 0
    )
    assert cfg["pgot_one_shot_memory_value_source"] == args.memory_value_source
    assert cfg["pgot_one_shot_memory_value_dim"] == args.memory_value_dim
    if args.memory_value_source == "dinov2":
        towers = cfg["mm_vision_tower_aux_list"]
        assert len(towers) == 3 and "dinov2" in towers[2]
    state = json.loads((ckpt / "trainer_state.json").read_text())
    assert state["global_step"] == args.steps
    histories = state["log_history"]
    for key in ("loss_recon", "loss_e8_owner", "loss_e8_reader", "memory_write_entropy",
                "memory_object_pair_cosine", "memory_reader_entropy", "eval_loss",
                "eval_loss_recon", "eval_memory_reader_entropy",
                "one_shot_direct_rae_query_enabled", "mllm_rae_query_tokens",
                "one_shot_rae_query_adapter_enabled",
                "rae_query_adapter_bottleneck", "rae_query_adapter_delta_rms"):
        values = [row[key] for row in histories if key in row]
        assert values and all(math.isfinite(x) for x in values), (key, values)
    value_source_values = [
        row["one_shot_memory_value_dinov2_enabled"]
        for row in histories
        if "one_shot_memory_value_dinov2_enabled" in row
    ]
    expected_dino = 1.0 if args.memory_value_source == "dinov2" else 0.0
    assert value_source_values and all(x == expected_dino for x in value_source_values)
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
    expected_adapter = 1.0 if args.rae_query_adapter == "enabled" else 0.0
    assert all(
        row.get("one_shot_rae_query_adapter_enabled", expected_adapter)
        == expected_adapter
        for row in histories
    )
    if args.contrastive == "enabled":
        contrastive_keys = (
            "loss_contrastive",
            "loss_recon_mixed",
            "loss_recon_objective",
            "contrastive_lambda_effective",
            "contrastive_error_gap",
            "contrastive_mixed_object_fraction",
            "contrastive_timestep",
            "contrastive_decoder_stopgrad",
            "one_shot_memory_contrastive_enabled",
            "one_shot_memory_contrastive_active",
        )
        for key in contrastive_keys:
            values = [row[key] for row in histories if key in row]
            assert values and all(math.isfinite(x) for x in values), (key, values)
        active_rows = [
            row
            for row in histories
            if row.get("one_shot_memory_contrastive_active") == 1.0
        ]
        assert active_rows, "CODA mixed branch never became active"
        for row in active_rows:
            assert math.isclose(
                row["contrastive_lambda_effective"],
                args.contrastive_lambda,
                rel_tol=0.0,
                abs_tol=1e-8,
            )
            assert row["contrastive_mixed_object_fraction"] > 0.0
            assert row["contrastive_decoder_stopgrad"] == 1.0
            assert row["loss_contrastive"] < 0.0
            expected = row["loss_recon"] + (
                args.contrastive_lambda * row["loss_contrastive"]
            )
            assert math.isclose(
                row["loss_recon_objective"], expected, rel_tol=1e-5, abs_tol=1e-5
            ), row
    banned = [
        "loss_e8_causal", "e8_write_gate_mean",
        "one_shot_hard_outside_mass", "e9_gru", "e12_centroid",
        "dit_soft_routing", "latent_distill", "owner_gradient_scale",
        "memory_object_allocation", "memory_register_allocation",
    ]
    if args.contrastive != "enabled":
        banned.extend(("loss_contrastive", "one_shot_memory_contrastive"))
    for row in histories:
        assert not any(any(b in key for b in banned) for key in row), row
    from safetensors import safe_open
    weights = {}
    for shard in ckpt.glob("*.safetensors"):
        with safe_open(shard, framework="pt", device="cpu") as handle:
            for key in handle.keys():
                if key.startswith(
                    (
                        "pgot_e8_writer.",
                        "pgot_e8_reader.",
                        "pgot_rae_query_adapter.",
                    )
                ):
                    weights[key] = handle.get_tensor(key)
    ids = weights["pgot_e8_writer.memory_id_embeddings"]
    assert weights["pgot_e8_writer.raw_value_norm.weight"].shape == (
        args.memory_value_dim,
    )
    assert weights["pgot_e8_writer.raw_value.weight"].shape[1] == args.memory_value_dim
    expected_id_count = max(args.object_memories, args.register_memories)
    assert ids.shape[0] == expected_id_count and ids[:args.object_memories].std().item() > 0.01
    assert not any("gate" in key or "memory_to_query" in key for key in weights)
    if args.mode == "content":
        assert "pgot_e8_reader.content_key.weight" in weights
    else:
        assert "pgot_e8_reader.memory_key_embeddings" in weights
    if args.rae_query_adapter == "enabled":
        down = weights["pgot_rae_query_adapter.down.weight"]
        up = weights["pgot_rae_query_adapter.up.weight"]
        assert down.shape == (args.adapter_bottleneck, ids.shape[1])
        assert up.shape == (ids.shape[1], args.adapter_bottleneck)
        assert weights["pgot_rae_query_adapter.norm.weight"].shape == (ids.shape[1],)
        assert up.abs().max().item() > 0, "RAE query adapter did not update"
    else:
        assert not any(key.startswith("pgot_rae_query_adapter.") for key in weights)
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
