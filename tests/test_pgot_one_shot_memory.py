import io

import pytest
import torch

from pgot.model.visual_memory import (
    PGOTDirectRAEQueryAdapter,
    PGOTOneShotMemoryReader,
    PGOTOneShotMemoryWriter,
)
from pgot.model.pgot_qwen2 import PGOTQwen2ForCausalLM


def test_direct_rae_query_adapter_starts_as_exact_identity_and_trains():
    torch.manual_seed(7)
    adapter = PGOTDirectRAEQueryAdapter(dim=8, bottleneck_dim=2)
    queries = torch.randn(2, 5, 8)
    torch.testing.assert_close(adapter(queries), queries, rtol=0, atol=0)

    loss = (adapter(queries) - torch.randn_like(queries)).square().mean()
    loss.backward()
    assert adapter.up.weight.grad is not None
    assert adapter.up.weight.grad.abs().sum() > 0
    with torch.no_grad():
        adapter.up.weight.add_(-0.1 * adapter.up.weight.grad)
    assert not torch.equal(adapter(queries), queries)


def test_coda_mixer_swaps_object_groups_and_freezes_register_inputs():
    torch.manual_seed(19)
    semantic = torch.arange(2 * 4 * 3, dtype=torch.float32).reshape(2, 4, 3)
    semantic.requires_grad_()
    memory = torch.arange(2 * 4 * 2 * 3, dtype=torch.float32).reshape(2, 4, 2, 3)
    memory.requires_grad_()
    object_valid = torch.ones(2, 2, dtype=torch.bool)
    memory_valid = torch.ones(2, 4, 2, dtype=torch.bool)

    mixed = PGOTQwen2ForCausalLM._pgot_one_shot_mix_object_memory_slots(
        semantic_slots=semantic,
        visual_memory=memory,
        object_valid=object_valid,
        memory_valid=memory_valid,
        sampling_rate=1.0,
    )
    partner = torch.tensor([1, 0])
    # Donor owners may be permuted, but each semantic owner and its complete
    # memory group must come from the paired image.
    for batch_idx in range(2):
        expected_semantic = {
            tuple(row.tolist()) for row in semantic[partner[batch_idx], :2]
        }
        expected_memory = {
            tuple(row.flatten().tolist()) for row in memory[partner[batch_idx], :2]
        }
        assert {
            tuple(row.tolist()) for row in mixed["semantic_slots"][batch_idx, :2]
        } == expected_semantic
        assert {
            tuple(row.flatten().tolist())
            for row in mixed["visual_memory"][batch_idx, :2]
        } == expected_memory
    torch.testing.assert_close(mixed["semantic_slots"][:, 2:], semantic[:, 2:])
    torch.testing.assert_close(mixed["visual_memory"][:, 2:], memory[:, 2:])
    assert mixed["swap_mask"].all()
    assert mixed["swapped_object_fraction"].item() == 1.0

    (mixed["semantic_slots"].sum() + mixed["visual_memory"].sum()).backward()
    assert semantic.grad[:, :2].abs().sum() > 0
    assert memory.grad[:, :2].abs().sum() > 0
    torch.testing.assert_close(semantic.grad[:, 2:], torch.zeros_like(semantic.grad[:, 2:]))
    torch.testing.assert_close(memory.grad[:, 2:], torch.zeros_like(memory.grad[:, 2:]))


def make_case(mode="memory_content", softmax_axis="patch", use_owner_prior=True):
    torch.manual_seed(43)
    writer = PGOTOneShotMemoryWriter(dim=8, raw_value_dim=6,
                                    object_memories_per_owner=2, register_memories_per_owner=4,
                                    softmax_axis=softmax_axis,
                                    use_owner_prior=use_owner_prior)
    reader = PGOTOneShotMemoryReader(dim=8, num_heads=2, memories_per_owner=4, readout_mode=mode)
    inputs = dict(semantic_slots=torch.randn(2, 3, 8), image_states=torch.randn(2, 9, 8),
                  raw_value_states=torch.randn(2, 9, 6), owner_probs=torch.softmax(torch.randn(2, 3, 9), 1),
                  slot_valid=torch.tensor([[True, False, True], [True, True, True]]), object_count=2)
    queries = torch.randn(2, 5, 8)
    return writer, reader, inputs, queries


def read(reader, inputs, queries, memory):
    return reader(rae_queries=queries, semantic_slots=inputs["semantic_slots"],
                  slot_valid=inputs["slot_valid"], visual_memory=memory["visual_memory"],
                  memory_valid=memory["memory_valid"])


def test_patch_normalization_heterogeneous_capacity_and_gradients():
    writer, reader, inputs, queries = make_case()
    inputs["owner_probs"].requires_grad_()
    inputs["semantic_slots"].requires_grad_()
    memory = writer(**inputs)
    valid = memory["memory_valid"]
    assert valid.sum().item() == 14  # (2 + 0 + 4) + (2 + 2 + 4)
    torch.testing.assert_close(memory["write_weights"].sum(-1), valid.float())
    assert torch.count_nonzero(memory["visual_memory"][~valid]) == 0
    out = read(reader, inputs, queries, memory)
    torch.testing.assert_close(out["reader_memory_attention"].sum(-1), torch.ones(2, 5))
    assert torch.count_nonzero(out["reader_memory_attention"].reshape(2, 5, 3, 4)[0, :, 1]) == 0
    loss = (out["condition_hidden"] - torch.randn_like(out["condition_hidden"])).square().mean()
    loss.backward()
    assert inputs["owner_probs"].grad is None  # only routing is detached
    assert inputs["semantic_slots"].grad.abs().sum() > 0
    for module in (writer, reader):
        for name, p in module.named_parameters():
            assert p.grad is not None and torch.isfinite(p.grad).all(), name
    assert writer.memory_id_embeddings.grad.abs().sum() > 0
    assert reader.content_key.weight.grad.abs().sum() > 0


def test_semantic_query_changes_patch_selection_with_fixed_ownership():
    writer, _, inputs, _ = make_case()
    before = writer(**inputs)["write_weights"]
    inputs["semantic_slots"] = torch.randn_like(inputs["semantic_slots"])
    after = writer(**inputs)["write_weights"]
    assert (before - after).abs().max() > 0.01


def test_no_owner_prior_makes_writer_independent_of_owner_map():
    writer, _, inputs, _ = make_case(use_owner_prior=False)
    first = writer(**inputs)
    changed = dict(inputs)
    changed["owner_probs"] = torch.softmax(
        torch.randn_like(inputs["owner_probs"]), dim=1
    )
    second = writer(**changed)
    torch.testing.assert_close(first["write_weights"], second["write_weights"])
    torch.testing.assert_close(first["visual_memory"], second["visual_memory"])


def test_no_owner_prior_blocks_reconstruction_gradient_to_owner_map():
    writer, _, inputs, _ = make_case(use_owner_prior=False)
    writer.detach_owner_routing = False
    owner = inputs["owner_probs"].detach().requires_grad_(True)
    result = writer(**{**inputs, "owner_probs": owner})
    result["visual_memory"].square().sum().backward()
    assert owner.grad is None


def test_memory_axis_competitively_allocates_each_patch_then_pools():
    writer, _, inputs, _ = make_case(softmax_axis="memory")
    result = writer(**inputs)
    valid = result["memory_valid"]
    weights = result["write_weights"]
    mass = result["memory_allocation_mass"]

    torch.testing.assert_close(weights.sum(-1), valid.float())
    assert torch.count_nonzero(weights[~valid]) == 0
    assert mass is not None and torch.count_nonzero(mass[~valid]) == 0

    dtype = writer.query.weight.dtype
    query = writer.query(writer.semantic_norm(inputs["semantic_slots"].to(dtype)))[:, :, None]
    query = query + writer.memory_id_embeddings[None, None]
    key = writer.key(writer.image_norm(inputs["image_states"].to(dtype)))
    logits = torch.einsum("bsjd,bpd->bsjp", query.float(), key.float())
    logits = logits / (writer.dim ** 0.5) / writer.temperature
    routing = inputs["owner_probs"].float() * inputs["slot_valid"][..., None].float()
    routing = routing / routing.sum(1, keepdim=True).clamp_min(1e-8)
    logits = logits + routing.clamp_min(1e-8).log()[:, :, None]
    allocation = torch.softmax(logits.masked_fill(~valid[..., None], -1e4), dim=2)
    allocation = allocation * valid[..., None].float()
    allocation = allocation / allocation.sum(2, keepdim=True).clamp_min(1e-8)
    joint = allocation * routing[:, :, None]
    expected = joint / joint.sum(-1, keepdim=True).clamp_min(1e-8)
    expected = expected * valid[..., None].float()
    torch.testing.assert_close(weights, expected)


def test_owner_gradient_scale_changes_backward_only():
    writer, _, inputs, _ = make_case()
    writer.detach_owner_routing = False
    owner = inputs["owner_probs"].detach().requires_grad_(True)
    inputs["owner_probs"] = owner

    stopped = writer(**inputs, owner_gradient_scale=0.0)
    stopped["visual_memory"].square().sum().backward()
    assert owner.grad is not None
    torch.testing.assert_close(owner.grad, torch.zeros_like(owner.grad))

    owner.grad = None
    live = writer(**inputs, owner_gradient_scale=1.0)
    torch.testing.assert_close(live["visual_memory"], stopped["visual_memory"])
    live["visual_memory"].square().sum().backward()
    assert owner.grad is not None and owner.grad.abs().sum() > 0


@pytest.mark.parametrize("mode", ["memory_content", "memory_id"])
def test_stored_memory_roundtrip_and_content_permutation(mode):
    writer, reader, inputs, queries = make_case(mode)
    memory = {k: v.detach().clone() for k, v in writer(**inputs).items() if k != "write_weights"}
    baseline = read(reader, inputs, queries, memory)
    # Decode with only saved owner states + memory, independent of raw inputs.
    buffer = io.BytesIO()
    torch.save(dict(reader=reader.state_dict(), memory=memory,
                    semantic_slots=inputs["semantic_slots"], slot_valid=inputs["slot_valid"]), buffer)
    buffer.seek(0)
    saved = torch.load(buffer, weights_only=True)
    _, restored, _, _ = make_case(mode)
    restored.load_state_dict(saved["reader"])
    actual = read(restored, saved, queries, saved["memory"])
    torch.testing.assert_close(actual["condition_hidden"], baseline["condition_hidden"])
    permuted = {k: v.clone() for k, v in memory.items()}
    permuted["visual_memory"][:, 2] = permuted["visual_memory"][:, 2].flip(1)
    changed = read(reader, inputs, queries, permuted)
    if mode == "memory_content":
        # Content addressing does not depend on arbitrary within-owner IDs.
        torch.testing.assert_close(changed["condition_hidden"], baseline["condition_hidden"], atol=2e-6, rtol=1e-5)
    else:
        assert not torch.allclose(changed["condition_hidden"], baseline["condition_hidden"])


def test_content_keys_respond_to_memory_but_owner_route_does_not():
    writer, reader, inputs, queries = make_case()
    memory = writer(**inputs)
    before = read(reader, inputs, queries, memory)
    memory["visual_memory"] = torch.randn_like(memory["visual_memory"])
    after = read(reader, inputs, queries, memory)
    torch.testing.assert_close(before["reader_attention_heads"], after["reader_attention_heads"])
    assert not torch.allclose(before["reader_inner_attention_heads"], after["reader_inner_attention_heads"])


def test_metric_profile_excludes_disabled_diagnostics_but_keeps_active_zeros():
    from pgot.train.pgot_trainer import PGOTTrainer
    trainer = object.__new__(PGOTTrainer)
    trainer._metric_profile = "one_shot_memory"
    for key in ("eval_loss_e8_causal", "e8_write_gate_mean",
                "one_shot_hard_outside_mass", "epoch", "total_flos"):
        assert not trainer._keep_metric(key)
    for key in ("loss", "eval_loss_recon", "e8_owner_fg_acc", "memory_reader_entropy",
                "loss_latent_distill", "latent_distill_weight_effective",
                "owner_gradient_scale", "loss_contrastive",
                "contrastive_decoder_stopgrad", "train_runtime"):
        assert trainer._keep_metric(key)
