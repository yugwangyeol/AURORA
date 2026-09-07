import torch

from pgot.model.visual_memory import PGOTOneShotOwnerReader


def _inputs():
    return {
        "rae_queries": torch.tensor(
            [[[2.0, -1.0, 1.0, -2.0], [-2.0, 1.0, -1.0, 2.0]]]
        ),
        "semantic_slots": torch.tensor(
            [[[2.0, -1.0, 1.0, -2.0], [-2.0, 1.0, -1.0, 2.0]]]
        ),
        "raw_patches": torch.tensor(
            [[
                [1.0, 0.0, 0.0, -1.0],
                [0.5, 1.0, -0.5, -1.0],
                [-1.0, 0.0, 0.0, 1.0],
                [-0.5, -1.0, 0.5, 1.0],
            ]]
        ),
        "owner_probs": torch.tensor(
            [[[0.9, 0.8, 0.1, 0.2], [0.1, 0.2, 0.9, 0.8]]]
        ),
        "slot_valid": torch.tensor([[True, True]]),
    }


def test_pooled_reader_is_exact_two_hop_attention():
    reader = PGOTOneShotOwnerReader(
        dim=4, raw_value_dim=4, num_heads=2, readout_mode="pooled"
    )
    result = reader(**_inputs())

    owner_attention = result["reader_attention_heads"]
    route = _inputs()["owner_probs"]
    route = route / route.sum(dim=-1, keepdim=True)
    expected = torch.einsum("bhqs,bsp->bhqp", owner_attention, route)
    expected = expected / expected.sum(dim=-1, keepdim=True)

    torch.testing.assert_close(
        result["reader_patch_attention_heads"], expected, rtol=1e-5, atol=1e-6
    )
    assert result["condition_hidden"].shape == (1, 2, 4)
    assert torch.isfinite(result["condition_hidden"]).all()


def test_owner_masked_reader_has_zero_cross_owner_access():
    reader = PGOTOneShotOwnerReader(
        dim=4, raw_value_dim=4, num_heads=2, readout_mode="owner_masked"
    )
    inputs = _inputs()
    result = reader(**inputs)

    selected_owner = result["reader_attention_heads"].argmax(dim=-1)
    patch_owner = inputs["owner_probs"].argmax(dim=1)
    allowed = selected_owner.unsqueeze(-1) == patch_owner[:, None, None]
    outside = result["reader_patch_attention_heads"] * (~allowed).float()
    torch.testing.assert_close(outside, torch.zeros_like(outside), rtol=0, atol=0)
    assert result["hard_outside_mass"].item() == 0.0

    # A value change outside query 0's selected owner cannot affect query 0.
    modified = {key: value.clone() for key, value in inputs.items()}
    owner0 = int(selected_owner[0, 0, 0].item())
    disallowed = patch_owner[0] != owner0
    modified["raw_patches"][0, disallowed] += 1000.0
    changed = reader(**modified)
    torch.testing.assert_close(
        changed["condition_hidden"][:, :1],
        result["condition_hidden"][:, :1],
        rtol=1e-5,
        atol=1e-5,
    )
