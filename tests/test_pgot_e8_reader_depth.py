import torch

from pgot.model.visual_memory import PGOTE8TypedRAEReader


def _inputs(dim: int = 16):
    generator = torch.Generator().manual_seed(17)
    return {
        "rae_queries": torch.randn(2, 5, dim, generator=generator),
        "semantic_slots": torch.randn(2, 3, dim, generator=generator),
        "visual_memory": torch.randn(2, 3, 2, dim, generator=generator),
        "slot_valid": torch.tensor([[True, True, True], [True, False, True]]),
        "object_count": 2,
    }


def test_reader3_zero_output_initialization_preserves_reader1_condition():
    reader1 = PGOTE8TypedRAEReader(
        dim=16,
        num_heads=4,
        memories_per_owner=2,
        num_layers=1,
    )
    reader3 = PGOTE8TypedRAEReader(
        dim=16,
        num_heads=4,
        memories_per_owner=2,
        num_layers=3,
    )
    missing, unexpected = reader3.load_state_dict(reader1.state_dict(), strict=False)
    assert not unexpected
    assert missing
    for refinement in reader3.refinement_layers:
        refinement.reset_as_identity()

    inputs = _inputs()
    output1 = reader1(**inputs)
    output3 = reader3(**inputs)
    torch.testing.assert_close(
        output3["condition_hidden"], output1["condition_hidden"], rtol=0, atol=0
    )
    torch.testing.assert_close(
        output3["reader_attention"], output1["reader_attention"], rtol=0, atol=0
    )
    assert output3["reader_num_layers"].item() == 3
    assert output3["reader_attention"].shape == (2, 5, 6)


def test_reader3_new_residual_outputs_receive_gradients():
    reader = PGOTE8TypedRAEReader(
        dim=16,
        num_heads=4,
        memories_per_owner=2,
        num_layers=3,
    )
    output = reader(**_inputs())
    output["condition_hidden"].square().mean().backward()
    for refinement in reader.refinement_layers:
        assert refinement.output.weight.grad is not None
        assert torch.isfinite(refinement.output.weight.grad).all()
        assert refinement.ffn_out.weight.grad is not None
        assert torch.isfinite(refinement.ffn_out.weight.grad).all()
