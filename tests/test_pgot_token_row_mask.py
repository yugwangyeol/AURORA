import pytest
import torch

from pgot.train.pgot_trainer import _install_token_row_gradient_mask


@pytest.mark.parametrize("kind", ["embedding", "linear"])
def test_token_mask_repeated_forwards_and_replaced_views(kind):
    module = torch.nn.Embedding(8, 4) if kind == "embedding" else torch.nn.Linear(4, 8, bias=False)
    _install_token_row_gradient_mask(module, [6, 7])
    inputs = torch.arange(8) if kind == "embedding" else torch.ones(8, 4)
    for replace in (False, True):
        if replace:
            # FSDP's forward uses an autograd view in place of the Parameter.
            original = torch.nn.Parameter(module.weight.detach().clone())
            del module._parameters["weight"]
            module._parameters["weight"] = original.view_as(original)
        else:
            original = module.weight
        original.grad = None
        with torch.no_grad():
            module(inputs)
        (module(inputs).sum() + module(inputs).sum()).backward()
        assert not torch.count_nonzero(original.grad[:6])
        assert torch.count_nonzero(original.grad[6:])
    assert module._pgot_token_row_mask_audit["hook_calls"] == 2


def test_tied_embedding_head_mask():
    embed = torch.nn.Embedding(8, 4)
    head = torch.nn.Linear(4, 8, bias=False)
    head.weight = embed.weight
    for module in (embed, head):
        _install_token_row_gradient_mask(module, [6, 7])
    head(embed(torch.arange(8))).sum().backward()
    assert not torch.count_nonzero(embed.weight.grad[:6])
    assert torch.count_nonzero(embed.weight.grad[6:])


def test_token_mask_invalid_ids():
    with pytest.raises(ValueError):
        _install_token_row_gradient_mask(torch.nn.Embedding(8, 4), [8])
