from contextlib import contextmanager
from types import SimpleNamespace

import torch
import pytest
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

from pgot.train.pgot_trainer import (
    PGOTTrainer,
    _enable_fsdp_sync_each_microbatch,
    _fsdp_full_state_dict,
    _install_token_row_gradient_mask,
)


def test_fsdp_microbatch_sync_skips_only_fsdp_no_sync():
    calls = []

    @contextmanager
    def original_no_sync(model):
        calls.append(model)
        yield

    accelerator = SimpleNamespace(no_sync=original_no_sync, sync_gradients=False)
    _enable_fsdp_sync_each_microbatch(accelerator)
    # No CUDA/process group is needed: the compatibility hook only checks type.
    fsdp = object.__new__(FSDP)
    with accelerator.no_sync(fsdp):
        assert accelerator.sync_gradients is False
    assert not calls
    ordinary = torch.nn.Linear(2, 2)
    with accelerator.no_sync(ordinary):
        pass
    assert calls == [ordinary]


@pytest.mark.parametrize("should_save", [True, False])
def test_manual_fsdp_save_gathers_on_all_ranks_and_writes_only_main(monkeypatch, should_save):
    trainer = object.__new__(PGOTTrainer)
    trainer.model = object.__new__(FSDP)
    trainer.args = SimpleNamespace(should_save=should_save, output_dir="checkpoint",
                                   push_to_hub=False)
    gathered, saved = [], []
    state_dict = {"weight": torch.ones(2)}

    def collect(model):
        gathered.append(model)
        return state_dict

    monkeypatch.setattr("pgot.train.pgot_trainer._fsdp_full_state_dict", collect)
    trainer._save = lambda path, state_dict: saved.append((path, state_dict))
    trainer.save_model("checkpoint-staging", _internal_call=True)
    assert gathered == [trainer.model]
    assert saved == ([("checkpoint-staging", state_dict)] if should_save else [])


def run_distributed_accumulation_probe():
    """Run with torchrun to compare two accumulated updates to full-batch SGD."""
    import copy
    import os

    from accelerate import Accelerator

    microbatch = int(os.environ.get("PGOT_PROBE_MICROBATCH", "1"))
    assert microbatch in (1, 2, 4, 8)
    accumulation = 8 // microbatch
    accelerator = Accelerator(gradient_accumulation_steps=accumulation)
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    torch.manual_seed(123)
    reference = torch.nn.Sequential(torch.nn.Linear(8, 16), torch.nn.GELU(),
                                    torch.nn.Linear(16, 4))
    reference[0].bias.requires_grad_(False)
    model = copy.deepcopy(reference)
    from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
        CheckpointImpl, checkpoint_wrapper,
    )
    for index in (0, 2):
        model[index] = FSDP(
            checkpoint_wrapper(model[index], checkpoint_impl=CheckpointImpl.NO_REENTRANT),
            device_id=accelerator.device, use_orig_params=True,
        )
    model = FSDP(model, device_id=accelerator.device, use_orig_params=True)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.03)
    reference_optimizer = torch.optim.SGD(reference.parameters(), lr=0.03)
    _enable_fsdp_sync_each_microbatch(accelerator)
    world_size = accelerator.num_processes
    for step in range(2):
        torch.manual_seed(700 + step)
        inputs = torch.randn(world_size * 8, 8)
        targets = torch.randn(world_size * 8, 4)
        for microstep in range(accumulation):
            index = (microstep * world_size + accelerator.process_index) * microbatch
            with accelerator.accumulate(model):
                loss = (model(inputs[index:index + microbatch].to(accelerator.device)) -
                        targets[index:index + microbatch].to(accelerator.device)).square().mean()
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    optimizer.step()
                    optimizer.zero_grad()
        (reference(inputs) - targets).square().mean().backward()
        reference_optimizer.step()
        reference_optimizer.zero_grad()
        with FSDP.summon_full_params(model):
            for actual, expected in zip(model.parameters(), reference.parameters()):
                torch.testing.assert_close(actual.cpu(), expected,
                                           rtol=1e-5, atol=1e-6)
    state_dict = _fsdp_full_state_dict(model)
    if accelerator.is_main_process:
        assert set(state_dict) == set(reference.state_dict())
        for name, tensor in reference.state_dict().items():
            assert state_dict[name].device.type == "cpu"
            torch.testing.assert_close(state_dict[name], tensor,
                                       rtol=1e-5, atol=1e-6)
        print(f"FSDP accumulation probe passed: batch={microbatch}, accumulation={accumulation}; "
              "two updates match full-batch SGD.")
        print("FSDP checkpoint probe passed: rank-0 CPU state dict matches full model.")
    else:
        assert not state_dict
    if os.environ.get("PGOT_PROBE_TOKEN_MASK") == "1":
        run_token_mask_probe(accelerator)
    torch.distributed.destroy_process_group()


def run_token_mask_probe(accelerator):
    """Verify live-view masks before clipping, with accumulation and AdamW."""
    import json
    import os
    from pathlib import Path

    class TinyCaptionModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = torch.nn.Embedding(8, 4)
            self.head = torch.nn.Linear(4, 8, bias=False)

        def forward(self, ids):
            return self.head(self.embed(ids))

    payload = {}
    microbatch = int(os.environ.get("PGOT_PROBE_MICROBATCH", "1"))
    accumulation = 8 // microbatch
    for optimizer_name in ("SGD", "AdamW"):
        torch.manual_seed(812)
        model = TinyCaptionModel()
        before = {name: value.detach().clone() for name, value in model.state_dict().items()}
        for module in (model.embed, model.head):
            _install_token_row_gradient_mask(module, [6, 7])
        wrapped = FSDP(model, device_id=accelerator.device, use_orig_params=True)
        reference = TinyCaptionModel()
        reference.load_state_dict(before)
        # Independent legacy reference: original-Parameter hooks work in plain PyTorch.
        keep = torch.zeros(8, 1)
        keep[6:] = 1
        for parameter in reference.parameters():
            parameter.register_hook(lambda g: g * keep)
        cls = torch.optim.SGD if optimizer_name == "SGD" else torch.optim.AdamW
        options = {"lr": 0.1, "weight_decay": 0 if optimizer_name == "SGD" else 0.01}
        optimizer = cls(wrapped.parameters(), **options)
        reference_optimizer = cls(reference.parameters(), **options)
        for step in range(2):
            ids = (torch.arange(accelerator.num_processes * 8) + step) % 8
            labels = (7 - ids) % 8
            for microstep in range(accumulation):
                index = (microstep * accelerator.num_processes + accelerator.process_index) * microbatch
                with accelerator.accumulate(wrapped):
                    loss = torch.nn.functional.cross_entropy(
                        wrapped(ids[index:index + microbatch].to(accelerator.device)),
                        labels[index:index + microbatch].to(accelerator.device))
                    accelerator.backward(loss)
                    if accelerator.sync_gradients:
                        norm = wrapped.clip_grad_norm_(0.1)
                        optimizer.step()
                        optimizer.zero_grad()
            torch.nn.functional.cross_entropy(reference(ids), labels).backward()
            reference_norm = torch.nn.utils.clip_grad_norm_(reference.parameters(), 0.1)
            torch.testing.assert_close(norm.cpu(), reference_norm, rtol=1e-5, atol=1e-6)
            reference_optimizer.step()
            reference_optimizer.zero_grad()
        after = _fsdp_full_state_dict(wrapped)
        calls = {"embed": model.embed._pgot_token_row_mask_audit,
                 "head": model.head._pgot_token_row_mask_audit}
        calls_by_rank = [None] * accelerator.num_processes
        torch.distributed.all_gather_object(calls_by_rank, calls)
        for rank_calls in calls_by_rank:
            for stats in rank_calls.values():
                assert stats["hook_calls"] > 0
                assert not stats["old_gradient_nonzero_any_sampled"]
        if accelerator.is_main_process:
            for name, value in reference.state_dict().items():
                torch.testing.assert_close(after[name], value, rtol=1e-5, atol=1e-6)
                if optimizer_name == "SGD":
                    assert torch.equal(after[name][:6], before[name][:6])
                else:
                    torch.testing.assert_close(after[name][:6], before[name][:6] * (1 - 0.1 * 0.01) ** 2)
                assert torch.count_nonzero(after[name][6:] - before[name][6:])
            payload[optimizer_name] = {"passed": True, "hook_calls_by_rank": calls_by_rank,
                                      "old_rows": "unchanged" if optimizer_name == "SGD" else "weight-decay-only",
                                      "updates": 2, "clipped_norm_matches_reference": True}
    if accelerator.is_main_process:
        Path(os.environ["PGOT_TOKEN_MASK_RESULT"]).write_text(json.dumps(payload, indent=2))
        print("Token-row mask probe passed:", json.dumps(payload), flush=True)


if __name__ == "__main__":
    run_distributed_accumulation_probe()
