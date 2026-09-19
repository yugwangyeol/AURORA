import json
from types import SimpleNamespace

import pytest
import torch

from scripts.benchmark_pgot_7b_entry import BenchmarkCallback


def test_benchmark_excludes_warmup_and_profile_from_speed(tmp_path):
    callback = BenchmarkCallback()
    callback.directory = tmp_path
    callback.rank = 0
    callback.trainable_shard_numel = 10
    callback.trainable_dtypes = ["torch.float32"]
    callback.steps = [
        {"step": 1, "seconds": 90, "profiled": False},
        {"step": 2, "seconds": 80, "profiled": False},
        {"step": 3, "seconds": 40, "profiled": False},
        {"step": 4, "seconds": 42, "profiled": False},
        {"step": 5, "seconds": 200, "profiled": True},
    ]
    args = SimpleNamespace(per_device_train_batch_size=2,
                           gradient_accumulation_steps=4, world_size=4,
                           optim="adamw_torch", bf16=False, fp16=False, tf32=True)
    callback._write(args)
    report = json.loads((tmp_path / "benchmark.rank0.json").read_text())
    assert report["median_seconds_after_warmup"] == 41
    assert report["batch_per_device"] * report["accumulation"] * report["world_size"] == 32


@pytest.mark.parametrize("save_steps,eval_steps", [(3, 1000), (1000, 3)])
def test_benchmark_refuses_periodic_save_or_eval(save_steps, eval_steps):
    callback = BenchmarkCallback()
    args = SimpleNamespace(save_steps=save_steps, eval_steps=eval_steps)
    with pytest.raises(ValueError, match="no periodic save/eval"):
        callback.on_train_begin(args, SimpleNamespace(max_steps=3), None,
                                model=torch.nn.Linear(1, 1))
