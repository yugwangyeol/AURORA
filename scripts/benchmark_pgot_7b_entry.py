"""Opt-in entrypoint: run the unchanged training loop with per-rank timing.

No model/checkpoint is saved. Use the production shell wrapper with
TRAIN_ENTRYPOINT pointing here, MAX_STEPS <= 10, and SAVE/EVAL_STEPS > MAX_STEPS.
The final step can optionally be profiled, and is excluded from throughput.
"""

import json
import os
from functools import wraps
from pathlib import Path
import runpy
import statistics
import time

import torch
from transformers import TrainerCallback

from pgot.train.pgot_trainer import PGOTTrainer, _FSDPCompatibleAdamW
from torch.distributed.fsdp import BackwardPrefetch, FullyShardedDataParallel as FSDP


class BenchmarkCallback(TrainerCallback):
    def __init__(self, stress_info=None):
        self.steps = []
        self.profiler = None
        self.losses = []
        self.stress_info = stress_info
        self.prefetch_switch = int(os.environ.get("PGOT_BENCH_PREFETCH_STEP", "0"))
        self.stress_step = int(os.environ.get("PGOT_BENCH_STRESS_STEP", "0"))
        self.prefetch_mode = "none"

    def on_train_begin(self, args, state, control, **kwargs):
        if state.max_steps > 10 or args.save_steps <= state.max_steps or args.eval_steps <= state.max_steps:
            raise ValueError("Benchmark requires <=10 updates and no periodic save/eval")
        self.directory = Path(args.output_dir)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.rank = args.process_index
        self.profile_enabled = os.environ.get("PGOT_BENCH_PROFILE", "0") == "1"
        # Keep other ranks from launching logging collectives while Kineto
        # drains CUDA activities. An NCCL barrier here would itself launch
        # CUDA work, so use a separate CPU/Gloo process group.
        self.profile_group = torch.distributed.new_group(backend="gloo") if self.profile_enabled else None
        model = kwargs["model"]
        self.trainable_shard_numel = sum(p.numel() for p in model.parameters() if p.requires_grad)
        self.trainable_dtypes = sorted({str(p.dtype) for p in model.parameters() if p.requires_grad})

    def on_step_begin(self, args, state, control, **kwargs):
        torch.cuda.synchronize()
        if self.prefetch_switch and state.global_step + 1 == self.prefetch_switch:
            count = 0
            for module in kwargs["model"].modules():
                if isinstance(module, FSDP):
                    module.backward_prefetch = BackwardPrefetch.BACKWARD_POST
                    count += 1
            if not count:
                raise ValueError("Expected FSDP units for prefetch benchmark")
            self.prefetch_mode = "backward_post"
            print(f"[PGOT/Benchmark rank={self.rank}] enabled backward_post in {count} FSDP units", flush=True)
        torch.cuda.reset_peak_memory_stats()
        self.profiled_step = self.profile_enabled and state.global_step + 1 == state.max_steps
        if self.profiled_step:
            torch.distributed.barrier(group=self.profile_group)
        if self.profiled_step and self.rank == args.world_size - 1:
            self.profiler = torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
                record_shapes=False, profile_memory=False, with_stack=False,
            )
            self.profiler.__enter__()
        if self.profiled_step:
            torch.distributed.barrier(group=self.profile_group)
        self.started = time.perf_counter()

    def on_step_end(self, args, state, control, **kwargs):
        torch.cuda.synchronize()
        seconds = time.perf_counter() - self.started
        memory = torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory
        self.steps.append({
            "step": state.global_step, "seconds": seconds,
            "profiled": self.profiled_step,
            "prefetch": self.prefetch_mode,
            "stress": bool(self.stress_step and state.global_step >= self.stress_step),
            "peak_allocated_gib": torch.cuda.max_memory_allocated() / 1024**3,
            "peak_reserved_gib": torch.cuda.max_memory_reserved() / 1024**3,
            "device_total_gib": memory / 1024**3,
        })
        # Persist measurements even if a platform's CUPTI collector stalls.
        self._write(args)
        if self.profiled_step:
            torch.distributed.barrier(group=self.profile_group)
        if self.profiler is not None:
            self.profiler.__exit__(None, None, None)
            events = self.profiler.key_averages()
            (self.directory / f"profile_cpu.rank{self.rank}.txt").write_text(
                events.table(sort_by="self_cpu_time_total", row_limit=40)
            )
            (self.directory / f"profile_cuda.rank{self.rank}.txt").write_text(
                events.table(sort_by="self_device_time_total", row_limit=40)
            )
            self.profiler.export_chrome_trace(str(self.directory / f"profile.rank{self.rank}.json"))
            self.profiler = None
        if self.profiled_step:
            torch.distributed.barrier(group=self.profile_group)
        self._write(args)
        print(f"[PGOT/Benchmark rank={self.rank}] {self.steps[-1]}", flush=True)

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and "loss" in logs:
            self.losses.append(float(logs["loss"]))
            self._write(args)

    def _write(self, args):
        measured_steps = [s for s in self.steps if s["step"] > 2 and not s["profiled"]
                          and not s.get("stress", False) and s["step"] != self.prefetch_switch]
        measured = [s["seconds"] for s in measured_steps]
        by_mode = {}
        for step in measured_steps:
            by_mode.setdefault(step.get("prefetch", "none"), []).append(step["seconds"])
        report = {
            "rank": self.rank,
            "batch_per_device": args.per_device_train_batch_size,
            "accumulation": args.gradient_accumulation_steps,
            "world_size": args.world_size,
            "optimizer": str(args.optim),
            "bf16": args.bf16, "fp16": args.fp16, "tf32": args.tf32,
            "trainable_shard_numel": self.trainable_shard_numel,
            "trainable_dtypes": self.trainable_dtypes,
            "stress_info": self.stress_info,
            "steps": self.steps, "logged_losses": self.losses,
            "median_seconds_after_warmup": statistics.median(measured) if measured else None,
            "median_seconds_by_prefetch": {mode: statistics.median(values) for mode, values in by_mode.items()},
        }
        (self.directory / f"benchmark.rank{self.rank}.json").write_text(json.dumps(report, indent=2))


original_init = PGOTTrainer.__init__


def benchmark_init(self, *args, **kwargs):
    original_init(self, *args, **kwargs)
    if os.environ.get("PGOT_BENCH_SKIP_METRICS", "0") == "1":
        self._keep_metric = lambda key: False
    stress_info = None
    if int(os.environ.get("PGOT_BENCH_STRESS_STEP", "0")):
        dataset = self.train_dataset
        candidates = sorted(range(len(dataset.samples)),
                            key=lambda i: len(dataset.samples[i].get("caption", "")), reverse=True)[:128]
        scored = []
        for index in candidates:
            caption, object_count = dataset._build_caption_with_ovt(dataset.samples[index]["segments"])
            length = len(dataset.tokenizer.encode(caption, add_special_tokens=False))
            scored.append((length, object_count, index))
        scored.sort(reverse=True)
        batch_size = self.args.per_device_train_batch_size
        rank = self.args.process_index
        # Put the longest captions on the rank with the highest observed
        # optimizer-state load, while exercising distinct images on other ranks.
        offset = 0 if rank == self.args.world_size - 1 else (rank + 1) * batch_size
        selected = scored[offset:offset + batch_size]
        samples = [dataset[index] for _, _, index in selected]
        self._bench_stress_batch = self.data_collator(samples)
        stress_info = {
            "image_ids": [dataset.samples[index]["image_id"] for _, _, index in selected],
            "caption_tokens": [int(sample["caption_input_ids"].numel()) for sample in samples],
            "object_counts": [int(sample["n_objects"]) for sample in samples],
            "selection": "longest packed captions among 128 longest manifest captions",
        }
        original_training_step = self.training_step
        stress_step = int(os.environ["PGOT_BENCH_STRESS_STEP"])

        def stress_training_step(model, inputs, *step_args, **step_kwargs):
            if self.state.global_step + 1 >= stress_step:
                inputs = self._bench_stress_batch
            return original_training_step(model, inputs, *step_args, **step_kwargs)

        self.training_step = stress_training_step
    self.add_callback(BenchmarkCallback(stress_info))


def record_call(cls, name, label):
    original = getattr(cls, name)

    @wraps(original)
    def recorded(*args, **kwargs):
        with torch.profiler.record_function(label):
            return original(*args, **kwargs)

    setattr(cls, name, recorded)


if __name__ == "__main__":
    os.environ["PGOT_SKIP_FINAL_SAVE"] = "1"
    PGOTTrainer.__init__ = benchmark_init
    if os.environ.get("PGOT_BENCH_PROFILE", "0") == "1":
        from scale_rae.model.diffusion_loss.models.lightningDiT import LightningDiT

        record_call(PGOTTrainer, "compute_loss", "PGOT/forward_and_metrics")
        record_call(PGOTTrainer, "training_step", "PGOT/forward_and_backward")
        record_call(_FSDPCompatibleAdamW, "step", "PGOT/optimizer")
        record_call(LightningDiT, "forward", "PGOT/DiT_forward")
    runpy.run_path(str(Path(__file__).resolve().parents[1] / "train.py"), run_name="__main__")
