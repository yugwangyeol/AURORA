"""Short ownership-only batch benchmark; never saves model weights."""
import gc
import json
import os
import time
from pathlib import Path

import torch


def benchmark(model, dataset, collator, args):
    if torch.distributed.get_world_size() != 1:
        raise ValueError("Capacity sweep runs on one GPU; validate selected batch with Trainer DDP afterward")
    candidates = [int(x) for x in os.environ["PGOT_TUNE_BATCH"].split(",")]
    # Stress padding with the longest available expressions, not the first
    # (usually short) captions from a single image.
    indices = sorted(range(len(dataset)), key=lambda i: len(dataset.samples[i]["caption"]), reverse=True)
    samples = [dataset[i] for i in indices[:max(candidates)]]
    if len(samples) < max(candidates):
        raise ValueError("Insufficient benchmark examples")
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=args.learning_rate, weight_decay=args.weight_decay)
    model.train()
    keys = ("images", "target_images", "caption_input_ids", "caption_attention_mask",
            "caption_labels", "ovt_positions_in_caption", "ovt_valid_mask", "ovt_is_thing",
            "gt_masks_per_ovt", "gt_rae_masks_per_ovt")
    rows = []
    def measure(size, checkpointing):
        optimizer.zero_grad(set_to_none=True)
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        batch = outputs = None
        try:
            batch = collator(samples[:size])
            batch = {key: batch[key].to(args.device) for key in keys}
            times, losses = [], []
            for repeat in range(4):
                torch.cuda.synchronize()
                start = time.perf_counter()
                outputs = model(**batch, pgot_contrastive_weight=0.0)
                if not torch.isfinite(outputs.loss):
                    raise RuntimeError("Nonfinite benchmark loss")
                losses.append(outputs.loss.item())
                outputs.loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(parameters, args.max_grad_norm, error_if_nonfinite=True)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                torch.cuda.synchronize()
                if repeat:
                    times.append(time.perf_counter() - start)
                del outputs
                outputs = None
            row = dict(batch=size, status="ok", seconds_per_microstep=sum(times)/len(times),
                       samples_per_second=size*len(times)/sum(times),
                       peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                       peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30,
                       total_gib=torch.cuda.get_device_properties(0).total_memory/2**30,
                       caption_tokens=batch["caption_input_ids"].shape[1], losses=losses)
        except torch.cuda.OutOfMemoryError:
            row = dict(batch=size, status="oom")
        finally:
            outputs = batch = None
            optimizer.zero_grad(set_to_none=True)
            gc.collect()
            torch.cuda.empty_cache()
        row["gradient_checkpointing"] = checkpointing
        return row

    for checkpointing in (False, True):
        if checkpointing:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        else:
            model.gradient_checkpointing_disable()
        for size in candidates:
            row = measure(size, checkpointing)
            rows.append(row)
            print("[RefCOCO batch benchmark] " + json.dumps(row), flush=True)
            Path(args.output_dir).mkdir(parents=True, exist_ok=True)
            Path(args.output_dir, "batch_benchmark.json").write_text(json.dumps(rows, indent=2))
            if row["status"] == "oom":
                break
    torch.distributed.destroy_process_group()
