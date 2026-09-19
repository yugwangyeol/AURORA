"""Small FP32 collective diagnostic; no model or optimizer is loaded."""

import json
import os
from pathlib import Path
import statistics
import time

import torch
import torch.distributed as dist


def main():
    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl")
    world = dist.get_world_size()
    results = []
    for shard_mib in (64, 256):
        elements = shard_mib * 1024**2 // 4
        shard = torch.full((elements,), float(rank + 1), device="cuda")
        full = torch.empty((elements * world,), device="cuda")
        reduced = torch.empty_like(shard)
        for name, operation in (
            ("all_gather", lambda: dist.all_gather_into_tensor(full, shard)),
            ("reduce_scatter", lambda: dist.reduce_scatter_tensor(reduced, full)),
        ):
            timings = []
            for iteration in range(9):
                dist.barrier()
                torch.cuda.synchronize()
                start = time.perf_counter()
                operation()
                torch.cuda.synchronize()
                if iteration >= 3:
                    timings.append(time.perf_counter() - start)
            if name == "all_gather":
                for owner in range(world):
                    assert full[owner * elements].item() == owner + 1
            else:
                assert reduced[0].item() == world * (rank + 1)
            results.append({"operation": name, "shard_mib": shard_mib,
                            "full_mib": shard_mib * world,
                            "median_seconds": statistics.median(timings)})
        del shard, full, reduced
        torch.cuda.empty_cache()
    reports = [None] * world
    dist.all_gather_object(reports, {"rank": rank, "results": results})
    if rank == 0:
        payload = {"protocol": os.environ.get("NCCL_PROTO", "auto"), "world_size": world,
                   "reports": reports}
        path = Path(os.environ["PGOT_COMM_RESULT"])
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2))
        print(json.dumps(payload, indent=2), flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
