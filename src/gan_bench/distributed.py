from __future__ import annotations

import os

import torch
import torch.distributed as dist


def distributed_info() -> tuple[int, int, int]:
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    return rank, local_rank, world_size


def init_distributed(device_type: str) -> tuple[int, int, int]:
    rank, local_rank, world_size = distributed_info()
    if world_size > 1 and not dist.is_initialized():
        backend = "nccl" if device_type == "cuda" else "gloo"
        dist.init_process_group(backend=backend)
    return rank, local_rank, world_size


def barrier() -> None:
    if dist.is_initialized():
        dist.barrier()


def mean_across_ranks(value: float, device: torch.device) -> float:
    tensor = torch.tensor(float(value), device=device)
    if dist.is_initialized():
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        tensor /= dist.get_world_size()
    return float(tensor.item())


def close_distributed() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()
