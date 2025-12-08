"""
Distributed utilities for MicroVLM-E.
"""

import os
import datetime
import functools
import torch
import torch.distributed as dist


def setup_for_distributed(is_master: bool):
    """
    Disable printing when not in master process.
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop("force", False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def is_dist_avail_and_initialized():
    """Check if distributed training is available and initialized."""
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    """Get world size for distributed training."""
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    """Get rank for distributed training."""
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    """Check if current process is main process."""
    return get_rank() == 0


def init_distributed_mode(cfg):
    """Initialize distributed training mode."""
    if hasattr(cfg, 'distributed') and not cfg.distributed:
        print("Not using distributed mode")
        cfg.distributed = False
        return

    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        cfg.rank = int(os.environ["RANK"])
        cfg.world_size = int(os.environ["WORLD_SIZE"])
        cfg.gpu = int(os.environ.get("LOCAL_RANK", 0))
    elif "SLURM_PROCID" in os.environ:
        cfg.rank = int(os.environ["SLURM_PROCID"])
        cfg.gpu = cfg.rank % torch.cuda.device_count()
        cfg.world_size = int(os.environ.get("SLURM_NTASKS", 1))
    else:
        print("Not using distributed mode")
        cfg.distributed = False
        return

    cfg.distributed = True

    torch.cuda.set_device(cfg.gpu)
    cfg.dist_backend = "nccl"

    print(f"| distributed init (rank {cfg.rank}): {cfg.dist_url}", flush=True)

    dist.init_process_group(
        backend=cfg.dist_backend,
        init_method=cfg.dist_url,
        world_size=cfg.world_size,
        rank=cfg.rank,
        timeout=datetime.timedelta(minutes=60),
    )
    dist.barrier()
    setup_for_distributed(cfg.rank == 0)


def all_gather(data):
    """
    Gather data from all processes.
    """
    world_size = get_world_size()
    if world_size == 1:
        return [data]

    # Serialize data
    buffer = torch.ByteTensor(torch.ByteStorage.from_buffer(data.numpy().tobytes()))

    # Gather sizes
    local_size = torch.tensor([buffer.numel()], device="cuda")
    size_list = [torch.tensor([0], device="cuda") for _ in range(world_size)]
    dist.all_gather(size_list, local_size)
    size_list = [int(size.item()) for size in size_list]
    max_size = max(size_list)

    # Pad and gather tensors
    tensor_list = [torch.empty((max_size,), dtype=torch.uint8, device="cuda") for _ in range(world_size)]

    if local_size.item() != max_size:
        padding = torch.empty(size=(max_size - local_size.item(),), dtype=torch.uint8, device="cuda")
        buffer = torch.cat((buffer.cuda(), padding), dim=0)
    else:
        buffer = buffer.cuda()

    dist.all_gather(tensor_list, buffer)

    # Deserialize
    data_list = []
    for size, tensor in zip(size_list, tensor_list):
        buffer = tensor[:size].cpu().numpy().tobytes()
        data_list.append(torch.from_numpy(torch.ByteStorage.from_buffer(buffer).numpy()))

    return data_list


@functools.lru_cache()
def _get_global_gloo_group():
    """Get global gloo group."""
    if dist.get_backend() == "nccl":
        return dist.new_group(backend="gloo")
    return dist.group.WORLD


def barrier():
    """Synchronize all processes."""
    if is_dist_avail_and_initialized():
        dist.barrier()


def reduce_mean(tensor):
    """Reduce tensor across all processes and return mean."""
    if not is_dist_avail_and_initialized():
        return tensor

    tensor = tensor.clone()
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    tensor /= get_world_size()
    return tensor

