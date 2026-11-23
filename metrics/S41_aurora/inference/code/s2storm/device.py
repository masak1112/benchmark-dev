"""Device helper: cuda (DSI), xpu (Aurora), or cpu."""
import os


def get_device(local_rank: int | None = None):
    import torch

    if local_rank is None:
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank % torch.cuda.device_count())
        return torch.device("cuda", local_rank % torch.cuda.device_count())
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        torch.xpu.set_device(local_rank % torch.xpu.device_count())
        return torch.device("xpu", local_rank % torch.xpu.device_count())
    return torch.device("cpu")


def backend(device) -> str:
    """torch.distributed backend for this device."""
    return {"cuda": "nccl", "xpu": "xccl"}.get(device.type, "gloo")


def synchronize(device):
    import torch

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "xpu":
        torch.xpu.synchronize(device)


def max_memory_gb(device) -> float:
    import torch

    if device.type == "cuda":
        return torch.cuda.max_memory_allocated(device) / 2**30
    if device.type == "xpu":
        return torch.xpu.max_memory_allocated(device) / 2**30
    return float("nan")
