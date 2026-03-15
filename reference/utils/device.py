"""
Device resolution and dtype helpers for ComfyUI-CorridorKey.

Design decisions:
  - MPS uses float32: float16 autocast on Metal causes NaN/inf in GreenFormer's
    dilated CNN refiner. Running fp32 is ~15% slower but numerically stable.
  - CUDA uses float16 with autocast: standard Tensor Core path.
  - get_autocast_ctx() returns a no-op context for MPS/CPU so call-sites don't
    need to branch on device type.
"""

import contextlib
import torch


def get_device() -> torch.device:
    """
    Resolve best available compute device.

    Prefers ComfyUI's model management so the node respects whatever device
    the user has configured (including --cpu fallback or multi-GPU selection).
    Falls back to manual detection if running outside ComfyUI.
    """
    try:
        import comfy.model_management
        return comfy.model_management.get_torch_device()
    except (ImportError, Exception):
        pass

    if torch.cuda.is_available():
        return torch.device("cuda")

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def get_dtype(device: torch.device) -> torch.dtype:
    """
    Select working precision for a device.

    CUDA  -> float16  (Tensor Cores, autocast-friendly)
    MPS   -> float32  (Metal float16 autocast unreliable for this workload)
    CPU   -> float32
    """
    if device.type == "cuda":
        return torch.float16
    return torch.float32


def get_autocast_ctx(device: torch.device):
    """
    Return an appropriate autocast context manager for the device.

    CUDA: torch.autocast float16 — standard optimised path.
    MPS:  contextlib.nullcontext — skip autocast entirely. Metal's float16
          path has edge-cases with GreenFormer's pos_embed interpolation and
          dilated residual blocks that cause NaN outputs. Full float32 is the
          safe and recommended path on Apple Silicon for this model.
    CPU:  contextlib.nullcontext.
    """
    if device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return contextlib.nullcontext()


def clear_cache(device: torch.device) -> None:
    """Free the device memory pool after inference."""
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps":
        torch.mps.empty_cache()


def get_total_memory_gb(device: torch.device):
    """
    Return total memory in GB for the device, or None if unknown.

    For MPS, Apple Silicon uses unified memory so we report total system RAM
    via psutil (if available). This is the relevant budget for MPS inference.
    """
    if device.type == "cuda":
        try:
            props = torch.cuda.get_device_properties(device)
            return props.total_memory / (1024 ** 3)
        except Exception:
            return None

    if device.type == "mps":
        try:
            import psutil
            return psutil.virtual_memory().total / (1024 ** 3)
        except ImportError:
            return None

    return None


def warn_if_low_memory(device: torch.device, required_gb: float = 24.0):
    """
    Return a warning string if memory is below required_gb, else None.

    CorridorKey needs ~22.7 GB at native 2048x2048 resolution. Callers should
    print this warning before loading the model so users know early.
    """
    avail = get_total_memory_gb(device)
    if avail is None:
        return None

    if avail < required_gb:
        label = "VRAM" if device.type == "cuda" else "unified RAM"
        return (
            f"[CorridorKey] WARNING: {avail:.1f} GB {label} detected. "
            f"CorridorKey requires ~{required_gb:.0f} GB at 2048x2048. "
            "You may hit OOM — consider reducing input resolution."
        )
    return None
