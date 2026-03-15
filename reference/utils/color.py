import torch
import numpy as np


def srgb_to_linear(x):
    """Convert sRGB to linear using piecewise transfer function.

    Works with both torch tensors and numpy arrays.
    """
    if isinstance(x, torch.Tensor):
        return torch.where(
            x <= 0.04045,
            x / 12.92,
            ((x + 0.055) / 1.055) ** 2.4,
        )
    return np.where(
        x <= 0.04045,
        x / 12.92,
        ((x + 0.055) / 1.055) ** 2.4,
    )


def linear_to_srgb(x):
    """Convert linear to sRGB using piecewise transfer function.

    Works with both torch tensors and numpy arrays.
    """
    if isinstance(x, torch.Tensor):
        return torch.where(
            x <= 0.0031308,
            x * 12.92,
            1.055 * x.clamp(min=0) ** (1.0 / 2.4) - 0.055,
        )
    return np.where(
        x <= 0.0031308,
        x * 12.92,
        1.055 * np.maximum(x, 0) ** (1.0 / 2.4) - 0.055,
    )


def despill_green(image, strength=1.0):
    """Luminance-preserving green spill removal.

    Args:
        image: [B, H, W, 3] sRGB tensor, values in [0, 1]
        strength: 0.0 (no despill) to 1.0 (full despill)

    Returns:
        Despilled image, same shape and range.
    """
    if strength <= 0:
        return image

    r, g, b = image[..., 0], image[..., 1], image[..., 2]
    limit = (r + b) / 2.0
    spill = (g - limit).clamp(min=0) * strength

    out = image.clone()
    out[..., 1] = g - spill
    out[..., 0] = r + spill / 2.0
    out[..., 2] = b + spill / 2.0
    return out.clamp(0, 1)
