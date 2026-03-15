"""
On-device optimised inference wrapper for CorridorKeyEngine.

Why this exists
---------------
The reference CorridorKeyEngine.process_frame() works entirely in CPU/numpy:
  GPU tensor -> .cpu().numpy() -> cv2.resize (CPU) -> numpy norm (CPU)
  -> torch.from_numpy().to(device) -> model forward -> .cpu().numpy()
  -> cv2.resize (CPU) -> return numpy

This wrapper replaces process_frame with an implementation that:
  1. Accepts ComfyUI tensors directly (no numpy conversion at the boundary)
  2. Keeps resize, normalisation, and post-processing on the GPU
  3. Uses channels_last (NHWC) memory format — Metal's native layout (~15% faster)
  4. Skips float16 autocast on MPS (NaN-prone), uses float32 throughout
  5. Calls empty_cache() after inference to keep the memory pool clean
  6. Only touches CPU for scipy despeckle (no PyTorch equivalent)
"""

import numpy as np
import torch
import torch.nn.functional as F

from .device import get_autocast_ctx, clear_cache

_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD  = [0.229, 0.224, 0.225]


class OptimizedEngine:
    """
    Thin wrapper around CorridorKeyEngine that replaces process_frame
    with an on-device implementation optimised for MPS and CUDA.
    """

    def __init__(self, engine):
        self._engine    = engine
        self.device     = engine.device
        self.img_size   = engine.img_size
        self.model      = engine.model
        self.use_refiner = engine.use_refiner

        # Pre-build ImageNet stats on device as [1, 3, 1, 1] for broadcast
        self._mean = torch.tensor(
            _IMAGENET_MEAN, device=self.device, dtype=torch.float32
        ).view(1, 3, 1, 1)

        self._std = torch.tensor(
            _IMAGENET_STD, device=self.device, dtype=torch.float32
        ).view(1, 3, 1, 1)

        # channels_last: Metal's native NHWC layout gives ~10-20% speedup
        if self.device.type in ("mps", "cuda"):
            try:
                self.model = self.model.to(memory_format=torch.channels_last)
                print(f"[CorridorKey] channels_last memory format enabled on {self.device}")
            except Exception as e:
                print(f"[CorridorKey] channels_last unavailable ({e}), using contiguous")


    @torch.no_grad()
    def process_frame_tensor(
        self,
        image_t: torch.Tensor,    # [H, W, 3] float32 sRGB [0, 1]
        mask_t:  torch.Tensor,    # [H, W]    float32 linear [0, 1]
        refiner_scale:    float = 1.0,
        input_is_linear:  bool  = False,
        despill_strength: float = 1.0,
        auto_despeckle:   bool  = True,
        despeckle_size:   int   = 400,
    ) -> dict:
        """
        Process a single frame entirely on-device.
        Returns dict of CPU float32 tensors:
            'alpha':     [H, W, 1]  linear alpha matte
            'fg':        [H, W, 3]  sRGB foreground (straight)
            'processed': [H, W, 4]  linear premultiplied RGBA
        """
        device = self.device
        h, w = image_t.shape[:2]

        img  = image_t.to(device=device, dtype=torch.float32)
        mask = mask_t.to(device=device, dtype=torch.float32)

        if input_is_linear:
            img = _linear_to_srgb(img)

        # Resize on device — replaces cv2.resize CPU calls
        img_4d  = img.permute(2, 0, 1).unsqueeze(0)      # [1, 3, H, W]
        mask_4d = mask.unsqueeze(0).unsqueeze(0)          # [1, 1, H, W]
        target  = (self.img_size, self.img_size)

        img_r = F.interpolate(img_4d,  size=target, mode="bilinear", align_corners=False)
        msk_r = F.interpolate(mask_4d, size=target, mode="bilinear", align_corners=False)

        # ImageNet normalisation on device
        img_norm = (img_r - self._mean) / self._std       # [1, 3, S, S]
        inp = torch.cat([img_norm, msk_r], dim=1)         # [1, 4, S, S]

        if device.type in ("mps", "cuda"):
            inp = inp.to(memory_format=torch.channels_last)


        # Forward pass — nullcontext on MPS (float32), autocast on CUDA
        handle = None
        if refiner_scale != 1.0 and self._engine.model.refiner is not None:
            def _scale_hook(module, input, output):
                return output * refiner_scale
            handle = self._engine.model.refiner.register_forward_hook(_scale_hook)

        with get_autocast_ctx(device):
            out = self.model(inp)

        if handle is not None:
            handle.remove()

        pred_alpha = out["alpha"]   # [1, 1, S, S]
        pred_fg    = out["fg"]      # [1, 3, S, S]

        # Resize outputs back to original resolution on device
        orig = (h, w)
        alpha_up = F.interpolate(pred_alpha.float(), size=orig,
                                 mode="bilinear", align_corners=False)
        fg_up    = F.interpolate(pred_fg.float(),    size=orig,
                                 mode="bilinear", align_corners=False)

        alpha_hwc = alpha_up[0].permute(1, 2, 0).contiguous()   # [H, W, 1]
        fg_hwc    = fg_up[0].permute(1, 2, 0).contiguous()      # [H, W, 3]

        # Despeckle — unavoidable CPU trip (scipy connected components)
        if auto_despeckle:
            alpha_np  = alpha_hwc.cpu().numpy()
            alpha_np  = _despeckle_np(alpha_np, despeckle_size)
            alpha_hwc = torch.from_numpy(alpha_np).to(device)

        # Despill and premultiply on device
        if despill_strength > 0.0:
            fg_hwc = _despill_green(fg_hwc, despill_strength)

        fg_lin    = _srgb_to_linear(fg_hwc)
        fg_premul = fg_lin * alpha_hwc
        processed = torch.cat([fg_premul, alpha_hwc], dim=-1)    # [H, W, 4]

        clear_cache(device)

        return {
            "alpha":     alpha_hwc.cpu().float(),
            "fg":        fg_hwc.cpu().float(),
            "processed": processed.cpu().float(),
        }


    def process_frame(self, image, mask_linear, **kwargs) -> dict:
        """
        Numpy shim — preserves original CorridorKeyEngine API.
        Converts numpy inputs to tensors, returns numpy outputs.
        """
        img_t = (
            torch.from_numpy(np.asarray(image, dtype=np.float32))
            if isinstance(image, np.ndarray) else image.float()
        )
        msk_t = (
            torch.from_numpy(np.asarray(mask_linear, dtype=np.float32))
            if isinstance(mask_linear, np.ndarray) else mask_linear.float()
        )
        if msk_t.ndim == 3:
            msk_t = msk_t[..., 0]

        result = self.process_frame_tensor(img_t, msk_t, **kwargs)
        return {k: v.numpy() for k, v in result.items()}


# ---------------------------------------------------------------------------
# On-device helpers
# ---------------------------------------------------------------------------

def _srgb_to_linear(x: torch.Tensor) -> torch.Tensor:
    return torch.where(
        x <= 0.04045,
        x / 12.92,
        ((x + 0.055) / 1.055).clamp(min=0.0) ** 2.4,
    )


def _linear_to_srgb(x: torch.Tensor) -> torch.Tensor:
    return torch.where(
        x <= 0.0031308,
        x * 12.92,
        1.055 * x.clamp(min=1e-12) ** (1.0 / 2.4) - 0.055,
    )


def _despill_green(image: torch.Tensor, strength: float = 1.0) -> torch.Tensor:
    r, g, b  = image[..., 0], image[..., 1], image[..., 2]
    limit    = (r + b) * 0.5
    spill    = (g - limit).clamp(min=0.0) * strength
    out      = image.clone()
    out[..., 0] = (r + spill * 0.5).clamp(0, 1)
    out[..., 1] = (g - spill).clamp(0, 1)
    out[..., 2] = (b + spill * 0.5).clamp(0, 1)
    return out


def _despeckle_np(alpha_np: np.ndarray, min_size: int) -> np.ndarray:
    """Connected-component despeckle. CPU/numpy only."""
    from scipy import ndimage
    squeeze = alpha_np.ndim == 3
    a2d     = alpha_np[..., 0] if squeeze else alpha_np
    binary  = a2d > 0.5
    labeled, n = ndimage.label(binary)
    cleaned = binary.copy()
    for lid in range(1, n + 1):
        comp = labeled == lid
        if comp.sum() < min_size:
            cleaned[comp] = False
    result = a2d * cleaned.astype(np.float32)
    return result[..., np.newaxis] if squeeze else result
