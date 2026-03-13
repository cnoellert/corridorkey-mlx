"""
test_frame.py — single-frame inference test for corridorkey-mlx

Matches the reference CorridorKeyEngine pipeline exactly:
  1. Read linear EXR
  2. Resize to 2048px (model's native size) in linear space
  3. linear → sRGB
  4. ImageNet normalise  (mean=[0.485,0.456,0.406] std=[0.229,0.224,0.225])
  5. Concatenate mask channel  [H,W,4]
  6. Single forward pass (no tiling — model has global context at 2048px)
  7. Lanczos upsample alpha + fg back to native resolution
  8. Despill green from fg (sRGB)
  9. Optional despeckle (clean small disconnected components)
 10. linear_to_srgb → premultiply → pack RGBA EXR
 11. Apply garbage matte post-inference (dilate + multiply)

Usage
-----
python test_frame.py /path/to/frame.exr
python test_frame.py /path/to/frame.exr --garbage-matte /path/to/matte.exr
python test_frame.py /path/to/frame.exr --garbage-matte /path/to/matte.exr --preview
"""
from __future__ import annotations
import argparse, time
from pathlib import Path
import numpy as np
import mlx.core as mx

# ImageNet stats (model trained with these)
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)
MODEL_SIZE = 2048  # model's native inference resolution

# ---------------------------------------------------------------------------
# Color helpers
# ---------------------------------------------------------------------------

def _linear_to_srgb(x: np.ndarray) -> np.ndarray:
    """Linear → sRGB.  Clips to [0,1] first — HDR highlights become white,
    which is correct for feeding a keyer network (screens are always < 1.0)."""
    x = np.clip(x, 0.0, 1.0)
    return np.where(x <= 0.0031308, x * 12.92, 1.055 * np.power(x, 1.0 / 2.4) - 0.055)

def _srgb_to_linear(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0.0, None)
    return np.where(x <= 0.04045, x / 12.92, np.power((x + 0.055) / 1.055, 2.4))

def _despill(rgb: np.ndarray, strength: float = 1.0) -> np.ndarray:
    """Green despill — luminance-preserving average method. rgb: [H,W,3] sRGB 0-1."""
    if strength <= 0.0:
        return rgb
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    limit      = (r + b) / 2.0
    spill      = np.maximum(g - limit, 0.0)
    g_new      = g - spill
    r_new      = r + spill * 0.5
    b_new      = b + spill * 0.5
    despilled  = np.stack([r_new, g_new, b_new], axis=-1)
    return rgb * (1.0 - strength) + despilled * strength if strength < 1.0 else despilled


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def _read_exr_rgb(path: Path) -> np.ndarray:
    """Returns float32 [H, W, 3] linear RGB."""
    import OpenEXR, Imath
    f  = OpenEXR.InputFile(str(path))
    dw = f.header()['dataWindow']
    W  = dw.max.x - dw.min.x + 1
    H  = dw.max.y - dw.min.y + 1
    pt = Imath.PixelType(Imath.PixelType.FLOAT)
    ch = f.header().get('channels', {})
    def _c(name):
        if name in ch:
            return np.frombuffer(f.channel(name, pt), dtype=np.float32).reshape(H, W)
        return np.zeros((H, W), dtype=np.float32)
    return np.stack([_c('R'), _c('G'), _c('B')], axis=-1)


def _read_exr_mask(path: Path, H: int, W: int) -> np.ndarray:
    """Returns float32 [H, W, 1] mask, resized to (H, W) if needed."""
    import cv2, OpenEXR, Imath
    f  = OpenEXR.InputFile(str(path))
    dw = f.header()['dataWindow']
    mW = dw.max.x - dw.min.x + 1
    mH = dw.max.y - dw.min.y + 1
    pt = Imath.PixelType(Imath.PixelType.FLOAT)
    ch = f.header().get('channels', {})
    name = 'A' if 'A' in ch else ('R' if 'R' in ch else next(iter(ch)))
    arr  = np.frombuffer(f.channel(name, pt), dtype=np.float32).reshape(mH, mW)
    if arr.shape != (H, W):
        arr = cv2.resize(arr, (W, H), interpolation=cv2.INTER_LINEAR)
    return arr[:, :, None]


def _write_exr(path: Path, data: np.ndarray) -> None:
    """data: float32 [H, W, C] where C is 1, 3, or 4."""
    import OpenEXR, Imath
    H, W, C = data.shape
    names   = {1: ['Y'], 3: ['R','G','B'], 4: ['R','G','B','A']}[C]
    header  = OpenEXR.Header(W, H)
    header['channels'] = {n: Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT)) for n in names}
    f = OpenEXR.OutputFile(str(path), header)
    f.writePixels({n: data[:, :, i].tobytes() for i, n in enumerate(names)})
    f.close()


def _write_preview(path: Path, rgba_premul_linear: np.ndarray) -> None:
    """Comp premult RGBA over black, sRGB encode, save PNG."""
    from PIL import Image
    rgb  = np.clip(rgba_premul_linear[:, :, :3], 0, 1)
    rgb8 = (_linear_to_srgb(rgb) * 255).clip(0, 255).astype(np.uint8)
    Image.fromarray(rgb8).save(str(path))
    print(f"  Preview: {path.name}")

# ---------------------------------------------------------------------------
# Inference  (matches reference engine exactly)
# ---------------------------------------------------------------------------

def infer_frame(
    model,
    rgb_linear: np.ndarray,          # [H, W, 3] linear, 0-1+
    mask_linear: np.ndarray | None,  # [H, W, 1] linear, 0-1  — or None
    despill_strength: float = 1.0,
    despeckle: bool = True,
    despeckle_size: int = 400,
) -> np.ndarray:
    """
    Full reference pipeline. Returns premult RGBA [H, W, 4] linear float32.

    Steps mirror CorridorKeyEngine.process_frame(input_is_linear=True):
      resize-in-linear → linear_to_srgb → ImageNet-norm → inference
      → Lanczos-upsample → despeckle → despill(sRGB) → srgb_to_linear → premultiply
    """
    import cv2
    H, W = rgb_linear.shape[:2]

    if mask_linear is None:
        mask_linear = np.full((H, W, 1), 0.5, dtype=np.float32)

    # --- 1. Resize to model native size in linear space ---
    img_2k   = cv2.resize(rgb_linear,           (MODEL_SIZE, MODEL_SIZE), interpolation=cv2.INTER_LINEAR)
    mask_2k  = cv2.resize(mask_linear[:, :, 0], (MODEL_SIZE, MODEL_SIZE), interpolation=cv2.INTER_LINEAR)
    mask_2k  = np.clip(mask_2k, 0.0, 1.0)[:, :, None]

    # --- 2. linear → sRGB  (model trained on sRGB) ---
    img_srgb = _linear_to_srgb(img_2k)

    # --- 3. ImageNet normalise ---
    img_norm = (img_srgb - _MEAN) / _STD

    # --- 4. Concatenate mask channel and run model ---
    inp = np.concatenate([img_norm, mask_2k], axis=-1).astype(np.float32)
    x   = mx.array(inp[None])   # [1, 2048, 2048, 4]

    t0  = time.time()
    out = model(x)
    mx.eval(out['alpha'], out['fg'])
    print(f"[infer] Forward pass: {time.time()-t0:.2f}s")

    pred_alpha = np.array(out['alpha'][0])   # [2048, 2048, 1]  (0-1)
    pred_fg    = np.array(out['fg'][0])      # [2048, 2048, 3]  (sRGB 0-1)

    # --- 5. Lanczos upsample back to native resolution ---
    if (H, W) != (MODEL_SIZE, MODEL_SIZE):
        pred_alpha = cv2.resize(pred_alpha[:, :, 0], (W, H), interpolation=cv2.INTER_LANCZOS4)[:, :, None]
        pred_fg    = cv2.resize(pred_fg,             (W, H), interpolation=cv2.INTER_LANCZOS4)
    pred_alpha = np.clip(pred_alpha, 0.0, 1.0).astype(np.float32)
    pred_fg    = np.clip(pred_fg,    0.0, 1.0).astype(np.float32)

    # --- 6. Despeckle alpha (clean small disconnected islands) ---
    if despeckle:
        pred_alpha = _clean_matte(pred_alpha, area_threshold=despeckle_size)

    # --- 7. Despill FG (sRGB space, before linear conversion) ---
    pred_fg = _despill(pred_fg, strength=despill_strength)

    # --- 8. sRGB → linear, then premultiply ---
    fg_lin    = _srgb_to_linear(pred_fg)
    fg_premul = fg_lin * pred_alpha

    return np.concatenate([fg_premul, pred_alpha], axis=-1)  # [H, W, 4]


def _clean_matte(alpha: np.ndarray, area_threshold: int = 400,
                 dilation: int = 25, blur_size: int = 5) -> np.ndarray:
    """Remove small disconnected islands from predicted matte."""
    import cv2
    squeeze = alpha.ndim == 3
    if squeeze:
        alpha = alpha[:, :, 0]
    mask8   = (alpha > 0.5).astype(np.uint8) * 255
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask8, connectivity=8)
    cleaned = np.zeros_like(mask8)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= area_threshold:
            cleaned[labels == i] = 255
    if dilation > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilation*2+1, dilation*2+1))
        cleaned = cv2.dilate(cleaned, k)
    if blur_size > 0:
        b = blur_size * 2 + 1
        cleaned = cv2.GaussianBlur(cleaned, (b, b), 0)
    result = alpha * (cleaned.astype(np.float32) / 255.0)
    return result[:, :, None] if squeeze else result


def _apply_garbage_matte(result: np.ndarray, matte: np.ndarray,
                          dilation_px: int = 15) -> np.ndarray:
    """Dilate matte and multiply against premult RGBA result."""
    import cv2
    mask = matte[:, :, 0]
    if dilation_px > 0:
        k    = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilation_px*2+1, dilation_px*2+1))
        mask = cv2.dilate(mask, k)
    mask = mask[:, :, None]
    out  = result.copy()
    out[:, :, :3] *= mask
    out[:, :, 3:4] *= mask
    return out

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description='CorridorKey MLX single-frame test')
    ap.add_argument('frame',                 type=Path)
    ap.add_argument('--garbage-matte',       type=Path,  default=None)
    ap.add_argument('--gm-dilation',         type=int,   default=15)
    ap.add_argument('--despill-strength',    type=float, default=1.0)
    ap.add_argument('--no-despeckle',        action='store_true')
    ap.add_argument('--despeckle-size',      type=int,   default=400)
    ap.add_argument('--model',               type=Path,
                    default=Path('/Users/cnoellert/ComfyUI/models/corridorkey/CorridorKey_v1.0.pth'))
    ap.add_argument('--out-dir',             type=Path,  default=None)
    ap.add_argument('--preview',             action='store_true')
    ap.add_argument('--quantize',            choices=['int8'], default=None)
    args = ap.parse_args()

    frame_path = args.frame.expanduser().resolve()
    out_dir    = (args.out_dir or frame_path.parent).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = frame_path.stem

    # Auto-convert weights
    model_path = args.model.expanduser().resolve()
    if model_path.suffix != '.npz':
        from inference import _ensure_converted
        npz_path = _ensure_converted(model_path, args.quantize)
    else:
        npz_path = model_path

    # Load model
    from model import GreenFormer
    print("[test] Building GreenFormer …")
    gf = GreenFormer()
    weights = {k: v for k, v in dict(mx.load(str(npz_path))).items()
               if not k.startswith('__')}
    gf.load_weights(list(weights.items()))
    mx.eval(gf.parameters())
    gf.eval()
    print(f"[test] Weights loaded ({len(weights)} tensors)")

    # Read frame
    print(f"[test] Reading {frame_path.name} …")
    rgb_linear = _read_exr_rgb(frame_path)
    H, W = rgb_linear.shape[:2]
    print(f"[test] Frame: {W}×{H}  (will process at {MODEL_SIZE}px, upsample back)")

    # Read garbage matte
    mask = None
    if args.garbage_matte:
        gm_path = args.garbage_matte.expanduser().resolve()
        mask = _read_exr_mask(gm_path, H, W)
        print(f"[test] Mask loaded: {gm_path.name}")
    else:
        print("[test] No mask — using all-0.5 trimap")

    # Inference
    print(f"[test] Inferring …")
    t0     = time.time()
    result = infer_frame(
        gf, rgb_linear, mask,
        despill_strength=args.despill_strength,
        despeckle=not args.no_despeckle,
        despeckle_size=args.despeckle_size,
    )
    elapsed = time.time() - t0

    alpha = result[:, :, 3:4]
    print(f"[test] Total: {elapsed:.2f}s")
    print(f"[test] Alpha: min={float(alpha.min()):.4f}  max={float(alpha.max()):.4f}  "
          f"mean={float(alpha.mean()):.4f}")

    # Apply garbage matte post-inference
    if args.garbage_matte:
        print(f"[test] Applying garbage matte (dilation={args.gm_dilation}px) …")
        gm = _read_exr_mask(args.garbage_matte.expanduser().resolve(), H, W)
        result = _apply_garbage_matte(result, gm, dilation_px=args.gm_dilation)
        alpha2 = result[:, :, 3:4]
        print(f"[test] Alpha after GM: min={float(alpha2.min()):.4f}  "
              f"max={float(alpha2.max()):.4f}  mean={float(alpha2.mean()):.4f}")

    # Write outputs
    print(f"[test] Writing to {out_dir} …")
    _write_exr(out_dir / f"{stem}_alpha.exr",  result[:, :, 3:4])
    _write_exr(out_dir / f"{stem}_fg.exr",     result)
    _write_exr(out_dir / f"{stem}_key.exr",    result)
    print(f"  {stem}_alpha.exr")
    print(f"  {stem}_fg.exr")
    print(f"  {stem}_key.exr")

    if args.preview:
        _write_preview(out_dir / f"{stem}_preview.png", result)

    print("[test] Complete.")


if __name__ == '__main__':
    main()
