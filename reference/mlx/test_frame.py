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
import argparse, re, time
import cv2
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

def _linear_to_srgb(x: np.ndarray, clip_input: bool = False) -> np.ndarray:
    """Linear → sRGB.
    clip_input=True  : clip to [0,1] before encode (use for model input prep).
    clip_input=False : preserve values > 1 via power curve (use for despill round-trip).
    """
    if clip_input:
        x = np.clip(x, 0.0, 1.0)
    else:
        x = np.clip(x, 0.0, None)   # only kill negatives
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


def _read_exr_compression(path: Path) -> dict:
    """Read compression type and dwaCompressionLevel from an EXR header."""
    import OpenEXR
    f = OpenEXR.InputFile(str(path))
    h = f.header()
    return {
        'compression':        h.get('compression'),       # Imath.Compression value
        'dwaCompressionLevel': h.get('dwaCompressionLevel'),  # float or None
    }


def _write_exr(path: Path, data: np.ndarray,
               compression: dict | None = None) -> None:
    """
    Write float32 EXR.  data: [H, W, C] where C is 1, 3, or 4.
    compression: dict from _read_exr_compression(), or None for ZIP default.
    """
    import OpenEXR, Imath
    H, W, C = data.shape
    names   = {1: ['Y'], 3: ['R','G','B'], 4: ['R','G','B','A']}[C]
    header  = OpenEXR.Header(W, H)
    if compression and compression.get('compression') is not None:
        header['compression'] = compression['compression']
        lvl = compression.get('dwaCompressionLevel')
        if lvl is not None:
            header['dwaCompressionLevel'] = float(lvl)
    header['channels'] = {n: Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT))
                          for n in names}
    f = OpenEXR.OutputFile(str(path), header)
    f.writePixels({n: data[:, :, i].tobytes() for i, n in enumerate(names)})
    f.close()


def _write_preview(path: Path, rgba_premul: np.ndarray,
                   input_is_srgb: bool = False) -> None:
    """Comp premult RGBA over black, sRGB encode, save PNG.
    If input_is_srgb, data is already in sRGB — skip gamma encoding.
    If linear, apply linear→sRGB before writing.
    """
    from PIL import Image
    rgb  = rgba_premul[:, :, :3]
    rgb  = np.clip(rgb, 0.0, None)
    if input_is_srgb:
        rgb8 = (rgb * 255).clip(0, 255).astype(np.uint8)
    else:
        rgb8 = (_linear_to_srgb(rgb, clip_input=True) * 255).clip(0, 255).astype(np.uint8)
    Image.fromarray(rgb8).save(str(path))
    print(f"  Preview: {path.name}")

# ---------------------------------------------------------------------------
# Inference  (matches reference engine exactly)
# ---------------------------------------------------------------------------

def infer_frame(
    model,
    rgb_linear: np.ndarray,          # [H, W, 3] linear, 0-1+
    mask_linear: np.ndarray | None,  # [H, W, 1] linear, 0-1  — or None
    input_is_srgb: bool = False,
    despill_strength: float = 1.0,
    despeckle: bool = False,
    despeckle_size: int = 400,
    trimap_radius: int = 40,         # erode+dilate radius in native pixels
) -> np.ndarray:
    """
    Full reference pipeline. Returns premult RGBA [H, W, 4] linear float32.

    FG color = original linear plate (not model reconstruction).
    Alpha     = model prediction, despeckled.
    Despill   = applied to original sRGB plate, converted back to linear.

    Steps:
      resize-in-linear → linear_to_srgb → ImageNet-norm → inference
      → Lanczos-upsample alpha → despeckle
      → despill original sRGB → srgb_to_linear → premultiply by model alpha
    """
    import cv2
    H, W = rgb_linear.shape[:2]

    trimap_full = None   # tight trimap returned to caller for hard-constraint enforcement

    # --- 1. Resize to model native size — straight stretch to square ---
    # The model was trained with stretched-square inputs (no padding).
    # Padding introduces out-of-distribution edge regions that corrupt the refiner.
    # Stretch distortion is minor at 2048x1080 -> 2048x2048 (~11% vertical squeeze).
    img_2k  = cv2.resize(rgb_linear,           (MODEL_SIZE, MODEL_SIZE), interpolation=cv2.INTER_LINEAR)
    mask_2k = cv2.resize(mask_linear[:, :, 0], (MODEL_SIZE, MODEL_SIZE), interpolation=cv2.INTER_LINEAR)
    mask_2k = np.clip(mask_2k, 0.0, 1.0)[:, :, None]

    # --- 2. linear → sRGB  (model trained on sRGB) ---
    if input_is_srgb:
        img_srgb = np.clip(img_2k, 0.0, 1.0).astype(np.float32)
    else:
        img_srgb = _linear_to_srgb(img_2k, clip_input=True)  # SDR for model

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

    # Flush Metal cache after numpy conversion — prevents graph/cache
    # accumulation across frames which causes steadily increasing frame times.
    mx.clear_cache()

    # --- 5. Bilinear upsample back to native resolution ---
    # Matches ComfyUI node: F.interpolate(..., mode="bilinear", align_corners=False).
    # Lanczos4 introduces ringing on fine semi-transparent edge detail.
    pred_alpha = cv2.resize(pred_alpha[:, :, 0], (W, H), interpolation=cv2.INTER_LINEAR)[:, :, None]
    pred_fg    = cv2.resize(pred_fg,             (W, H), interpolation=cv2.INTER_LINEAR)
    pred_alpha = np.clip(pred_alpha, 0.0, 1.0).astype(np.float32)
    pred_fg    = np.clip(pred_fg,    0.0, 1.0).astype(np.float32)

    # --- 6. Despeckle alpha (clean small disconnected islands) ---
    if despeckle:
        pred_alpha = _clean_matte(pred_alpha, area_threshold=despeckle_size)

    # --- 7. Despill, output raw — no colorspace transforms ---
    # Write exactly what the model gives us. No linearise, no premult assumptions.
    fg_out_raw    = _despill(pred_fg, strength=despill_strength)  # sRGB, straight, 0-1

    fg_straight   = np.concatenate([fg_out_raw, pred_alpha], axis=-1)          # [H, W, 4]
    key_premult   = np.concatenate([fg_out_raw * pred_alpha, pred_alpha], axis=-1)

    return key_premult, fg_straight, trimap_full


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


def _make_trimap(mask: np.ndarray, erode_r: int = 40, dilate_r: int = 40) -> np.ndarray:
    """
    Convert a binary or soft mask into a proper trimap:
        1.0 = definite foreground  (eroded core)
        0.0 = definite background  (outside dilated edge)
        0.5 = uncertain            (narrow band between erode and dilate)

    erode_r / dilate_r are in native image pixels.
    Default 40px at 6K gives ~14px uncertain band at 2048px model resolution.

    This dramatically stabilises temporal consistency: the model receives the
    same clear FG/BG/unknown map every frame, only refining the edge band.
    """
    import cv2
    squeeze = mask.ndim == 3
    if squeeze:
        mask = mask[:, :, 0]
    mask8  = (mask > 0.5).astype(np.uint8) * 255
    ke     = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erode_r*2+1,  erode_r*2+1))
    kd     = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_r*2+1, dilate_r*2+1))
    fg     = cv2.erode(mask8,  ke).astype(np.float32) / 255.0   # definite FG
    bg_inv = cv2.dilate(mask8, kd).astype(np.float32) / 255.0   # dilated → outside is BG
    trimap = np.where(fg > 0.5, 1.0, np.where(bg_inv < 0.5, 0.0, 0.5)).astype(np.float32)
    return trimap[:, :, None] if squeeze else trimap


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
    ap.add_argument('--gm-dilation',         type=int,   default=-1,
                    help='Post-inference GM dilation px. -1 = model input only, no multiply (default)')
    ap.add_argument('--input-is-srgb',      action='store_true',
                    help='Input is already sRGB/REC709 — skip linear→sRGB conversion')
    ap.add_argument('--despill-strength',    type=float, default=1.0)
    ap.add_argument('--trimap-radius',     type=int,   default=40,
                    help='Erode+dilate radius in native px for trimap construction (0=disable, default 40)')
    ap.add_argument('--despeckle',            action='store_true',  help='Enable alpha despeckle (default: off)')
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
    # Split stem into base name + frame number for correct sequence naming:
    # DC_0190_raw_L02.1053  →  base='DC_0190_raw_L02'  frame_token='1053'
    _m = re.match(r'^(.*?)[._](\d+)$', frame_path.stem)
    if _m:
        _base, _frame_token = _m.group(1), _m.group(2)
    else:
        _base, _frame_token = frame_path.stem, ''
    stem = _base  # used for output naming below

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
    compression = _read_exr_compression(frame_path)
    print(f"[test] Compression: {compression['compression']}  "
          f"dwaLevel={compression.get('dwaCompressionLevel')}")
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
    result, fg_straight, trimap_full = infer_frame(
        gf, rgb_linear, mask,
        input_is_srgb=args.input_is_srgb,
        despill_strength=args.despill_strength,
        despeckle=args.despeckle,
        despeckle_size=args.despeckle_size,
        trimap_radius=args.trimap_radius,
    )
    elapsed = time.time() - t0

    alpha = result[:, :, 3:4]
    print(f"[test] Total: {elapsed:.2f}s")
    print(f"[test] Alpha: min={float(alpha.min()):.4f}  max={float(alpha.max()):.4f}  "
          f"mean={float(alpha.mean()):.4f}")

    # Enforce trimap hard constraints using the trimap computed inside infer_frame
    # (no recomputation — morphological ops only happen once per frame).
    # No post-inference constraint enforcement — original engine applies none.

    # Apply garbage matte post-inference
    if args.garbage_matte and args.gm_dilation >= 0:
        print(f"[test] Applying garbage matte (dilation={args.gm_dilation}px) …")
        gm = _read_exr_mask(args.garbage_matte.expanduser().resolve(), H, W)
        result      = _apply_garbage_matte(result,      gm, dilation_px=args.gm_dilation)
        fg_straight = _apply_garbage_matte(fg_straight, gm, dilation_px=args.gm_dilation)
    elif args.garbage_matte:
        print(f"[test] Garbage matte used as model input only (no post-inference multiply)")
        alpha2 = result[:, :, 3:4]
        print(f"[test] Alpha after GM: min={float(alpha2.min()):.4f}  "
              f"max={float(alpha2.max()):.4f}  mean={float(alpha2.mean()):.4f}")

    # Unpack premult result into constituent outputs
    alpha_out = result[:, :, 3:4]      # linear alpha, no gamma
    key_out   = result                 # linear premult RGBA — comp-ready
    fg_out    = fg_straight            # linear straight RGBA (scene-linear for EXR)

    print(f"[test] Writing to {out_dir} …")
    def _outpath(suffix):
        # Produces:  base_suffix.frame.exr  e.g. DC_0190_raw_L02_alpha.1053.exr
        if _frame_token:
            return out_dir / f"{stem}_{suffix}.{_frame_token}.exr"
        return out_dir / f"{stem}_{suffix}.exr"

    # Always write outputs with lossless ZIP compression.
    _lossless = {'compression': None, 'dwaCompressionLevel': None}
    _write_exr(_outpath('alpha'), alpha_out, compression=_lossless)
    _write_exr(_outpath('fg'),    fg_out,    compression=_lossless)
    _write_exr(_outpath('key'),   key_out,   compression=_lossless)

    # Diagnostic: unpremult (FG / alpha) — treat fg_out as if it were premultiplied
    # and divide out the alpha. Useful to compare against Comfy if their FG is premult.
    _alpha3 = np.clip(alpha_out, 1e-6, 1.0)   # avoid div/0
    fg_unpremult = np.concatenate([fg_out[:,:,:3] / _alpha3, alpha_out], axis=-1)
    fg_unpremult = np.clip(fg_unpremult, 0.0, 1.0).astype(np.float32)
    _write_exr(_outpath('fg_unpremult'), fg_unpremult, compression=_lossless)
    print(f"  {_outpath('alpha').name}  (single channel)")
    print(f"  {_outpath('fg').name}     (straight RGBA)")
    print(f"  {_outpath('key').name}    (premult RGBA — comp-ready)")

    if args.preview:
        _write_preview(out_dir / f"{stem}_preview.png", result, input_is_srgb=False)  # key is linear premult

    print("[test] Complete.")


if __name__ == '__main__':
    main()
