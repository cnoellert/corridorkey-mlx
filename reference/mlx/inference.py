"""
inference.py  —  CorridorKey MLX batch processor

Usage
-----
python inference.py /path/to/frames/*.exr --model CorridorKey_v1.0.pth [OPTIONS]

On first run the .pth is automatically converted to an .mlx.npz cache beside it.
Subsequent runs load the .npz directly; no PyTorch required.

Options
-------
--model     PATH    Source .pth (or pre-converted .mlx.npz / .mlx.int8.npz)
--out-dir   DIR     Output directory (default: ./output)
--tile-size INT     Inference tile size (default: 512; increase if RAM allows)
--overlap   INT     Tile overlap in pixels to suppress seams (default: 64)
--despill           Apply green despill to FG output (default: True)
--no-despill        Disable despill
--quantize  INT8    Force int8 quantisation on first conversion
--workers   INT     Number of parallel I/O threads (default: 4)
"""
from __future__ import annotations

import argparse
import hashlib
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional

import mlx.core as mx
import mlx.nn as nn
import numpy as np

# Optional: OpenEXR I/O — falls back to OpenCV if unavailable
try:
    import OpenEXR
    import Imath
    _HAS_OPENEXR = True
except ImportError:
    _HAS_OPENEXR = False
    import cv2  # type: ignore


# ---------------------------------------------------------------------------
# Weight loading with folded BN support
# ---------------------------------------------------------------------------

def _load_weights(npz_path: Path, model: nn.Module) -> None:
    """Load converted .npz weights into the MLX model."""
    weights = dict(mx.load(str(npz_path)))
    # Strip metadata keys — all payload keys load directly (FoldedBN fix)
    weights = {k: v for k, v in weights.items() if not k.startswith("__")}
    model.load_weights(list(weights.items()))
    mx.eval(model.parameters())


# ---------------------------------------------------------------------------
# Auto-convert: check cache validity via SHA-256
# ---------------------------------------------------------------------------

def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while data := f.read(1 << 20):
            h.update(data)
    return h.hexdigest()


def _cache_path(pth: Path, quantize: Optional[str]) -> Path:
    stem = pth.stem
    if quantize:
        return pth.parent / f"{stem}.mlx.{quantize}.npz"
    return pth.parent / f"{stem}.mlx.npz"


def _ensure_converted(model_path: Path, quantize: Optional[str] = None) -> Path:
    """
    Option C: return path to MLX .npz, auto-converting from .pth if needed.
    Conversion is skipped if a valid cache already exists (SHA-256 match).
    """
    if model_path.suffix == ".npz":
        return model_path  # user passed pre-converted weights directly

    cache = _cache_path(model_path, quantize)

    if cache.exists():
        # Validate cache freshness
        npz = dict(mx.load(str(cache)))
        stored_hash_bytes = bytes(np.array(npz.get("__src_sha256__", mx.array([])), dtype=np.uint8))
        stored_hash = stored_hash_bytes.decode(errors="replace")
        current_hash = _sha256(model_path)
        if stored_hash == current_hash:
            print(f"[inference] Cache valid → {cache.name}")
            return cache
        else:
            print("[inference] Source weights changed — re-converting …")

    # First run or stale cache
    from convert import convert  # lazy import keeps PyTorch out of hot path
    print(f"[inference] First run: converting {model_path.name} → {cache.name}")
    print("[inference] This takes ~30 s and will not happen again.")
    convert(model_path, cache, quantize=quantize)
    return cache


# ---------------------------------------------------------------------------
# EXR I/O
# ---------------------------------------------------------------------------

def _read_exr(path: Path) -> np.ndarray:
    """Returns float32 [H, W, C] RGB in linear light, values typically 0-1+."""
    if _HAS_OPENEXR:
        f = OpenEXR.InputFile(str(path))
        h = f.header()
        dw = h["dataWindow"]
        W = dw.max.x - dw.min.x + 1
        H = dw.max.y - dw.min.y + 1
        pt = Imath.PixelType(Imath.PixelType.FLOAT)
        R = np.frombuffer(f.channel("R", pt), dtype=np.float32).reshape(H, W)
        G = np.frombuffer(f.channel("G", pt), dtype=np.float32).reshape(H, W)
        B = np.frombuffer(f.channel("B", pt), dtype=np.float32).reshape(H, W)
        return np.stack([R, G, B], axis=-1)
    else:
        img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED | cv2.IMREAD_ANYDEPTH)
        if img is None:
            raise IOError(f"Could not read {path}")
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)


def _write_exr(path: Path, rgba: np.ndarray) -> None:
    """rgba: float32 [H, W, 4]."""
    if _HAS_OPENEXR:
        H, W = rgba.shape[:2]
        header = OpenEXR.Header(W, H)
        header["channels"] = {
            c: Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT))
            for c in "RGBA"
        }
        f = OpenEXR.OutputFile(str(path), header)
        f.writePixels({
            "R": rgba[:, :, 0].tobytes(),
            "G": rgba[:, :, 1].tobytes(),
            "B": rgba[:, :, 2].tobytes(),
            "A": rgba[:, :, 3].tobytes(),
        })
        f.close()
    else:
        bgra = cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA)
        cv2.imwrite(str(path), bgra)


# ---------------------------------------------------------------------------
# Tiled inference
# ---------------------------------------------------------------------------

def _pad_to_multiple(x: np.ndarray, multiple: int = 32) -> tuple[np.ndarray, tuple]:
    """Pad H and W to next multiple of `multiple`. Returns padded array + pad amounts."""
    H, W = x.shape[:2]
    pH = (multiple - H % multiple) % multiple
    pW = (multiple - W % multiple) % multiple
    padded = np.pad(x, ((0, pH), (0, pW), (0, 0)), mode="reflect")
    return padded, (pH, pW)


def _infer_tile(model: nn.Module, tile_rgba: np.ndarray) -> np.ndarray:
    """
    tile_rgba: float32 [H, W, 4] (RGB + trimap, linear, 0-1)
    Returns:   float32 [H, W, 4] (premult RGBA)
    """
    x = mx.array(tile_rgba[None])  # [1, H, W, 4]
    out = model(x)
    mx.eval(out["alpha"], out["fg"])
    alpha = np.array(out["alpha"][0])  # [H, W, 1]
    fg    = np.array(out["fg"][0])     # [H, W, 3]
    return np.concatenate([fg * alpha, alpha], axis=-1)  # premult RGBA


def infer_frame(
    model: nn.Module,
    rgb: np.ndarray,
    trimap: Optional[np.ndarray],
    tile_size: int = 512,
    overlap: int = 64,
) -> np.ndarray:
    """
    Full-resolution frame inference with overlapping tiles.
    rgb:    float32 [H, W, 3]  linear light
    trimap: float32 [H, W, 1]  or None (uses all-green placeholder)
    Returns float32 [H, W, 4] premult RGBA
    """
    H, W = rgb.shape[:2]

    if trimap is None:
        trimap = np.ones((H, W, 1), dtype=np.float32) * 0.5

    rgba_in = np.concatenate([rgb, trimap], axis=-1)  # [H, W, 4]

    # Fast path: image fits in one tile
    if H <= tile_size and W <= tile_size:
        padded, (pH, pW) = _pad_to_multiple(rgba_in, 32)
        result = _infer_tile(model, padded)
        return result[:H, :W]

    # Tiled path with feathered blending
    output   = np.zeros((H, W, 4), dtype=np.float64)
    weights  = np.zeros((H, W, 1), dtype=np.float64)
    step = tile_size - overlap

    ys = list(range(0, H - tile_size + 1, step)) + ([H - tile_size] if H > tile_size else [])
    xs = list(range(0, W - tile_size + 1, step)) + ([W - tile_size] if W > tile_size else [])

    for y in set(ys):
        for x in set(xs):
            tile = rgba_in[y:y+tile_size, x:x+tile_size]
            padded, (pH, pW) = _pad_to_multiple(tile, 32)
            result = _infer_tile(model, padded)[:tile_size, :tile_size]

            # Hanning window for smooth blending
            win_y = np.hanning(tile_size).reshape(-1, 1)
            win_x = np.hanning(tile_size).reshape(1, -1)
            win   = (win_y * win_x)[:, :, None].astype(np.float64)

            output [y:y+tile_size, x:x+tile_size] += result * win
            weights[y:y+tile_size, x:x+tile_size] += win

    return (output / np.maximum(weights, 1e-8)).astype(np.float32)


# ---------------------------------------------------------------------------
# CLI driver
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="CorridorKey MLX batch inference")
    ap.add_argument("frames", nargs="+", type=Path, help="Input EXR/image frames")
    ap.add_argument("--model",     required=True, type=Path, help=".pth or .mlx.npz checkpoint")
    ap.add_argument("--out-dir",   type=Path, default=Path("./output"), help="Output directory")
    ap.add_argument("--tile-size", type=int,  default=512,  help="Tile size (default: 512)")
    ap.add_argument("--overlap",   type=int,  default=64,   help="Tile overlap px (default: 64)")
    ap.add_argument("--quantize",  choices=["int8"], default=None)
    ap.add_argument("--no-despill", action="store_true")
    ap.add_argument("--workers",   type=int,  default=4)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # ---- Option C: auto-convert on first use --------------------------------
    model_path = args.model.expanduser().resolve()
    npz_path   = _ensure_converted(model_path, args.quantize)

    # ---- Build model and load weights ---------------------------------------
    from model import GreenFormer
    print("[inference] Building GreenFormer …")
    model = GreenFormer()
    _load_weights(npz_path, model)
    model.eval()
    print(f"[inference] Weights loaded from {npz_path.name}")

    # ---- Process frames -----------------------------------------------------
    frames = sorted(args.frames)
    print(f"[inference] Processing {len(frames)} frame(s) …")

    def _process(frame_path: Path) -> None:
        t0 = time.time()
        rgb = _read_exr(frame_path)
        result = infer_frame(
            model, rgb,
            trimap=None,
            tile_size=args.tile_size,
            overlap=args.overlap,
        )
        out_path = args.out_dir / (frame_path.stem + "_key.exr")
        _write_exr(out_path, result)
        elapsed = time.time() - t0
        print(f"  {frame_path.name} → {out_path.name}  ({elapsed:.2f}s)")

    if args.workers > 1:
        # I/O threads; MLX inference stays on the main thread (Metal serialised)
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            list(ex.map(_process, frames))
    else:
        for f in frames:
            _process(f)

    print("[inference] Complete.")


if __name__ == "__main__":
    main()
