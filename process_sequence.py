"""
process_sequence.py  —  CorridorKey MLX sequence processor

Usage
-----
# Full sequence, matching per-frame garbage matte, outputs to explicit dir
python process_sequence.py \\
    '/path/to/frames/shot.1001.exr' \\
    --garbage-matte '/path/to/mattes/shot_matte.1001.exr' \\
    --out-dir '/path/to/output'

# Subrange only
python process_sequence.py \\
    '/path/to/frames/shot.1001.exr' \\
    --garbage-matte '/path/to/mattes/shot_matte.1001.exr' \\
    --out-dir '/path/to/output' \\
    --start 1010 --end 1050

# Single matte applied to all frames
python process_sequence.py \\
    '/path/to/frames/shot.1001.exr' \\
    --garbage-matte '/path/to/mattes/shot_matte_static.exr' \\
    --static-matte \\
    --out-dir '/path/to/output'

Auto-detection
--------------
Given a first-frame path like  shot.1001.exr  the script:
  1. Parses the frame number (last contiguous digit run before the extension)
  2. Globs the directory for all matching frames
  3. Sorts numerically and processes in order

Output naming
-------------
Input:  shot.1001.exr
Output: <out-dir>/shot.1001_alpha.exr   (single channel)
        <out-dir>/shot.1001_fg.exr      (straight RGBA)
        <out-dir>/shot.1001_key.exr     (premult RGBA, comp-ready)
"""
from __future__ import annotations
import argparse, re, sys, time
from pathlib import Path
from typing import Optional
import numpy as np
import mlx.core as mx

# ---------------------------------------------------------------------------
# Sequence parsing
# ---------------------------------------------------------------------------

def _parse_frame_number(path: Path) -> tuple[str, int, str, str]:
    """
    Split a Flame-style path into (dir, stem_prefix, frame_number, extension).
    Handles:  shot_name.1001.exr  /  shot_name_1001.exr  /  shot1001.exr
    Returns:  (prefix, frame_num, zero_pad_width, suffix)
    E.g. 'DC_0190_raw_L02.1053.exr' → ('DC_0190_raw_L02.', 1053, 4, '.exr')
    """
    name = path.name
    ext  = path.suffix                         # '.exr'
    stem = path.stem                           # 'DC_0190_raw_L02.1053'
    m    = re.search(r'^(.*?)(\d+)$', stem)
    if not m:
        raise ValueError(f"Cannot parse frame number from: {name}")
    prefix    = m.group(1)   # 'DC_0190_raw_L02.'
    frame_str = m.group(2)   # '1053'
    return prefix, int(frame_str), len(frame_str), ext


def _build_frame_path(directory: Path, prefix: str, frame: int,
                       pad: int, ext: str) -> Path:
    return directory / f"{prefix}{str(frame).zfill(pad)}{ext}"


def _detect_sequence(first_frame: Path) -> list[Path]:
    """
    Given the first frame, glob the directory for all frames in the same
    sequence and return them sorted numerically.
    """
    directory          = first_frame.parent
    prefix, _, pad, ext = _parse_frame_number(first_frame)
    # Glob: match same prefix + any digits + same extension
    pattern            = f"{re.escape(prefix)}{'[0-9]' * pad}{re.escape(ext)}"
    # Fallback to broader glob if exact-width match yields nothing
    candidates         = sorted(directory.glob(f"{prefix}*{ext}"))
    # Filter to purely numeric suffix of same or similar width
    result = []
    for p in candidates:
        try:
            _parse_frame_number(p)
            result.append(p)
        except ValueError:
            pass
    # Sort numerically by frame number
    result.sort(key=lambda p: _parse_frame_number(p)[1])
    return result


def _matte_path_for_frame(matte_first: Path, frame: int) -> Path:
    """Given the first matte frame path, build the path for frame N."""
    prefix, _, pad, ext = _parse_frame_number(matte_first)
    return matte_first.parent / f"{prefix}{str(frame).zfill(pad)}{ext}"


# ---------------------------------------------------------------------------
# Model loader  (load once, reuse for all frames)
# ---------------------------------------------------------------------------

def _load_model(model_path: Path, quantize: Optional[str] = None):
    """Load GreenFormer with weights. Returns ready-to-use model."""
    from model import GreenFormer
    from inference import _ensure_converted

    if model_path.suffix != '.npz':
        npz_path = _ensure_converted(model_path, quantize)
    else:
        npz_path = model_path

    gf = GreenFormer()
    weights = {k: v for k, v in dict(mx.load(str(npz_path))).items()
               if not k.startswith('__')}
    gf.load_weights(list(weights.items()))
    mx.eval(gf.parameters())
    gf.eval()
    return gf, len(weights)


# ---------------------------------------------------------------------------
# Per-frame processor  (thin wrapper around test_frame pipeline)
# ---------------------------------------------------------------------------

def _process_frame(
    model,
    frame_path: Path,
    matte_path: Optional[Path],
    out_dir: Path,
    despill_strength: float = 1.0,
    despeckle: bool = True,
    despeckle_size: int = 400,
    gm_dilation: int = 15,
    trimap_radius: int = 40,
) -> dict:
    """
    Process one frame. Returns timing + alpha stats dict.
    Raises on hard error; caller decides whether to skip or abort.
    """
    from test_frame import (
        _read_exr_rgb, _read_exr_mask, _write_exr,
        _read_exr_compression, infer_frame, _apply_garbage_matte,
    )

    t_start = time.time()

    # Read inputs
    rgb_linear   = _read_exr_rgb(frame_path)
    H, W         = rgb_linear.shape[:2]
    compression  = _read_exr_compression(frame_path)

    mask = None
    if matte_path and matte_path.exists():
        mask = _read_exr_mask(matte_path, H, W)

    # Inference
    result = infer_frame(
        model, rgb_linear, mask,
        despill_strength=despill_strength,
        despeckle=despeckle,
        despeckle_size=despeckle_size,
        trimap_radius=trimap_radius,
    )

    # Enforce trimap hard constraints post-inference
    if mask is not None and trimap_radius > 0:
        import cv2 as _cv2
        trimap = _make_trimap(mask, erode_r=trimap_radius, dilate_r=trimap_radius)
        tri_full = _cv2.resize(trimap[:, :, 0], (W, H), interpolation=_cv2.INTER_NEAREST)
        alpha_ch = result[:, :, 3]
        alpha_ch = np.where(tri_full >= 1.0, 1.0,
                   np.where(tri_full <= 0.0, 0.0, alpha_ch)).astype(np.float32)
        result   = np.concatenate([result[:, :, :3] * alpha_ch[:, :, None],
                                   alpha_ch[:, :, None]], axis=-1)

    # Apply garbage matte post-inference
    if matte_path and matte_path.exists():
        gm = _read_exr_mask(matte_path, H, W)
        result = _apply_garbage_matte(result, gm, dilation_px=gm_dilation)

    # Unpack result into constituent outputs
    alpha_out = result[:, :, 3:4]
    key_out   = result
    eps       = 1e-6
    fg_out    = np.concatenate(
        [result[:, :, :3] / (alpha_out + eps), alpha_out], axis=-1
    )

    # Build output paths: base_suffix.frame.exr  e.g. DC_0190_raw_L02_alpha.1053.exr
    import re as _re
    _m = _re.match(r'^(.*?)[._](\d+)$', frame_path.stem)
    if _m:
        _base, _frame_token = _m.group(1), _m.group(2)
    else:
        _base, _frame_token = frame_path.stem, ''

    def _outpath(suffix):
        if _frame_token:
            return out_dir / f"{_base}_{suffix}.{_frame_token}.exr"
        return out_dir / f"{_base}_{suffix}.exr"

    _write_exr(_outpath('alpha'), alpha_out, compression=compression)
    _write_exr(_outpath('fg'),    fg_out,    compression=compression)
    _write_exr(_outpath('key'),   key_out,   compression=compression)

    elapsed = time.time() - t_start
    return {
        'elapsed': elapsed,
        'alpha_min':  float(alpha_out.min()),
        'alpha_max':  float(alpha_out.max()),
        'alpha_mean': float(alpha_out.mean()),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description='CorridorKey MLX sequence processor')
    ap.add_argument('first_frame',          type=Path,
                    help='Path to first frame of input sequence')
    ap.add_argument('--garbage-matte',      type=Path, default=None,
                    help='First frame of garbage matte sequence (or single static matte with --static-matte)')
    ap.add_argument('--static-matte',       action='store_true',
                    help='Use a single matte file for all frames (no per-frame lookup)')
    ap.add_argument('--out-dir',            type=Path, required=True,
                    help='Output directory (created if needed)')
    ap.add_argument('--start',              type=int,  default=None,
                    help='First frame number to process (default: first detected)')
    ap.add_argument('--end',                type=int,  default=None,
                    help='Last frame number to process (default: last detected)')
    ap.add_argument('--model',              type=Path,
                    default=Path('/Users/cnoellert/ComfyUI/models/corridorkey/CorridorKey_v1.0.pth'))
    ap.add_argument('--despill-strength',   type=float, default=1.0)
    ap.add_argument('--trimap-radius',  type=int,   default=40,
                    help='Erode+dilate radius in native px (0=disable, default 40)')
    ap.add_argument('--no-despeckle',       action='store_true')
    ap.add_argument('--despeckle-size',     type=int,   default=400)
    ap.add_argument('--gm-dilation',        type=int,   default=15)
    ap.add_argument('--quantize',           choices=['int8'], default=None)
    ap.add_argument('--skip-existing',      action='store_true',
                    help='Skip frames whose _key.exr already exists in out-dir')
    ap.add_argument('--on-error',           choices=['skip', 'abort'], default='skip',
                    help='What to do if a frame fails (default: skip and continue)')
    args = ap.parse_args()

    first_frame = args.first_frame.expanduser().resolve()
    out_dir     = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- Detect sequence ---------------------------------------------------
    print(f"[seq] Scanning sequence from {first_frame.name} …")
    all_frames = _detect_sequence(first_frame)
    if not all_frames:
        sys.exit(f"[seq] No frames found matching {first_frame}")

    # Apply start/end filter
    def _fnum(p): return _parse_frame_number(p)[1]
    start = args.start if args.start is not None else _fnum(all_frames[0])
    end   = args.end   if args.end   is not None else _fnum(all_frames[-1])
    frames = [p for p in all_frames if start <= _fnum(p) <= end]

    if not frames:
        sys.exit(f"[seq] No frames in range {start}–{end}")

    print(f"[seq] {len(frames)} frames  ({_fnum(frames[0])}–{_fnum(frames[-1])})")

    # ---- Garbage matte setup -----------------------------------------------
    gm_first  = args.garbage_matte.expanduser().resolve() if args.garbage_matte else None
    static_gm = args.static_matte

    if gm_first:
        if static_gm:
            print(f"[seq] Static matte: {gm_first.name}")
        else:
            print(f"[seq] Per-frame matte sequence starting: {gm_first.name}")

    # ---- Load model (once) -------------------------------------------------
    print(f"[seq] Loading model …")
    t0 = time.time()
    model, n_weights = _load_model(
        args.model.expanduser().resolve(), args.quantize
    )
    print(f"[seq] Model ready ({n_weights} tensors, {time.time()-t0:.1f}s)")
    print()

    # ---- Process -----------------------------------------------------------
    errors   = []
    skipped  = 0
    t_seq    = time.time()

    for idx, frame_path in enumerate(frames, 1):
        frame_num  = _fnum(frame_path)
        stem       = frame_path.stem
        _m2 = re.match(r'''(.*?)[._](\d+)$''', frame_path.stem)
        _b2, _ft2 = (_m2.group(1), _m2.group(2)) if _m2 else (frame_path.stem, '')
        key_output = (out_dir / f"{_b2}_key.{_ft2}.exr") if _ft2 else (out_dir / f"{_b2}_key.exr")

        # Skip existing
        if args.skip_existing and key_output.exists():
            print(f"  [{idx:>4}/{len(frames)}] {stem}  SKIP (exists)")
            skipped += 1
            continue

        # Resolve matte path for this frame
        if gm_first is None:
            matte_path = None
        elif static_gm:
            matte_path = gm_first
        else:
            matte_path = _matte_path_for_frame(gm_first, frame_num)

        try:
            stats = _process_frame(
                model, frame_path, matte_path, out_dir,
                despill_strength=args.despill_strength,
                despeckle=not args.no_despeckle,
                despeckle_size=args.despeckle_size,
                gm_dilation=args.gm_dilation,
                trimap_radius=args.trimap_radius,
            )
            elapsed   = stats['elapsed']
            remaining = (len(frames) - idx) * elapsed
            print(
                f"  [{idx:>4}/{len(frames)}] {stem}"
                f"  {elapsed:.1f}s"
                f"  α=[{stats['alpha_min']:.3f},{stats['alpha_max']:.3f}]"
                f"  mean={stats['alpha_mean']:.3f}"
                f"  ETA {remaining/60:.1f}min"
            )
        except Exception as exc:
            msg = f"  [{idx:>4}/{len(frames)}] {stem}  ERROR: {exc}"
            print(msg)
            errors.append((frame_path.name, str(exc)))
            if args.on_error == 'abort':
                print("[seq] Aborting on first error (--on-error=abort)")
                break

    # ---- Summary -----------------------------------------------------------
    total_time = time.time() - t_seq
    processed  = idx - len(errors) - skipped
    print()
    print(f"[seq] Done.  {processed} processed, {skipped} skipped, {len(errors)} errors  "
          f"in {total_time:.1f}s  ({total_time/max(processed,1):.1f}s/frame avg)")
    if errors:
        print(f"[seq] Errors:")
        for name, err in errors:
            print(f"  {name}: {err}")


if __name__ == '__main__':
    main()
