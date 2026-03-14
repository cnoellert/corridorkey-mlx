"""smooth_sequence.py  —  Temporal median/mean smoothing for CorridorKey alpha output.

GreenFormer is a single-frame model with no temporal awareness.  On sequences the
output alpha can flicker at edge pixels because:
  - The trimap uncertain band shifts slightly frame-to-frame as the subject moves
  - The model solves each frame independently
  - The 2.8× upsample from 2048→6K amplifies small per-pixel variance

This script reads a completed alpha sequence, applies a temporal rolling window
filter, then rebuilds the _key (premult) EXR from the smoothed alpha + original
plate colours from the _fg (straight) EXR.

Usage
-----
python smooth_sequence.py \\
    '/path/to/output/shot_alpha.0991.exr' \\
    [--window 3]           # frames either side (default 3 → 7-frame window)
    [--mode median]        # median (default) or mean
    [--out-dir /path]      # defaults to same dir as input, suffix _smooth

The script is non-destructive: original EXRs are never overwritten.
Outputs are named  shot_alpha_smooth.####.exr  /  shot_key_smooth.####.exr
"""

import argparse, sys, re, time
from pathlib import Path

import numpy as np
import OpenEXR, Imath


# ---------------------------------------------------------------------------
# EXR helpers
# ---------------------------------------------------------------------------

def _read_alpha_exr(path: Path) -> np.ndarray:
    """Return [H, W] float32 from single-channel EXR (channel Y or first available)."""
    f   = OpenEXR.InputFile(str(path))
    dw  = f.header()['dataWindow']
    W   = dw.max.x - dw.min.x + 1
    H   = dw.max.y - dw.min.y + 1
    pt  = Imath.PixelType(Imath.PixelType.FLOAT)
    ch  = list(f.header()['channels'].keys())[0]
    arr = np.frombuffer(f.channel(ch, pt), dtype=np.float32).reshape(H, W)
    return arr


def _read_rgba_exr(path: Path) -> np.ndarray:
    """Return [H, W, 4] float32 RGBA EXR."""
    f   = OpenEXR.InputFile(str(path))
    dw  = f.header()['dataWindow']
    W   = dw.max.x - dw.min.x + 1
    H   = dw.max.y - dw.min.y + 1
    pt  = Imath.PixelType(Imath.PixelType.FLOAT)
    channels = ['R', 'G', 'B', 'A']
    planes = [np.frombuffer(f.channel(c, pt), dtype=np.float32).reshape(H, W)
              for c in channels]
    return np.stack(planes, axis=-1)


def _read_exr_compression(path: Path) -> dict:
    f = OpenEXR.InputFile(str(path))
    h = f.header()
    return {
        'compression':         h.get('compression'),
        'dwaCompressionLevel': h.get('dwaCompressionLevel'),
    }


def _write_alpha_exr(path: Path, data: np.ndarray, compression: dict) -> None:
    """data: [H, W] float32."""
    H, W = data.shape
    header = OpenEXR.Header(W, H)
    if compression.get('compression') is not None:
        header['compression'] = compression['compression']
        lvl = compression.get('dwaCompressionLevel')
        if lvl is not None:
            header['dwaCompressionLevel'] = float(lvl)
    header['channels'] = {'Y': Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT))}
    f = OpenEXR.OutputFile(str(path), header)
    f.writePixels({'Y': data.tobytes()})
    f.close()


def _write_rgba_exr(path: Path, data: np.ndarray, compression: dict) -> None:
    """data: [H, W, 4] float32."""
    H, W, _ = data.shape
    header = OpenEXR.Header(W, H)
    if compression.get('compression') is not None:
        header['compression'] = compression['compression']
        lvl = compression.get('dwaCompressionLevel')
        if lvl is not None:
            header['dwaCompressionLevel'] = float(lvl)
    header['channels'] = {c: Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT))
                          for c in ['R', 'G', 'B', 'A']}
    f = OpenEXR.OutputFile(str(path), header)
    f.writePixels({c: data[:, :, i].tobytes() for i, c in enumerate(['R', 'G', 'B', 'A'])})
    f.close()


# ---------------------------------------------------------------------------
# Frame discovery
# ---------------------------------------------------------------------------

def _parse_frame_number(path: Path) -> int | None:
    m = re.search(r'\.(\d+)\.exr$', path.name, re.IGNORECASE)
    return int(m.group(1)) if m else None


def _discover_alpha_sequence(first_frame: Path) -> list[Path]:
    """Given the first *_alpha.####.exr, find all frames in the same dir."""
    m = re.match(r'^(.*_alpha)\.\d+$', first_frame.stem)
    if not m:
        sys.exit(f'ERROR: expected name like shot_alpha.0991.exr, got {first_frame.name}')
    base   = m.group(1)
    frames = sorted(
        first_frame.parent.glob(f'{base}.*.exr'),
        key=lambda p: _parse_frame_number(p) or 0,
    )
    return frames


def _fg_path_for_alpha(alpha_path: Path) -> Path:
    """DC_0190_raw_L02_alpha.0991.exr → DC_0190_raw_L02_fg.0991.exr"""
    name = alpha_path.name.replace('_alpha.', '_fg.')
    return alpha_path.parent / name


def _out_name(alpha_path: Path, suffix: str, out_dir: Path) -> Path:
    """shot_alpha.0991.exr → out_dir/shot_{suffix}.0991.exr"""
    m = re.match(r'^(.*)_alpha(\.\d+\.exr)$', alpha_path.name)
    if not m:
        sys.exit(f'Cannot derive output name from {alpha_path.name}')
    return out_dir / f'{m.group(1)}_{suffix}{m.group(2)}'


# ---------------------------------------------------------------------------
# Smoothing
# ---------------------------------------------------------------------------

def _smooth_alpha_sequence(
    alpha_paths: list[Path],
    window: int = 3,
    mode: str = 'median',
    out_dir: Path | None = None,
    skip_existing: bool = False,
) -> None:
    """
    For each frame i, gather frames [i-window .. i+window] (clamped to sequence
    bounds), stack their alphas, apply median or mean along the time axis,
    and write smoothed alpha + rebuilt premult key EXR.
    """
    n = len(alpha_paths)
    print(f'[smooth] {n} frames  window={window}  mode={mode}')

    compression = _read_exr_compression(alpha_paths[0])

    # Lazy frame cache — only keep what the current window needs
    cache: dict[int, np.ndarray] = {}

    def _load(idx: int) -> np.ndarray:
        if idx not in cache:
            cache[idx] = _read_alpha_exr(alpha_paths[idx])
        return cache[idx]

    t0 = time.time()
    for i, alpha_path in enumerate(alpha_paths):
        frame_num = _parse_frame_number(alpha_path)

        # Output paths
        dest_alpha = _out_name(alpha_path, 'alpha_smooth', out_dir)
        dest_key   = _out_name(alpha_path, 'key_smooth',   out_dir)

        if skip_existing and dest_alpha.exists() and dest_key.exists():
            print(f'  [{i+1:4d}/{n}] frame {frame_num}  SKIP')
            continue

        # Gather temporal neighbourhood
        lo = max(0,   i - window)
        hi = min(n-1, i + window)
        stack = np.stack([_load(j) for j in range(lo, hi+1)], axis=0)  # [T, H, W]

        if mode == 'median':
            smooth_alpha = np.median(stack, axis=0).astype(np.float32)
        else:
            smooth_alpha = stack.mean(axis=0).astype(np.float32)

        # Evict frames no longer needed
        evict_before = i - window
        for k in list(cache.keys()):
            if k < evict_before:
                del cache[k]

        # Write smoothed alpha
        _write_alpha_exr(dest_alpha, smooth_alpha, compression)

        # Rebuild premult key: multiply original plate RGB (from fg) by smooth alpha
        fg_path = _fg_path_for_alpha(alpha_path)
        if fg_path.exists():
            fg = _read_rgba_exr(fg_path)                   # [H,W,4] straight RGBA
            rgb_straight = fg[:, :, :3]
            key = np.concatenate(
                [rgb_straight * smooth_alpha[:, :, None], smooth_alpha[:, :, None]],
                axis=-1,
            ).astype(np.float32)
            _write_rgba_exr(dest_key, key, compression)
        else:
            print(f'    WARNING: _fg EXR not found at {fg_path.name}, skipping key rebuild')

        elapsed = time.time() - t0
        fps     = (i+1) / elapsed
        eta     = (n - i - 1) / fps / 60 if fps > 0 else 0
        print(f'  [{i+1:4d}/{n}] frame {frame_num}  {elapsed/(i+1):.2f}s/frame  ETA {eta:.1f}min')

    print(f'[smooth] Done.  {n} frames in {time.time()-t0:.1f}s')


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description='Temporal alpha smoother for CorridorKey output')
    ap.add_argument('first_alpha',      type=Path,
                    help='Path to first *_alpha.####.exr in the output sequence')
    ap.add_argument('--window',         type=int, default=3,
                    help='Frames either side for smoothing window (default 3 → 7-frame window)')
    ap.add_argument('--mode',           choices=['median', 'mean'], default='median',
                    help='Aggregation mode (default: median)')
    ap.add_argument('--out-dir',        type=Path, default=None,
                    help='Output directory (default: same as input)')
    ap.add_argument('--skip-existing',  action='store_true')
    args = ap.parse_args()

    if not args.first_alpha.exists():
        sys.exit(f'ERROR: {args.first_alpha} not found')

    out_dir = args.out_dir or args.first_alpha.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    alpha_paths = _discover_alpha_sequence(args.first_alpha)
    if not alpha_paths:
        sys.exit('ERROR: no alpha frames found')

    print(f'[smooth] Discovered {len(alpha_paths)} frames')
    print(f'[smooth] Output → {out_dir}')

    _smooth_alpha_sequence(
        alpha_paths,
        window=args.window,
        mode=args.mode,
        out_dir=out_dir,
        skip_existing=args.skip_existing,
    )


if __name__ == '__main__':
    main()
