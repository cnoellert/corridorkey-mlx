"""
corridorkey_daemon.py  —  Inference server for the CorridorKey Flame PyBox

Runs in the corridorkey-mlx conda env. Loads the GreenFormer model once,
then serves frames on demand via FIFO IPC from corridorkey_pybox.py.

Not intended to be run directly — spawned by the PyBox handler on initialize().
Log output goes to /tmp/corridorkey_daemon.log.
"""

import argparse
import json
import sys
import os
from pathlib import Path

import numpy as np

# Ensure the repo root is importable regardless of cwd
_HERE = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(_HERE))

import mlx.core as mx
from model import GreenFormer
from test_frame import (
    infer_frame,
    _read_exr_rgb,
    _read_exr_mask,
    _write_exr,
)

LOSSLESS = {"compression": None, "dwaCompressionLevel": None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights",   required=True)
    ap.add_argument("--cmd-fifo",  required=True)
    ap.add_argument("--done-fifo", required=True)
    ap.add_argument("--in-plate",  required=True)
    ap.add_argument("--in-matte",  required=True)
    ap.add_argument("--out-fg",    required=True)
    ap.add_argument("--out-alpha", required=True)
    args = ap.parse_args()

    weights_path = Path(args.weights).expanduser().resolve()
    cmd_fifo  = args.cmd_fifo
    done_fifo = args.done_fifo
    in_plate  = Path(args.in_plate)
    in_matte  = Path(args.in_matte)
    out_fg    = Path(args.out_fg)
    out_alpha = Path(args.out_alpha)

    # ------------------------------------------------------------------
    # Load model once
    # ------------------------------------------------------------------
    print(f"[daemon] Loading GreenFormer from {weights_path} …", flush=True)
    model = GreenFormer(img_size=2048)
    weights = mx.load(str(weights_path))
    model.load_weights(list(weights.items()))
    mx.eval(model.parameters())
    print("[daemon] Model ready", flush=True)

    # ------------------------------------------------------------------
    # Serve frames — opening cmd_fifo for read blocks until handler writes,
    # which also serves as the "ready" signal the handler polls for via lsof.
    # ------------------------------------------------------------------
    while True:
        with open(cmd_fifo, "r") as f:
            raw = f.read().strip()

        if not raw:
            continue

        params = json.loads(raw)

        if params.get("quit"):
            print("[daemon] Quit signal — shutting down", flush=True)
            break

        frame = params.get("frame", "?")
        print(f"[daemon] Frame {frame} — inferring …", flush=True)

        done_ok = False
        try:
            # Read inputs written by Flame to the socket paths
            rgb_linear = _read_exr_rgb(in_plate)           # [H, W, 3] float32 linear
            H, W       = rgb_linear.shape[:2]
            mask       = _read_exr_mask(in_matte, H, W)    # [H, W, 1] float32

            result, fg_straight, _ = infer_frame(
                model,
                rgb_linear,
                mask,
                input_is_srgb    = params.get("input_is_srgb",    True),
                despill_strength = params.get("despill_strength",  1.0),
                despeckle        = params.get("despeckle",         False),
            )

            # result is premult RGBA [H, W, 4] — extract alpha channel
            alpha = result[:, :, 3:4].astype(np.float32)
            _write_exr(out_alpha, alpha,                       compression=LOSSLESS)
            _write_exr(out_fg,    fg_straight.astype(np.float32), compression=LOSSLESS)

            mx.clear_cache()
            print(f"[daemon] Frame {frame} done", flush=True)
            done_ok = True

        except Exception as e:
            print(f"[daemon] ERROR on frame {frame}: {e}", flush=True, file=sys.stderr)
            import traceback; traceback.print_exc()

        finally:
            # Always signal done so handler doesn't block forever
            try:
                with open(done_fifo, "w") as f:
                    f.write("ok\n" if done_ok else "err\n")
            except Exception:
                pass


if __name__ == "__main__":
    main()
