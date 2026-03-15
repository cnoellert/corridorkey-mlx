"""
corridorkey_daemon.py  —  Inference server for the CorridorKey Flame PyBox

File-based IPC (no FIFOs):
  Polls for TRIGGER file written by handler, reads params from PARAMS_FILE,
  runs inference, writes outputs, drops DONE sentinel.

Spawned once by corridorkey_pybox.py. Logs to /tmp/corridorkey_daemon.log.
"""

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np

# Repo root on sys.path.
# __file__ resolves correctly here (daemon is spawned directly, not copied by Flame).
# But hardcode for safety in case install layout changes.
_REPO_ROOT = Path(os.path.expanduser("~/Documents/GitHub/corridorkey-mlx"))
sys.path.insert(0, str(_REPO_ROOT))

import mlx.core as mx
from model import GreenFormer
from test_frame import infer_frame, _read_exr_rgb, _read_exr_mask, _write_exr

LOSSLESS     = {"compression": None, "dwaCompressionLevel": None}
POLL_INTERVAL = 0.1   # seconds between trigger polls


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights",   required=True)
    ap.add_argument("--in-plate",  required=True)
    ap.add_argument("--in-matte",  required=True)
    ap.add_argument("--out-fg",    required=True)
    ap.add_argument("--out-alpha", required=True)
    ap.add_argument("--params",    required=True)
    ap.add_argument("--trigger",   required=True)
    ap.add_argument("--ready",     required=True)
    ap.add_argument("--done",      required=True)
    ap.add_argument("--error",     required=True)
    args = ap.parse_args()

    weights_path = Path(args.weights).expanduser().resolve()
    in_plate     = Path(args.in_plate)
    in_matte     = Path(args.in_matte)
    out_fg       = Path(args.out_fg)
    out_alpha    = Path(args.out_alpha)
    params_file  = args.params
    trigger      = args.trigger
    ready        = args.ready
    done         = args.done
    error        = args.error

    # ------------------------------------------------------------------
    # Load model once
    # ------------------------------------------------------------------
    print(f"[daemon] Loading GreenFormer from {weights_path} …", flush=True)
    model = GreenFormer()
    weights = mx.load(str(weights_path))
    model.load_weights(list(weights.items()))
    mx.eval(model.parameters())
    print("[daemon] Model ready", flush=True)

    # Signal handler that we're ready
    open(ready, "w").close()

    # ------------------------------------------------------------------
    # Serve frames — poll for trigger file
    # ------------------------------------------------------------------
    while True:
        if not os.path.exists(trigger):
            time.sleep(POLL_INTERVAL)
            continue

        # Consume trigger immediately to avoid double-fire
        try:
            os.unlink(trigger)
        except OSError:
            pass

        # Read params
        try:
            params = json.loads(open(params_file).read())
        except Exception as e:
            open(error, "w").write(f"Could not read params: {e}")
            continue

        if params.get("quit"):
            print("[daemon] Quit signal — shutting down", flush=True)
            break

        frame = params.get("frame", "?")
        print(f"[daemon] Frame {frame} — inferring …", flush=True)

        try:
            rgb_linear = _read_exr_rgb(in_plate)
            H, W       = rgb_linear.shape[:2]
            mask       = _read_exr_mask(in_matte, H, W)

            result, fg_straight, _ = infer_frame(
                model,
                rgb_linear,
                mask,
                input_is_srgb    = params.get("input_is_srgb",    True),
                despill_strength = params.get("despill_strength",  1.0),
                despeckle        = params.get("despeckle",         False),
            )

            alpha = result[:, :, 3:4].astype(np.float32)
            _write_exr(out_alpha, alpha,                           compression=LOSSLESS)
            _write_exr(out_fg,    fg_straight.astype(np.float32), compression=LOSSLESS)

            mx.clear_cache()
            print(f"[daemon] Frame {frame} done", flush=True)

            # Signal done
            open(done, "w").close()

        except Exception as e:
            msg = f"Frame {frame}: {e}\n{traceback.format_exc()}"
            print(f"[daemon] ERROR {msg}", flush=True)
            open(error, "w").write(msg)


if __name__ == "__main__":
    main()
