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
_REPO_ROOT = Path("/opt/corridorkey/mlx")
sys.path.insert(0, str(_REPO_ROOT))

import mlx.core as mx
from model import GreenFormer
from test_frame import infer_frame, _read_exr_rgb, _read_exr_mask, _write_exr, _srgb_to_linear

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
    ap.add_argument("--quantized", action="store_true", default=False,
                    help="Dequantize int8 weights on load")
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
    raw = {k: v for k, v in dict(mx.load(str(weights_path))).items()
           if not k.startswith("__")}

    if args.quantized:
        # Dequantize int8 weights: quantize.py stores
        #   foo.weight  (int8)  + foo.weight_scale (float32 [out_ch])
        # Reconstruct fp32: w_fp32 = w_int8.astype(float32) * scale
        scale_keys = {k for k in raw if k.endswith("_scale")}
        weights = {}
        for k, v in raw.items():
            if k in scale_keys:
                continue  # consumed alongside its weight
            sk = k + "_scale"
            if sk in raw:
                w = v.astype(mx.float32)
                s = raw[sk]
                for _ in range(w.ndim - 1):
                    s = s[..., None]
                weights[k] = w * s
            else:
                weights[k] = v
        print(f"[daemon] Dequantized {len(scale_keys)} int8 weight tensors", flush=True)
    else:
        weights = raw

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

        # Debounce: wait until no new trigger has arrived for 300ms.
        # Unlinking trigger creates a window where a new one can appear;
        # the naive sleep+drain loop gets stuck if Flame scrubs faster
        # than the sleep interval. This "idle for N ms" approach correctly
        # waits for scrubbing to settle regardless of scrub speed.
        _last_trigger = time.time()
        while True:
            if os.path.exists(trigger):
                try: os.unlink(trigger)
                except OSError: pass
                _last_trigger = time.time()
            elif time.time() - _last_trigger > 0.3:
                break
            time.sleep(0.05)

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

            despeckle_val  = params.get("despeckle", 0.0)
            add_srgb_gamma = params.get("add_srgb_gamma", False)

            # add_srgb_gamma=True  → input is linear; encode to sRGB before model,
            #                        decode FG back to linear after inference.
            # add_srgb_gamma=False → input is already sRGB; pass through as-is.
            # infer_frame's input_is_srgb means "don't encode — it's already sRGB",
            # so it is the logical inverse of add_srgb_gamma.
            result, fg_straight, _ = infer_frame(
                model,
                rgb_linear,
                mask,
                input_is_srgb    = not add_srgb_gamma,
                despill_strength = params.get("despill_strength",  1.0),
                despeckle        = despeckle_val > 0.0,
                despeckle_size   = int(despeckle_val) if despeckle_val > 0.0 else 400,
            )

            alpha = np.ascontiguousarray(result[:, :, 3], dtype=np.float32)  # [H, W] contiguous
            fg    = fg_straight.astype(np.float32)         # [H, W, 4]

            # Decode FG back to scene-linear if we encoded on the way in.
            # Alpha is always linear (model output is 0-1 matte).
            if add_srgb_gamma:
                fg[:, :, :3] = _srgb_to_linear(fg[:, :, :3])

            # Atomic writes -- write to .tmp then rename so Flame never
            # sees a partial file when it wakes on the DONE sentinel.
            alpha_rgb = np.stack([alpha, alpha, alpha], axis=-1)
            _write_exr(str(out_fg)    + ".tmp", fg,        compression=LOSSLESS)
            _write_exr(str(out_alpha) + ".tmp", alpha_rgb, compression=LOSSLESS)
            os.rename(str(out_fg)    + ".tmp", str(out_fg))
            os.rename(str(out_alpha) + ".tmp", str(out_alpha))

            mx.clear_cache()
            print(f"[daemon] Frame {frame} done", flush=True)

            # Write done_frame so handler knows which frame OUT_FG belongs to
            open(params_file + ".done_frame", "w").write(str(frame))
            # Signal done
            open(done, "w").close()

        except Exception as e:
            msg = f"Frame {frame}: {e}\n{traceback.format_exc()}"
            print(f"[daemon] ERROR {msg}", flush=True)
            open(error, "w").write(msg)


if __name__ == "__main__":
    main()
