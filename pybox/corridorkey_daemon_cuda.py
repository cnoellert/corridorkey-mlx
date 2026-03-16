"""
corridorkey_daemon_cuda.py  --  PyTorch/CUDA inference server for CorridorKey Flame PyBox

Identical IPC contract to corridorkey_daemon_mlx.py:
  Polls for TRIGGER sentinel, reads params JSON, runs inference,
  writes output EXRs, drops DONE sentinel.

Backend: CorridorKeyEngine + OptimizedEngine from /opt/corridorkey/reference/
Weights:  /opt/corridorkey/models/CorridorKey_v1.0.pth
Device:   CUDA if available, CPU fallback
"""

# Must be set before any CUDA/torch import -- once CUDA is initialized
# the allocator config is frozen and env var changes have no effect.
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np

# Reference inference code lives at /opt/corridorkey/reference/ on all platforms.
# Contains CorridorKeyModule/ and utils/ from the comfyui-corridorkey repo.
_REF_ROOT = Path("/opt/corridorkey/reference")
sys.path.insert(0, str(_REF_ROOT))

from utils.device import get_device, get_autocast_ctx, clear_cache
from utils.inference import OptimizedEngine, _srgb_to_linear

LOSSLESS      = {"compression": None, "dwaCompressionLevel": None}
POLL_INTERVAL = 0.1

# EXR helpers -- reuse same functions as MLX daemon
def _read_exr_rgb(path):
    import OpenEXR, Imath, array as arr
    f = OpenEXR.InputFile(str(path))
    dw = f.header()["dataWindow"]
    W = dw.max.x - dw.min.x + 1
    H = dw.max.y - dw.min.y + 1
    FLOAT = Imath.PixelType(Imath.PixelType.FLOAT)
    def ch(c):
        raw = arr.array("f", f.channel(c, FLOAT))
        return np.frombuffer(raw, dtype=np.float32).reshape(H, W)
    rgb = np.stack([ch("R"), ch("G"), ch("B")], axis=-1)
    f.close()
    return rgb

def _read_exr_mask(path, H, W):
    import OpenEXR, Imath, array as arr
    try:
        f = OpenEXR.InputFile(str(path))
        FLOAT = Imath.PixelType(Imath.PixelType.FLOAT)
        channels = list(f.header()["channels"].keys())
        c = "R" if "R" in channels else channels[0]
        raw = arr.array("f", f.channel(c, FLOAT))
        mask = np.frombuffer(raw, dtype=np.float32).reshape(H, W, 1)
        f.close()
        return mask
    except Exception:
        return np.zeros((H, W, 1), dtype=np.float32)

def _write_exr(path, img, compression=None):
    import OpenEXR, Imath
    if img.ndim == 2:
        img = img[:, :, np.newaxis]
    H, W, C = img.shape
    header = OpenEXR.Header(W, H)
    header["compression"] = Imath.Compression(Imath.Compression.NO_COMPRESSION)
    FLOAT = Imath.PixelType(Imath.PixelType.FLOAT)
    channel_names = ["R", "G", "B", "A"][:C]
    header["channels"] = {c: Imath.Channel(FLOAT) for c in channel_names}
    f = OpenEXR.OutputFile(str(path), header)
    f.writePixels({c: img[:, :, i].astype(np.float32).tobytes()
                   for i, c in enumerate(channel_names)})
    f.close()

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

    in_plate  = args.in_plate
    in_matte  = args.in_matte
    out_fg    = args.out_fg
    out_alpha = args.out_alpha
    params_f  = args.params
    trigger   = args.trigger
    ready     = args.ready
    done      = args.done
    error     = args.error

    # Load model
    device = get_device()
    print(f"[daemon] Loading CorridorKey on {device} from {args.weights} ...", flush=True)

    from CorridorKeyModule.inference_engine import CorridorKeyEngine
    import torch
    # Load model to CPU first -- move to CUDA only during inference.
    # Flame holds ~20GB of the A5000's 24GB permanently. Keeping the model
    # on CPU (system RAM) between frames avoids competing with Flame for VRAM.
    # ComfyUI uses the same pattern via comfy.model_management.
    import torch
    cpu_device = torch.device("cpu")
    engine = CorridorKeyEngine(
        checkpoint_path=args.weights,
        device=str(cpu_device),
        img_size=2048,
        use_refiner=True,
    )
    if device.type in ("cuda", "mps"):
        engine = OptimizedEngine(engine)
    if device.type == "cuda":
        engine.model = engine.model.half()
        # Prefer memory-efficient attention (O(N) memory vs O(N^2) for Hiera at 2048px).
        # Keep math_sdp=True as fallback so we never hit "no available kernel".
        import torch.backends.cuda as _cuda_backends
        _cuda_backends.enable_flash_sdp(True)
        _cuda_backends.enable_mem_efficient_sdp(True)
        _cuda_backends.enable_math_sdp(True)

    print(f"[daemon] Model ready on {device}", flush=True)
    open(ready, "w").close()

    # Poll loop
    while True:
        if not os.path.exists(trigger):
            time.sleep(POLL_INTERVAL)
            continue

        os.unlink(trigger)

        try:
            params = json.loads(open(params_f).read())
        except Exception:
            time.sleep(POLL_INTERVAL)
            continue

        if params.get("quit"):
            print("[daemon] Quit signal -- shutting down", flush=True)
            break

        frame          = params.get("frame", 0)
        add_srgb_gamma = params.get("add_srgb_gamma", False)
        despill        = params.get("despill_strength", 1.0)
        despeckle_val  = params.get("despeckle", 0.0)

        print(f"[daemon] Frame {frame} -- inferring ...", flush=True)

        try:
            rgb_linear = _read_exr_rgb(in_plate)
            H, W = rgb_linear.shape[:2]
            mask = _read_exr_mask(in_matte, H, W)

            import torch
            img_t  = torch.from_numpy(rgb_linear)
            mask_t = torch.from_numpy(mask[..., 0])

            # Move model to GPU just for this frame, back to CPU after.
            # Two-stage flush: sync+cache clear, then a brief sleep to let
            # the CUDA allocator consolidate fragmented blocks before the
            # 8GB Hiera attention allocation.
            if device.type == "cuda":
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                time.sleep(0.3)
                torch.cuda.empty_cache()
                engine.model         = engine.model.to(device)
                engine._engine.model = engine.model
                engine.device        = device
                engine._mean         = engine._mean.to(device)
                engine._std          = engine._std.to(device)

            result = None
            try:
                with torch.no_grad():
                    result = engine.process_frame_tensor(
                        img_t, mask_t,
                        input_is_linear  = add_srgb_gamma,
                        despill_strength = float(despill),
                        auto_despeckle   = despeckle_val > 0.0,
                        despeckle_size   = int(despeckle_val) if despeckle_val > 0.0 else 400,
                    )
            finally:
                # ALWAYS move model back to CPU -- even on error.
                # Without this the model stays on GPU permanently.
                if device.type == "cuda":
                    engine.model         = engine.model.to(cpu_device)
                    engine._engine.model = engine.model
                    engine.device        = cpu_device
                    engine._mean         = engine._mean.to(cpu_device)
                    engine._std          = engine._std.to(cpu_device)
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()

            if result is None:
                raise RuntimeError("Inference returned no result")

            # fg is sRGB straight [H,W,3], alpha is linear [H,W,1]
            fg_np    = result["fg"].numpy().astype(np.float32)
            alpha_np = result["alpha"].numpy().astype(np.float32)
            del result

            # Decode FG back to linear if we encoded on the way in
            if add_srgb_gamma:
                fg_t  = torch.from_numpy(fg_np)
                fg_np = _srgb_to_linear(fg_t).numpy().astype(np.float32)

            fg_rgba   = np.concatenate([fg_np, alpha_np], axis=-1)
            alpha_rgb = np.concatenate([alpha_np, alpha_np, alpha_np], axis=-1)

            # Atomic writes -- write to tmp then rename so Flame never
            # sees a partial file when it wakes on the DONE sentinel.
            _write_exr(out_fg    + ".tmp", fg_rgba)
            _write_exr(out_alpha + ".tmp", alpha_rgb)
            os.rename(out_fg    + ".tmp", out_fg)
            os.rename(out_alpha + ".tmp", out_alpha)

            print(f"[daemon] Frame {frame} done", flush=True)
            open(done, "w").close()

        except Exception as e:
            tb = traceback.format_exc()
            print(f"[daemon] ERROR Frame {frame}: {e}\n{tb}", flush=True)
            open(error, "w").write(str(e))


if __name__ == "__main__":
    main()
