# CorridorKey PyBox for Flame

Runs CorridorKey MLX inference as a native batch compositing node in Autodesk Flame.

## Files

| File | Purpose |
|------|---------|
| `corridorkey_pybox.py` | PyBox handler — runs in Flame's embedded Python |
| `corridorkey_daemon.py` | Inference server — runs in corridorkey-mlx conda env |

## Requirements

- Autodesk Flame 2025+ (macOS, Apple Silicon)
- `corridorkey-mlx` conda env with MLX 0.31+ installed
- CorridorKey weights: `CorridorKey_v1.0.mlx.npz`

## Install

1. Copy the `pybox/` folder somewhere Flame can reach it, e.g.:
   ```
   ~/flame/pybox/corridorkey/
   ```

2. In Flame batch schematic:
   - Add node → PyBox
   - Point it at `corridorkey_pybox.py`

## Node Inputs / Outputs

| Socket | Type | Description |
|--------|------|-------------|
| Front (in 0) | EXR | Green screen plate |
| Matte (in 1) | EXR | Alpha hint / garbage matte |
| Result (out 0) | EXR | Foreground (straight RGBA, sRGB) |
| OutMatte (out 1) | EXR | Alpha matte (single channel) |

## UI Controls

| Control | Default | Description |
|---------|---------|-------------|
| Weights | ~/ComfyUI/models/corridorkey/CorridorKey_v1.0.mlx.npz | Path to .mlx.npz weights |
| Despill Strength | 1.0 | Green spill removal (0=off) |
| Input is sRGB | ON | Enable for REC709 footage; off for scene-linear EXR |
| Despeckle | OFF | Removes isolated alpha artifacts; degrades hair edges |
| Reprocess | OFF | Force re-inference on current frame after param changes |

## How It Works

On node load (`initialize`), the handler spawns `corridorkey_daemon.py` as a
background process in the `corridorkey-mlx` conda env. The daemon loads the
GreenFormer model (~2s warmup) then waits on a named pipe for work.

On each frame render (`execute`), the handler writes inference params to the
command FIFO and blocks on the done FIFO. The daemon reads the plate and matte
EXRs that Flame wrote to the input socket paths, runs inference, writes FG +
alpha EXRs to the output socket paths, and signals done.

Model loads once — per-frame cost is inference time only (~3s on M4 Max).

## Logs

Daemon stdout/stderr: `/tmp/corridorkey_daemon.log`
