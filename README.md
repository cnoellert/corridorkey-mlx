# corridorkey-flame

Flame PyBox integration for [CorridorKey](https://github.com/cnoellert/comfyui-corridorkey) — a neural green screen keyer based on GreenFormer/Hiera.

Runs on **macOS Apple Silicon** (MLX) and **Linux CUDA** (Rocky Linux / Ubuntu). Platform is detected automatically.

---

## Requirements

| Platform | Hardware | Software |
|----------|----------|----------|
| macOS | Apple Silicon (M1–M4) | Miniconda, Flame 2025+ |
| Linux | NVIDIA GPU (RTX/A-series) | Miniconda, CUDA 12.x, Flame 2025+ |

> **Linux note:** Do not run ComfyUI or other GPU-heavy processes alongside Flame. The daemon uses CPU offload (model lives in system RAM, moves to GPU per-frame) but Flame itself holds ~20GB of VRAM on a 24GB card.

---

## Installation

### 1. Clone the repo

```bash
git clone https://github.com/cnoellert/corridorkey-flame.git
cd corridorkey-flame
```

### 2. Run the installer

```bash
bash install.sh
```

The installer handles everything automatically:
- Detects platform (macOS → MLX env, Linux → CUDA env)
- Creates the conda environment (`corridorkey-mlx` or `corridorkey-cuda`)
- Installs Python dependencies (auto-detects CUDA version on Linux)
- Creates `/opt/corridorkey/{models,pybox,reference}/`
- Copies pybox and reference inference code into place
- Downloads model weights from GitHub Releases (~380MB)

If you already have weights locally, skip the download:

```bash
bash install.sh --weights /path/to/CorridorKey_v1.0.pth
```

### 3. Add to Flame

In Flame Batch, add a **PyBox** node and point it at:

```
/opt/corridorkey/pybox/corridorkey_pybox.py
```

---

## Clean Reinstall

To start fresh on any machine:

```bash
rm -rf /opt/corridorkey /opt/corridorkey-flame
git clone https://github.com/cnoellert/corridorkey-flame.git /opt/corridorkey-flame
cd /opt/corridorkey-flame
bash install.sh
```

---

## Updating

```bash
cd /opt/corridorkey-flame
git pull
bash install.sh
```

The installer skips steps that are already complete (existing conda env, existing weights).

---

## Inputs / Outputs

| Pin | Description |
|-----|-------------|
| `IN_PLATE` | RGB plate (EXR, scene-linear or sRGB) |
| `IN_MATTE` | Rough matte / holdout mask (EXR) |
| `OUT_FG` | Keyed foreground RGBA (EXR) |
| `OUT_ALPHA` | Alpha channel RGB (EXR) |

---

## PyBox Parameters

**Model page**
- `Weights` — path to model weights file
- `Quantized` — enable quantized inference (macOS only, reduces memory)

**Settings page**
- `Add sRGB Gamma` — enable if input is scene-linear (converts to sRGB before inference, back to linear after)
- `Despill Strength` — green spill suppression (0–1)
- `Despeckle` — minimum alpha island area to remove (0–2000 px²)

---

## Troubleshooting

**`EnvironmentNameNotFound: corridorkey-mlx`** — Old pybox file installed. Run `git pull && bash install.sh`.

**`can't open file '/opt/corridorkey/pybox/corridorkey_daemon_cuda.py'`** — `/opt/corridorkey/` was not created. Run `bash install.sh` from the repo directory.

**`CUDA out of memory`** — ComfyUI or another GPU-heavy process is running alongside Flame. Kill it and retry. Check with `ps aux | grep -i comfy`.

**Daemon not starting** — Check `/tmp/corridorkey_daemon.log` for the full error.

**Frames not updating** — Check `/tmp/corridorkey_ready` exists (daemon loaded). If not, the model is still loading — give it 30–60 seconds on first run.

**Force daemon restart** — Run:
```bash
pkill -f corridorkey_daemon_cuda   # or corridorkey_daemon_mlx on Mac
rm -f /tmp/corridorkey_params.json.* /tmp/corridorkey_ready
```
Then trigger a frame in Flame.
