# corridorkey-flame

Flame PyBox integration for [CorridorKey](https://github.com/cnoellert/comfyui-corridorkey) — a neural green screen keyer based on GreenFormer/Hiera.

Runs on **macOS Apple Silicon** (MLX) and **Linux CUDA** (Rocky Linux / Ubuntu). Platform is detected automatically.

▶️ **[Demo on YouTube](https://youtu.be/_P5dZQjD6hA)**

---

## Requirements

| Platform | Hardware | Software |
|----------|----------|----------|
| macOS | Apple Silicon (M1–M4) | **Miniconda**, Flame 2025+ |
| Linux | NVIDIA GPU (RTX/A-series, 16GB+ VRAM) | **Miniconda**, CUDA 12.x, Flame 2025+ |

**Miniconda is required** — the installer uses conda to manage Python environments.

- macOS: `brew install miniconda` or https://docs.conda.io/en/latest/miniconda.html
- Linux: `wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && bash Miniconda3-latest-Linux-x86_64.sh`

> **Linux note:** Do not run ComfyUI or other GPU-heavy processes alongside Flame. The daemon uses CPU offload (model lives in system RAM, moves to GPU per-frame) but Flame itself holds significant VRAM. Use `Img Size: 1024` if you have less than 24GB.

---

## Installation

### 1. Clone the repo

Clone anywhere you like:

```bash
git clone https://github.com/cnoellert/corridorkey-flame.git
cd corridorkey-flame
```

### 2. Run the installer

The installer writes to `/opt/corridorkey/`, which is owned by root. It uses `sudo` internally for the initial directory creation, then immediately transfers ownership to your user account so subsequent runs and file copies don't need `sudo`. **You will be prompted for your password on first install.**

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
sudo rm -rf /opt/corridorkey
cd /path/to/corridorkey-flame   # wherever you cloned the repo
git pull
bash install.sh
```

> The clone can live anywhere — `/opt/corridorkey` is the install target, not the repo location. `sudo rm` is needed because even though your user owns the directory, some platforms require it to remove from `/opt`.

---

## Updating

```bash
cd /path/to/corridorkey-flame   # wherever you cloned the repo
git pull
bash install.sh
```

The installer skips steps that are already complete (existing conda env, existing weights) and overwrites all code files in place. No `sudo` needed after the initial install since your user owns `/opt/corridorkey`.

> **Mac note:** Inference code lives in `/opt/corridorkey/mlx/`, not `/opt/corridorkey/pybox/`. The installer handles this correctly. If you ever need to copy a file manually, copy to `/opt/corridorkey/mlx/` on Mac and `/opt/corridorkey/reference/` on Linux.

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
- `Img Size` — inference resolution: `2048 (Full Quality)` or `1024 (Fast)`. Defaults to 1024 on Linux (Flame reserves most GPU VRAM) and 2048 on macOS (unified memory). 1024 is ~3x faster with minimal quality loss on clean commercial GS work. Changing this respawns the daemon (~10s reload).

**Settings page**
- `Add sRGB Gamma` — enable if input is scene-linear (converts to sRGB before inference, back to linear after)
- `Despill Strength` — green spill suppression (0–1)
- `Despeckle` — minimum alpha island area to remove (0–2000 px²)

---

## Troubleshooting

**`EnvironmentNameNotFound: corridorkey-mlx`** — Old pybox file installed. Run `git pull && bash install.sh`.

**`can't open file '/opt/corridorkey/pybox/corridorkey_daemon_cuda.py'`** — `/opt/corridorkey/` was not created. Run `bash install.sh` from the repo directory.

**`CUDA out of memory`** — Switch `Img Size` to `1024` in the Model tab, or flush GPU memory with `sudo fuser -k /dev/nvidia*` and retry at 2048. Check for other GPU processes with `nvidia-smi`.

**Daemon not starting** — Check `/tmp/corridorkey_daemon.log` for the full error.

**Frames not updating** — Check `/tmp/corridorkey_ready` exists (daemon loaded). If not, the model is still loading — give it 30–60 seconds on first run.

**Force daemon restart** — Run:
```bash
pkill -f corridorkey_daemon_cuda   # or corridorkey_daemon_mlx on Mac
rm -f /tmp/corridorkey_params.json.* /tmp/corridorkey_ready
```
Then trigger a frame in Flame.
