# CorridorKey PyBox — Architecture & Hard-Won Lessons

**Last updated:** 2026-03-16  
**Repo:** `cnoellert/corridorkey-flame`

---

## The One Thing You Must Understand

```python
if not self.is_processing():
    return
```

This single guard in `execute()` is why everything works. If you remove it, or
move any inference code above it, scrubbing will flood Flame's shell with errors
and the node will be unusable. Do not touch it without reading this document first.

---

## How Flame Calls a PyBox

Flame calls `execute()` **constantly** — on every scrub frame, every timeline
navigation event, every UI interaction. It is not a "run when I need output" callback.
It is more like a game loop tick that happens to also carry render requests.

The key distinction is `is_processing()`:

| Situation | `is_processing()` | What Flame wants |
|---|---|---|
| User scrubs timeline | `False` | Nothing — just moving the playhead |
| User clicks Result button | `True` | Actual rendered EXRs, right now |
| Background render / export | `True` | Actual rendered EXRs, right now |
| UI parameter change | `False` | Nothing — just acknowledge the change |
| Node first connects | `False` | Nothing — just initialize |

When `is_processing()` is `False`, `execute()` should return immediately.
No file I/O, no IPC, no blocking of any kind.

When `is_processing()` is `True`, Flame is **deliberately waiting** for
`execute()` to return. You can block for as long as inference takes.
There is no timeout that fires during a real render request.

---

## The Reference: BLG for Flame (FilmLight, 2017)

We discovered `is_processing()` by reading `BLG.py`, the PyBox from FilmLight's
BLG-for-Flame product. Their `execute()` does this at the top:

```python
if not self.is_processing():
    if self.FLAME_VERSION == 2019:
        self.set_dialog_msg('')
    return
```

Then, unconditionally below that guard, they call:

```python
response = urlopen(req, timeout=45)
```

A **45-second blocking HTTP request** — directly in `execute()`. This works
because `is_processing()` guarantees Flame is waiting. BLG has never had a
scrubbing problem because of this guard.

The BLG pattern also uses fixed socket paths (set once in `initialize()`),
not frame-stamped paths. Since `execute()` blocks until the result is ready,
Flame always reads the file after it has been written. There is no race.

---

## Why the Previous Async Approach Failed

Before `is_processing()` was known, the handler assumed it had to return fast
on every call. This created an unsolvable conflict:

**If `execute()` blocks:**  
Flame has a hard timeout during scrubbing. Blocking ~3 s causes Flame to
abort the PyBox process entirely ("PYBOX process aborted").

**If `execute()` returns immediately (async):**  
Flame checks whether the output EXR exists. It does not. Flame logs:
```
Unable to open file '/tmp/corridorkey_out_fg.exr': No such file or directory
Node "pybox1" must have output Result0
```
Flame retries. The daemon is still running. Flame retries again. The shell
floods. The daemon may finish, but by then the frame has changed.

Every attempted workaround failed for the same underlying reason:
`execute()` was being called during scrubbing, and the code was trying to
do real work on those calls.

### Specific failed approaches (do not re-try)

| Approach | Why it failed |
|---|---|
| Debounce in daemon (wait N ms before processing) | Handler writes triggers faster than the debounce window |
| Idle-for-300ms debounce | Daemon unlinks trigger; handler writes new one in the gap |
| INFERRING lockfile | Blocked first-frame render entirely |
| `done_frame` sentinel (check EXR belongs to current frame) | Helped stale results but didn't stop flooding |
| Delete EXRs before trigger (so Flame errors cleanly) | Flame retries faster than inference completes |
| `_send_frame()` blocking on every call | Flame timeout killed the PyBox during scrubbing |

---

## Current Architecture

```
Flame (main process)
  │
  │  calls execute() on every frame/tick
  │
  ▼
corridorkey_pybox.py
  ├── is_processing() == False  →  return immediately (scrubbing, UI ticks)
  │
  └── is_processing() == True   →  Flame is waiting
        │
        ├── ensure daemon running (spawn if needed, wait for READY sentinel)
        │
        ├── write params.json
        ├── write TRIGGER sentinel
        │
        └── poll DONE/ERROR — BLOCK until done (~3-8 s)
              │
              Flame reads OUT_FG / OUT_ALPHA  ← always present, always current frame

corridorkey_daemon_{mlx,cuda}.py  (background process)
  ├── loads model once at startup, writes READY
  └── poll loop:
        wait for TRIGGER
        read params.json
        run inference
        write out_fg.exr.tmp → rename to out_fg.exr  (atomic)
        write out_alpha.exr.tmp → rename to out_alpha.exr
        write DONE sentinel
```

### Why a daemon at all?

The model takes ~5-15 seconds to load (GreenFormer + Hiera weights).
Loading on every frame would be unusable. The daemon keeps the model warm
in GPU/MPS memory between frames.

On CUDA: model lives on CPU between frames to avoid competing with Flame
for the A5000's 24 GB VRAM. It is moved to CUDA just for inference, then
back to CPU in a `try/finally` block (guaranteed even on exception).

### Socket paths

Fixed paths set once in `initialize()`:

```
/tmp/corridorkey_in_plate.exr     ← Flame writes input here
/tmp/corridorkey_in_matte.exr     ← Flame writes matte here
/tmp/corridorkey_out_fg.exr       ← daemon writes FG here
/tmp/corridorkey_out_alpha.exr    ← daemon writes alpha here
```

Daemon writes atomically: `out_fg.exr.tmp` → `rename` → `out_fg.exr`.
Since `execute()` blocks until DONE, Flame never reads a partial file.

---

## IPC Sentinel Files

```
/tmp/corridorkey_trigger     created by handler → consumed by daemon (starts inference)
/tmp/corridorkey_ready       created by daemon → signals model loaded
/tmp/corridorkey_done        created by daemon → signals inference complete
/tmp/corridorkey_error       created by daemon on failure → contains error message
/tmp/corridorkey_params.json written by handler → read by daemon (frame params)
```

The daemon unlinks TRIGGER immediately when it picks it up. It writes DONE
only after the output EXRs have been atomically renamed into place. The handler
polls at 100 ms intervals until DONE or ERROR appears, then returns.

---

## Critical Rules (Do Not Violate)

### 1. `is_processing()` guard must be the first real work check in `execute()`

Only UI-state changes (weights path, quantized toggle) may be handled before it.
Everything involving file I/O, IPC, or blocking **must** be after it.

### 2. `try/finally` must wrap the actual CUDA inference call

```python
# CORRECT
try:
    result = engine.process_frame_tensor(...)
finally:
    engine.model = engine.model.to(cpu_device)
    torch.cuda.empty_cache()
```

If the `finally` is outside the loop or attached to a retry block, it will
not run if inference raises. The model stays on GPU permanently and VRAM
fills up.

### 3. No OOM retry loops on CUDA

Retrying on `torch.cuda.OutOfMemoryError` while model state is uncertain
causes a CUDA kernel panic and a hard system crash. If VRAM is exhausted,
log the error and let the user reduce resolution or restart.

### 4. `PYTORCH_CUDA_ALLOC_CONF` must be set before `import torch`

```python
# TOP OF FILE — before any torch import
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
```

CUDA's allocator is frozen at first import. Setting this env var anywhere
else (inside `main()`, after `import torch`) has no effect.

### 5. Do not add `max_split_size_mb` to CUDA allocator config

It causes large allocation spikes on every frame instead of reusing existing
blocks. The default allocator handles Hiera's attention allocation better
without it.

### 6. `pkill` is asynchronous — wait 0.5 s after killing the daemon

```python
os.system(f"pkill -f '{DAEMON_SCRIPT}'")
time.sleep(0.5)   # process must actually die before spawning replacement
_cleanup_sentinels()
```

Spawning before the old process dies results in two daemons racing on the
same trigger file.

### 7. Do not modify `reference/utils/inference.py`

This is the upstream CorridorKey inference code restored to its original
working state (commit 530e607). All attempts to add `channels_last` memory
format changes or explicit `float16` input casts made results worse, not
better. The original code handles this via PyTorch autocast. Leave it alone.

### 8. `--quantized` must be declared in CUDA daemon argparse

Even if it does nothing on CUDA, it must be in `ap.add_argument(...)`.
Undeclared args cause a silent `SystemExit` when the daemon spawns, which
looks like a model load failure.

---

## Weights & Install Layout

```
/opt/corridorkey/
├── models/
│   ├── CorridorKey_v1.0.mlx.npz    (Mac — MLX float32 or int8)
│   └── CorridorKey_v1.0.pth         (Linux — PyTorch float32)
├── mlx/                              (Mac only — GreenFormer MLX impl)
│   ├── model.py
│   ├── test_frame.py
│   └── ...
├── reference/                        (Linux only — original PyTorch impl)
│   ├── CorridorKeyModule/
│   └── utils/
│       └── inference.py              ← DO NOT MODIFY
└── pybox/
    ├── corridorkey_pybox.py
    ├── corridorkey_daemon_mlx.py
    └── corridorkey_daemon_cuda.py
```

---

## Reinstall / Update Commands

**Mac (Studio M4 Max):**
```bash
cd ~/Documents/GitHub/corridorkey-flame
git pull
make -C pybox install
pkill -f corridorkey_daemon_mlx
rm -f /tmp/corridorkey_*.json* /tmp/corridorkey_ready /tmp/corridorkey_trigger \
       /tmp/corridorkey_done /tmp/corridorkey_error
```

**Rocky Linux (flame-01):**
```bash
cd /opt/corridorkey-flame
git pull
make -C pybox install
make -C pybox install-ref   # only if reference/utils/ changed
pkill -f corridorkey_daemon_cuda
rm -f /tmp/corridorkey_*.json* /tmp/corridorkey_ready /tmp/corridorkey_trigger \
       /tmp/corridorkey_done /tmp/corridorkey_error
```

**Clean reinstall (either platform):**
```bash
sudo rm -rf /opt/corridorkey
bash install.sh
```

---

## Debugging

**Daemon log (tail -f while running):**
```bash
tail -f /tmp/corridorkey_daemon.log
```

**Check if daemon is alive:**
```bash
pgrep -fl corridorkey_daemon
```

**Force daemon restart:**
```bash
pkill -f corridorkey_daemon; sleep 1; rm -f /tmp/corridorkey_*
```
Then click Result in Flame to trigger auto-spawn.

**VRAM still held after crash (Linux):**
```bash
sudo fuser -k /dev/nvidia*   # kill all GPU processes
```

---

## Known Backlog

- **ONNX export of GreenFormer** — eliminate the daemon/IPC layer entirely.
  PyTorch `.pth` → ONNX → run directly in-process.
  Estimated: ~1 day. Key risk: Hiera windowed attention (`F.pad` calls may
  not trace cleanly through `torch.onnx.export`). Validate with
  `torch.onnx.export(..., dynamo=True)` and check for unsupported ops.
