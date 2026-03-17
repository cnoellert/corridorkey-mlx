from __future__ import print_function
import json, os, platform, subprocess, sys, time
import pybox_v1 as pybox

# ---------------------------------------------------------------------------
# HOW THIS PYBOX WORKS — READ THIS BEFORE TOUCHING execute()
# ---------------------------------------------------------------------------
# Flame calls execute() constantly: on every scrub frame, every UI tick,
# every timeline navigation event.  It is NOT a "run when needed" callback.
#
# The critical gate is:
#
#     if not self.is_processing():
#         return
#
# is_processing() returns True ONLY when Flame has committed to rendering
# a frame — i.e. it is actually waiting for output EXRs right now.
# During interactive scrubbing, is_processing() is False.  execute() just
# returns immediately with no work done.
#
# This means that when is_processing() IS True, it is safe to BLOCK inside
# execute() for the full duration of inference (~3 s Mac, ~8 s Linux).
# Flame is deliberately waiting.  There is no timeout that fires during a
# real render request.
#
# We discovered this pattern by reading BLGforFlame's PyBox (FilmLight, 2017),
# which uses the identical guard and then does a 45-second blocking urlopen.
# See docs/PYBOX_ARCHITECTURE.md for the full story.
# ---------------------------------------------------------------------------

PYBOX_DIR  = "/opt/corridorkey/pybox"
_IS_MACOS  = platform.system() == "Darwin"
DAEMON_SCRIPT = os.path.join(PYBOX_DIR,
    "corridorkey_daemon_mlx.py" if _IS_MACOS else "corridorkey_daemon_cuda.py")
CONDA_ENV  = "corridorkey-mlx" if _IS_MACOS else "corridorkey-cuda"
_PFX          = "/tmp/corridorkey_"
IN_PLATE      = _PFX + "in_plate.exr"
IN_MATTE      = _PFX + "in_matte.exr"
OUT_FG        = _PFX + "out_fg.exr"
OUT_ALPHA     = _PFX + "out_alpha.exr"
PARAMS_FILE   = _PFX + "params.json"
TRIGGER       = _PFX + "trigger"
READY         = _PFX + "ready"
DONE          = _PFX + "done"
ERROR         = _PFX + "error"
DEFAULT_WEIGHTS = (
    "/opt/corridorkey/models/CorridorKey_v1.0.mlx.npz" if _IS_MACOS
    else "/opt/corridorkey/models/CorridorKey_v1.0.pth"
)
FRAME_TIMEOUT = 60   # seconds — Flame is actively waiting, be generous
POLL_INTERVAL = 0.1  # seconds between DONE/ERROR polls


def _cleanup_sentinels():
    """Remove all IPC sentinel files. Call after killing daemon."""
    for f in (TRIGGER, READY, DONE, ERROR):
        try: os.unlink(f)
        except OSError: pass


def _daemon_running():
    result = subprocess.run(["pgrep", "-f", DAEMON_SCRIPT], capture_output=True)
    return result.returncode == 0


def _spawn_daemon(weights_path, quantized=False):
    """Spawn the background inference daemon if not already running."""
    if _daemon_running():
        return
    _cleanup_sentinels()
    env_vars = "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True " if not _IS_MACOS else ""
    cmd = (
        "source ~/.zprofile 2>/dev/null; "
        "source ~/miniconda3/etc/profile.d/conda.sh; "
        f"conda activate {CONDA_ENV}; "
        f"{env_vars}"
        f"python3 {DAEMON_SCRIPT} "
        f"  {'--quantized' if quantized else ''} "
        f"  --weights '{weights_path}' "
        f"  --in-plate {IN_PLATE} --in-matte {IN_MATTE} "
        f"  --out-fg {OUT_FG} --out-alpha {OUT_ALPHA} "
        f"  --params {PARAMS_FILE} --trigger {TRIGGER} "
        f"  --ready {READY} --done {DONE} --error {ERROR} "
        f"  >> /tmp/corridorkey_daemon.log 2>&1 &"
    )
    os.system(f"zsh -c '{cmd}'")


def _kill_daemon():
    """Kill the daemon and clean up sentinels. Always wait 0.5 s after pkill."""
    os.system(f"pkill -f '{DAEMON_SCRIPT}' 2>/dev/null; true")
    time.sleep(0.5)   # pkill is async — must wait before spawning replacement
    _cleanup_sentinels()


def _send_frame(params):
    """
    Write params + trigger, then BLOCK until daemon writes DONE or ERROR.

    Safe to call only when is_processing() is True (Flame is waiting for us).
    """
    for f in (DONE, ERROR):
        try: os.unlink(f)
        except OSError: pass

    with open(PARAMS_FILE, "w") as fh:
        json.dump(params, fh)
    open(TRIGGER, "w").close()

    deadline = time.time() + FRAME_TIMEOUT
    while time.time() < deadline:
        if os.path.exists(ERROR):
            msg = open(ERROR).read().strip()
            try: os.unlink(ERROR)
            except OSError: pass
            raise RuntimeError(f"CorridorKey daemon error: {msg}")
        if os.path.exists(DONE):
            try: os.unlink(DONE)
            except OSError: pass
            return
        time.sleep(POLL_INTERVAL)

    raise TimeoutError(
        f"CorridorKey daemon did not respond within {FRAME_TIMEOUT}s. "
        "Check /tmp/corridorkey_daemon.log."
    )


class CorridorKeyBox(pybox.BaseClass):

    def initialize(self):
        self.set_img_format("exr")
        self.remove_in_sockets()
        self.add_in_socket("Front", IN_PLATE)
        self.add_in_socket("Matte", IN_MATTE)
        self.remove_out_sockets()
        self.add_out_socket("Result",   OUT_FG)
        self.add_out_socket("OutMatte", OUT_ALPHA)
        self.set_state_id("setup_ui")
        self.setup_ui()

    def setup_ui(self):
        self.add_render_elements(
            pybox.create_file_browser(
                "Weights", DEFAULT_WEIGHTS, "npz", os.path.expanduser("~"),
                row=0, col=0, page=0,
                tooltip="Path to model weights (.mlx.npz on Mac, .pth on Linux).",
            ),
            pybox.create_toggle_button(
                "Quantized", value=False, default=False,
                row=2, col=0, page=0,
                tooltip="Use int8 quantized weights (faster, smaller). Mac only.",
            ),
            pybox.create_toggle_button(
                "Add sRGB Gamma", value=False, default=False,
                row=0, col=0, page=1,
                tooltip="Encode linear input to sRGB before inference, decode FG back to linear on output.",
            ),
            pybox.create_float_numeric(
                "Despill", value=1.0, default=1.0, min=0.0, max=1.0, inc=0.05,
                row=1, col=0, page=1,
                channel_name="despill_strength_chn",
                tooltip="Green spill suppression strength (0=off, 1=full).",
            ),
            pybox.create_float_numeric(
                "Despeckle", value=0.0, default=0.0, min=0.0, max=2000.0, inc=50.0,
                row=2, col=0, page=1,
                channel_name="despeckle_area_chn",
                tooltip="Remove alpha specks smaller than this area in pixels. 0=off.",
            ),
        )
        self.set_ui_pages(
            pybox.create_page("Model",    "Model"),
            pybox.create_page("Settings", "Settings"),
        )
        self.set_state_id("execute")

    def execute(self):
        # Handle weights/quantized changes regardless of render state
        changes = self.get_ui_changes()
        weights_changed = any(el.get("name") in ("Weights", "Quantized") for el in changes)
        if weights_changed:
            try:    new_weights = self.get_render_element_value("Weights") or DEFAULT_WEIGHTS
            except: new_weights = DEFAULT_WEIGHTS
            try:    quantized = bool(self.get_render_element_value("Quantized"))
            except: quantized = False
            _kill_daemon()
            _spawn_daemon(new_weights, quantized=quantized)
            return

        # THE CRITICAL GATE — do not remove or move inference code above this.
        # is_processing() is True only when Flame is actively waiting for output
        # EXRs. During scrubbing and UI ticks it is False; return immediately.
        if not self.is_processing():
            return

        # Flame is waiting. Blocking is correct from here.

        # Auto-spawn daemon on first use
        if not os.path.exists(READY):
            try:    weights = self.get_render_element_value("Weights") or DEFAULT_WEIGHTS
            except: weights = DEFAULT_WEIGHTS
            try:    quantized = bool(self.get_render_element_value("Quantized"))
            except: quantized = False
            if not _daemon_running():
                _spawn_daemon(weights, quantized=quantized)
            self.set_notice_msg("CorridorKey: loading model…")
            deadline = time.time() + 60
            while time.time() < deadline:
                if os.path.exists(READY):
                    break
                time.sleep(0.5)
            if not os.path.exists(READY):
                self.set_warning_msg("CorridorKey: model load timeout. Check /tmp/corridorkey_daemon.log.")
                return
            self.set_notice_msg("")

        try:
            params = {
                "frame":            self.get_frame(),
                "add_srgb_gamma":   bool(self.get_render_element_value("Add sRGB Gamma")),
                "despill_strength": float(self.get_render_element_value("Despill")),
                "despeckle":        float(self.get_render_element_value("Despeckle")),
            }
        except Exception as e:
            self.set_warning_msg(f"CorridorKey: param error: {e}")
            return

        try:
            _send_frame(params)
        except Exception as e:
            self.set_warning_msg(str(e))

    def teardown(self):
        try:
            with open(PARAMS_FILE, "w") as fh:
                json.dump({"quit": True}, fh)
            open(TRIGGER, "w").close()
            time.sleep(0.5)
        except Exception:
            pass
        _kill_daemon()
        _cleanup_sentinels()
        try: os.unlink(PARAMS_FILE)
        except OSError: pass


def _main(argv):
    p = CorridorKeyBox(argv[0], argv[1] if len(argv) > 1 else "")
    p.dispatch()
    p.write_to_disk(argv[0])

if __name__ == "__main__":
    _main(sys.argv[1:])
