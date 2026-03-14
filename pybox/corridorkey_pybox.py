"""
corridorkey_pybox.py  —  Flame PyBox handler for CorridorKey MLX inference

Two-file architecture:
  corridorkey_pybox.py   — this file, runs in Flame's embedded Python (thin IPC glue)
  corridorkey_daemon.py  — inference server, runs in corridorkey-mlx conda env

The daemon is spawned once on initialize(), loads the model, and serves frames
via FIFO. Per-frame cost is inference time only (~3s on M4 Max), no reload overhead.

Install:
  Copy both files somewhere accessible from Flame (e.g. ~/flame/pybox/).
  In Flame batch, add a PyBox node and point it at corridorkey_pybox.py.

Tested on Flame 2026.2.1 / macOS / Apple Silicon.
"""

from __future__ import print_function

import json
import os
import sys
import time

import pybox_v1 as pybox

# ---------------------------------------------------------------------------
# Paths — edit CONDA_ENV and DEFAULT_WEIGHTS if your setup differs
# ---------------------------------------------------------------------------
_HERE         = os.path.dirname(os.path.abspath(__file__))
DAEMON_SCRIPT = os.path.join(_HERE, "corridorkey_daemon.py")
CONDA_ENV     = "corridorkey-mlx"

_PFX      = "/tmp/corridorkey_"
IN_PLATE  = _PFX + "in_plate.exr"
IN_MATTE  = _PFX + "in_matte.exr"
OUT_FG    = _PFX + "out_fg.exr"
OUT_ALPHA = _PFX + "out_alpha.exr"
CMD_FIFO  = _PFX + "cmd"
DONE_FIFO = _PFX + "done"

DEFAULT_WEIGHTS = os.path.expanduser(
    "~/ComfyUI/models/corridorkey/CorridorKey_v1.0.mlx.npz"
)
DAEMON_READY_TIMEOUT = 60   # seconds to wait for model load on first launch


# ---------------------------------------------------------------------------
# Daemon helpers
# ---------------------------------------------------------------------------

def _make_fifos():
    for f in (CMD_FIFO, DONE_FIFO):
        if os.path.exists(f):
            os.unlink(f)
        os.mkfifo(f)


def _spawn_daemon(weights_path):
    """Launch corridorkey_daemon.py in the corridorkey-mlx conda env."""
    cmd = (
        f"source ~/.zprofile 2>/dev/null; "
        f"source ~/miniconda3/etc/profile.d/conda.sh; "
        f"conda activate {CONDA_ENV}; "
        f"python3 {DAEMON_SCRIPT} "
        f"--weights '{weights_path}' "
        f"--cmd-fifo {CMD_FIFO} "
        f"--done-fifo {DONE_FIFO} "
        f"--in-plate {IN_PLATE} "
        f"--in-matte {IN_MATTE} "
        f"--out-fg {OUT_FG} "
        f"--out-alpha {OUT_ALPHA} "
        f"> /tmp/corridorkey_daemon.log 2>&1 &"
    )
    os.system(f"zsh -c '{cmd}'")


def _wait_for_daemon(timeout=DAEMON_READY_TIMEOUT):
    """Block until daemon opens CMD_FIFO for reading (signals it's ready)."""
    # The daemon opens CMD_FIFO for reading after model load.
    # We detect readiness by checking that the FIFO has a reader via lsof.
    deadline = time.time() + timeout
    while time.time() < deadline:
        result = os.popen(f"lsof {CMD_FIFO} 2>/dev/null").read()
        if result.strip():
            return True
        time.sleep(0.5)
    return False


def _send_frame(params: dict):
    """Write params to CMD_FIFO, block on DONE_FIFO until daemon signals done."""
    with open(CMD_FIFO, "w") as f:
        f.write(json.dumps(params) + "\n")
    # Block until daemon signals completion
    with open(DONE_FIFO, "r") as f:
        f.read()


def _kill_daemon():
    os.system(f"pkill -f '{DAEMON_SCRIPT}' 2>/dev/null || true")


# ---------------------------------------------------------------------------
# Handler class
# ---------------------------------------------------------------------------

class CorridorKeyBox(pybox.BaseClass):

    def initialize(self):
        self.set_img_format("exr")

        self.remove_in_sockets()
        self.add_in_socket("Front", IN_PLATE)
        self.add_in_socket("Matte", IN_MATTE)

        self.remove_out_sockets()
        self.add_out_socket("Result",   OUT_FG)
        self.add_out_socket("OutMatte", OUT_ALPHA)

        # Spawn the daemon (model loads in background)
        _kill_daemon()   # clean up any stale instance
        _make_fifos()
        weights = self.get_global_element_value("Weights") or DEFAULT_WEIGHTS
        _spawn_daemon(weights)

        self.set_state_id("setup_ui")
        self.setup_ui()

    def setup_ui(self):
        self.add_global_elements(
            pybox.create_file_browser(
                "Weights", DEFAULT_WEIGHTS, "npz", os.path.expanduser("~"),
                row=0, col=0,
                tooltip="Path to .mlx.npz weights file",
            ),
            pybox.create_float_numeric(
                "Despill Strength", value=1.0, default=1.0, min=0.0, max=1.0, inc=0.05,
                row=1, col=0,
                tooltip="Green spill removal intensity (0=off, 1=full)",
            ),
            pybox.create_toggle_button(
                "Input is sRGB", value=True, default=True,
                row=2, col=0,
                tooltip="Enable for REC709/sRGB footage. Disable for scene-linear EXR",
            ),
            pybox.create_toggle_button(
                "Despeckle", value=False, default=False,
                row=3, col=0,
                tooltip="Remove isolated alpha specks. Off by default — degrades refiner edges",
            ),
            pybox.create_toggle_button(
                "Reprocess", value=False, default=False,
                row=0, col=1,
                tooltip="Force re-inference on current frame after changing params",
            ),
        )
        self.set_ui_pages(pybox.create_page("CorridorKey", "Settings", "Actions"))
        self.set_state_id("execute")


    def execute(self):
        changes = self.get_ui_changes()
        reprocess = self.get_global_element_value("Reprocess")

        # If the weights path changed, restart the daemon with new weights
        for el in changes:
            if el.get("name") == "Weights":
                _kill_daemon()
                _make_fifos()
                _spawn_daemon(self.get_global_element_value("Weights"))
                if not _wait_for_daemon():
                    self.set_error_msg("CorridorKey daemon failed to start — check /tmp/corridorkey_daemon.log")
                    return
                break

        # UI-only change, no new frame to render — skip inference unless Reprocess
        if changes and not reprocess:
            return

        # Reset Reprocess toggle
        if reprocess:
            self.set_global_element_value("Reprocess", False)

        # Wait for daemon to be ready (first frame after initialize)
        if not os.path.exists(CMD_FIFO) or not os.path.exists(DONE_FIFO):
            _make_fifos()
            weights = self.get_global_element_value("Weights") or DEFAULT_WEIGHTS
            _spawn_daemon(weights)

        if not _wait_for_daemon():
            self.set_error_msg("CorridorKey daemon not ready — check /tmp/corridorkey_daemon.log")
            return

        params = {
            "frame":            self.get_frame(),
            "despill_strength": float(self.get_global_element_value("Despill Strength")),
            "input_is_srgb":    bool(self.get_global_element_value("Input is sRGB")),
            "despeckle":        bool(self.get_global_element_value("Despeckle")),
        }

        try:
            _send_frame(params)
        except Exception as e:
            self.set_error_msg(f"CorridorKey IPC error: {e}")

    def teardown(self):
        try:
            # Send quit signal
            with open(CMD_FIFO, "w") as f:
                f.write(json.dumps({"quit": True}) + "\n")
            time.sleep(0.5)
        except Exception:
            pass
        _kill_daemon()
        for f in (CMD_FIFO, DONE_FIFO):
            try:
                os.unlink(f)
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Entry point (called by Flame)
# ---------------------------------------------------------------------------
def _main(argv):
    p = CorridorKeyBox(argv[0], argv[1] if len(argv) > 1 else "")
    p.dispatch()
    p.write_to_disk(argv[0])


if __name__ == "__main__":
    _main(sys.argv[1:])
