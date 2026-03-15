"""
corridorkey_pybox.py  —  Flame PyBox handler for CorridorKey MLX inference

IPC: file-based polling (no FIFOs — avoids blocking open() killing the handler).

  Handler writes:  /tmp/corridorkey_params.json  (inference params)
                   /tmp/corridorkey_trigger       (empty, signals daemon to run)
  Daemon writes:   /tmp/corridorkey_ready         (empty, signals model loaded)
                   /tmp/corridorkey_done          (empty, signals frame complete)
                   /tmp/corridorkey_error         (error message if failed)

The daemon is spawned once on initialize(). Model loads in ~2s.
Per-frame cost = inference time only (~3s on M4 Max).

Install: copy pybox/ dir to e.g. ~/flame/pybox/corridorkey/
         In Flame batch > Add node > PyBox > point at corridorkey_pybox.py
"""

from __future__ import print_function

import json
import os
import sys
import time

import pybox_v1 as pybox

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
# Flame copies the handler to /var/tmp/ before running it, so __file__
# is useless for locating the daemon. Use the absolute install path instead.
PYBOX_DIR     = os.path.expanduser("~/flame/pybox/corridorkey")
DAEMON_SCRIPT = os.path.join(PYBOX_DIR, "corridorkey_daemon.py")
CONDA_ENV     = "corridorkey-mlx"

_PFX        = "/tmp/corridorkey_"
IN_PLATE    = _PFX + "in_plate.exr"
IN_MATTE    = _PFX + "in_matte.exr"
OUT_FG      = _PFX + "out_fg.exr"
OUT_ALPHA   = _PFX + "out_alpha.exr"
PARAMS_FILE = _PFX + "params.json"
TRIGGER     = _PFX + "trigger"
READY       = _PFX + "ready"
DONE        = _PFX + "done"
ERROR       = _PFX + "error"

DEFAULT_WEIGHTS = os.path.expanduser(
    "~/ComfyUI/models/corridorkey/CorridorKey_v1.0.mlx.npz"
)
READY_TIMEOUT = 90    # seconds to wait for model load
FRAME_TIMEOUT = 30    # seconds to wait per frame
POLL_INTERVAL = 0.2   # polling interval in seconds


# ---------------------------------------------------------------------------
# Daemon helpers
# ---------------------------------------------------------------------------

def _cleanup_sentinels():
    for f in (TRIGGER, READY, DONE, ERROR):
        try:
            os.unlink(f)
        except OSError:
            pass


def _spawn_daemon(weights_path):
    _cleanup_sentinels()
    cmd = (
        "source ~/.zprofile 2>/dev/null; "
        "source ~/miniconda3/etc/profile.d/conda.sh; "
        f"conda activate {CONDA_ENV}; "
        f"python3 {DAEMON_SCRIPT} "
        f"  --weights '{weights_path}' "
        f"  --in-plate  {IN_PLATE}  "
        f"  --in-matte  {IN_MATTE}  "
        f"  --out-fg    {OUT_FG}    "
        f"  --out-alpha {OUT_ALPHA} "
        f"  --params    {PARAMS_FILE} "
        f"  --trigger   {TRIGGER}   "
        f"  --ready     {READY}     "
        f"  --done      {DONE}      "
        f"  --error     {ERROR}     "
        f"  >> /tmp/corridorkey_daemon.log 2>&1 &"
    )
    os.system(f"zsh -c '{cmd}'")


def _kill_daemon():
    os.system(f"pkill -f '{DAEMON_SCRIPT}' 2>/dev/null; true")


def _wait_for_ready(timeout=READY_TIMEOUT):
    """Poll for READY sentinel — written by daemon after model load."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        if os.path.exists(READY):
            return True
        time.sleep(POLL_INTERVAL)
    return False


def _send_frame(params):
    """Write params, drop trigger, poll for done."""
    # Clear any stale done/error from previous frame
    for f in (DONE, ERROR):
        try:
            os.unlink(f)
        except OSError:
            pass

    with open(PARAMS_FILE, "w") as f:
        json.dump(params, f)

    # Drop trigger file — daemon polls for this
    open(TRIGGER, "w").close()

    # Poll for done (or error)
    deadline = time.time() + FRAME_TIMEOUT
    while time.time() < deadline:
        if os.path.exists(ERROR):
            msg = open(ERROR).read().strip()
            raise RuntimeError(f"Daemon error: {msg}")
        if os.path.exists(DONE):
            return
        time.sleep(POLL_INTERVAL)

    raise TimeoutError(f"Daemon did not complete frame within {FRAME_TIMEOUT}s")


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

        # NOTE: Do NOT spawn daemon here — os.system() in initialize() crashes Flame.
        # Daemon is spawned lazily on first execute() call instead.

        self.set_state_id("setup_ui")
        self.setup_ui()

    def setup_ui(self):
        self.add_global_elements(
            pybox.create_file_browser(
                "Weights", DEFAULT_WEIGHTS, "npz", os.path.expanduser("~"),
                row=0, col=0, page=0,
                tooltip="Path to .mlx.npz weights file",
            ),
            pybox.create_float_numeric(
                "Despill Strength", value=1.0, default=1.0,
                min=0.0, max=1.0, inc=0.05,
                row=1, col=0, page=0,
                tooltip="Green spill removal (0=off, 1=full)",
            ),
            pybox.create_toggle_button(
                "Input is sRGB", value=True, default=True,
                row=2, col=0, page=0,
                tooltip="On for REC709 footage; off for scene-linear EXR",
            ),
            pybox.create_toggle_button(
                "Despeckle", value=False, default=False,
                row=3, col=0, page=0,
                tooltip="Remove isolated alpha specks (off by default)",
            ),
            pybox.create_toggle_button(
                "Reprocess", value=False, default=False,
                row=0, col=0, page=1,
                tooltip="Bump to re-run inference on current frame",
            ),
        )
        self.set_ui_pages(pybox.create_page("CorridorKey", "Settings", "Actions"))
        self.set_state_id("execute")


    def execute(self):
        changes   = self.get_ui_changes()
        reprocess = self.get_global_element_value("Reprocess")

        # Weights changed — restart daemon with new path
        weights_changed = False
        for el in changes:
            if el.get("name") == "Weights":
                weights_changed = True
                break

        if weights_changed:
            try:
                new_weights = self.get_global_element_value("Weights") or DEFAULT_WEIGHTS
            except Exception:
                new_weights = DEFAULT_WEIGHTS
            _kill_daemon()
            # Clear spawned breadcrumb so lazy init re-triggers
            for f in (PARAMS_FILE + ".spawned", READY):
                try: os.unlink(f)
                except OSError: pass
            _spawn_daemon(new_weights)
            return  # Non-blocking — model loads async, next frame will pick it up

        # UI-only change — skip inference unless Reprocess toggled
        if changes and not reprocess:
            return

        if reprocess:
            self.set_global_element_value("Reprocess", False)

        # Lazy daemon spawn — first execute() after node load.
        if not os.path.exists(READY):
            if not os.path.exists(PARAMS_FILE + ".spawned"):
                try:
                    weights = self.get_global_element_value("Weights") or DEFAULT_WEIGHTS
                except Exception:
                    weights = DEFAULT_WEIGHTS
                _spawn_daemon(weights)
                open(PARAMS_FILE + ".spawned", "w").close()

            # Block briefly (up to 30s) so the user sees a loading notice.
            # Poll in small increments — return early once ready.
            self.set_notice_msg("CorridorKey: loading model, please wait...")
            deadline = time.time() + 30
            while time.time() < deadline:
                if os.path.exists(READY):
                    break
                time.sleep(0.5)

            if not os.path.exists(READY):
                self.set_warning_msg(
                    "CorridorKey: model still loading. "
                    "Try scrubbing to this frame again in a moment."
                )
                return

            self.set_notice_msg("")  # Clear loading notice

        params = {
            "frame":            self.get_frame(),
            "despill_strength": float(self.get_global_element_value("Despill Strength")),
            "input_is_srgb":    bool(self.get_global_element_value("Input is sRGB")),
            "despeckle":        bool(self.get_global_element_value("Despeckle")),
        }

        try:
            _send_frame(params)
        except Exception as e:
            self.set_error_msg(f"CorridorKey: {e}")

    def teardown(self):
        # Signal daemon to quit then kill
        try:
            with open(PARAMS_FILE, "w") as f:
                json.dump({"quit": True}, f)
            open(TRIGGER, "w").close()
            time.sleep(0.5)
        except Exception:
            pass
        _kill_daemon()
        _cleanup_sentinels()
        try:
            os.unlink(PARAMS_FILE)
        except OSError:
            pass


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def _main(argv):
    p = CorridorKeyBox(argv[0], argv[1] if len(argv) > 1 else "")
    p.dispatch()
    p.write_to_disk(argv[0])


if __name__ == "__main__":
    _main(sys.argv[1:])
