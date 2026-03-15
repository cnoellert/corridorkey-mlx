from __future__ import print_function
import json, os, subprocess, sys, time
import pybox_v1 as pybox

PYBOX_DIR     = "/opt/corridorkey/pybox"
DAEMON_SCRIPT = os.path.join(PYBOX_DIR, "corridorkey_daemon.py")
CONDA_ENV     = "corridorkey-mlx"
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
DEFAULT_WEIGHTS = "/opt/corridorkey/models/CorridorKey_v1.0.mlx.npz"
FRAME_TIMEOUT = 30
POLL_INTERVAL = 0.2

def _cleanup_sentinels():
    for f in (TRIGGER, READY, DONE, ERROR):
        try: os.unlink(f)
        except OSError: pass

def _daemon_running():
    import subprocess
    result = subprocess.run(["pgrep", "-f", DAEMON_SCRIPT], capture_output=True)
    return result.returncode == 0

def _spawn_daemon(weights_path, quantized=False):
    if _daemon_running():
        return   # already running, don't spawn a second
    _cleanup_sentinels()
    cmd = (
        "source ~/.zprofile 2>/dev/null; "
        "source ~/miniconda3/etc/profile.d/conda.sh; "
        f"conda activate {CONDA_ENV}; "
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
    os.system(f"pkill -f '{DAEMON_SCRIPT}' 2>/dev/null; true")

def _send_frame(params):
    for f in (DONE, ERROR):
        try: os.unlink(f)
        except OSError: pass
    with open(PARAMS_FILE, "w") as f:
        json.dump(params, f)
    open(TRIGGER, "w").close()
    deadline = time.time() + FRAME_TIMEOUT
    while time.time() < deadline:
        if os.path.exists(ERROR):
            raise RuntimeError(open(ERROR).read().strip())
        if os.path.exists(DONE):
            return
        time.sleep(POLL_INTERVAL)
    raise TimeoutError(f"Daemon timeout after {FRAME_TIMEOUT}s")

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
            # Page 0 — Model: weights file + quantized toggle
            pybox.create_file_browser(
                "Weights", DEFAULT_WEIGHTS, "npz", os.path.expanduser("~"),
                row=0, col=0, page=0, tooltip="Path to .mlx.npz weights file",
            ),
            pybox.create_toggle_button(
                "Quantized", value=False, default=False,
                row=2, col=0, page=0,
                tooltip="Use int8 quantized weights (faster, smaller, minimal quality loss).",
            ),
            # Page 1 — Settings: processing controls
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
            pybox.create_toggle_button(
                "Reprocess", value=False, default=False,
                row=3, col=0, page=1,
                tooltip="Bump to re-run inference on current frame.",
            ),
        )
        self.set_ui_pages(
            pybox.create_page("Model",    "Model"),
            pybox.create_page("Settings", "Settings"),
        )
        self.set_state_id("execute")

    def execute(self):
        changes   = self.get_ui_changes()
        reprocess = self.get_render_element_value("Reprocess")

        weights_changed = any(el.get("name") in ("Weights", "Quantized") for el in changes)
        if weights_changed:
            try:    new_weights = self.get_render_element_value("Weights") or DEFAULT_WEIGHTS
            except: new_weights = DEFAULT_WEIGHTS
            try:    quantized = bool(self.get_render_element_value("Quantized"))
            except: quantized = False
            _kill_daemon()
            for f in (PARAMS_FILE + ".spawned", READY):
                try: os.unlink(f)
                except OSError: pass
            _spawn_daemon(new_weights, quantized=quantized)
            return

        # Run inference when:
        #   1. No output exists yet (first run)
        #   2. Frame has changed since last inference
        #   3. Reprocess explicitly toggled
        current_frame = self.get_frame()
        last_frame_file = PARAMS_FILE + ".last_frame"
        try:
            last_frame = int(open(last_frame_file).read().strip())
        except Exception:
            last_frame = None

        frame_changed = (last_frame != current_frame)
        first_run     = not os.path.exists(OUT_FG)

        if not first_run and not frame_changed and not reprocess:
            return

        # Clear Reprocess so it acts as a momentary button
        if reprocess:
            self.set_render_element_value("Reprocess", False)

        # Skip if daemon is still busy
        if os.path.exists(TRIGGER):
            self.set_warning_msg("CorridorKey: still processing, try again shortly.")
            return

        # Record current frame
        open(last_frame_file, "w").write(str(current_frame))

        if not os.path.exists(READY):
            if not os.path.exists(PARAMS_FILE + ".spawned"):
                try:    weights = self.get_render_element_value("Weights") or DEFAULT_WEIGHTS
                except: weights = DEFAULT_WEIGHTS
                try:    quantized = bool(self.get_render_element_value("Quantized"))
                except: quantized = False
                _spawn_daemon(weights, quantized=quantized)
                open(PARAMS_FILE + ".spawned", "w").close()
            self.set_notice_msg("CorridorKey: loading model, please wait...")
            deadline = time.time() + 30
            while time.time() < deadline:
                if os.path.exists(READY): break
                time.sleep(0.5)
            if not os.path.exists(READY):
                self.set_warning_msg("CorridorKey: model still loading. Try again shortly.")
                return
            self.set_notice_msg("")

        # "Add sRGB Gamma": encode linear->sRGB before inference, decode FG back to linear after
        add_srgb_gamma = bool(self.get_render_element_value("Add sRGB Gamma"))
        params = {
            "frame":            self.get_frame(),
            "add_srgb_gamma":    add_srgb_gamma,
            "despill_strength": float(self.get_render_element_value("Despill")),
            "despeckle":        float(self.get_render_element_value("Despeckle")),
        }
        try:
            _send_frame(params)
        except Exception as e:
            self.set_error_msg(f"CorridorKey: {e}")

    def teardown(self):
        try:
            with open(PARAMS_FILE, "w") as f: json.dump({"quit": True}, f)
            open(TRIGGER, "w").close()
            time.sleep(0.5)
        except: pass
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
