# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Dual-arm keyboard teleoperation for the bimanual Push-T environment.

Key bindings (always available)
--------------------------------
W / S / A / D       Move LEFT  arm EE in the table plane (+Y / -Y / -X / +X).
I / K / J / L       Move RIGHT arm EE in the table plane (+Y / -Y / -X / +X).
Q / E               Raise / lower BOTH arm EEs along the world Z axis.
[ / ]               Decrease / increase robot moving speed (both arms).
R                   Reset arm poses to home (discards current take if recording).

With --dataset_file (recording mode, additionally):
P                   START recording the current take (arms can move freely before this).
O                   STOP & SAVE current take to the dataset (arms stay in place).

Fixed top-down camera — no orbit/zoom keys needed.
Action vector layout: [vx_L, vy_L, vz_L, vx_R, vy_R, vz_R]  (6-dim, world frame).
"""

from __future__ import annotations

import argparse
import os
import time

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Bimanual teleoperation for the Push-T task.")
parser.add_argument("--task", type=str, default="Isaac-PushT-Bimanual-v0", help="Task name.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments.")
parser.add_argument("--step_hz", type=int, default=30, help="Teleoperation step rate.")
parser.add_argument("--sensitivity", type=float, default=0.6, help="Key velocity sensitivity.")
parser.add_argument("--speed", type=float, default=1.0, help="Initial EE speed multiplier.")
parser.add_argument(
    "--dataset_file",
    type=str,
    default=None,
    help="If set, record demonstrations to this HDF5 file path (enables P/O/R recording controls).",
)
parser.add_argument(
    "--overlay",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Draw the T-block and goal contour overlay on the interactive viewport (default: on).",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import numpy as np
import torch
import weakref

import carb
import omni

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg
from isaaclab_tasks.manager_based.manipulation.pusht.mdp.observations import (
    t_outline_world,
    _get_outline_local,
    _yaw_from_quat as _t_yaw_from_quat,
)

# Recording infrastructure (only imported when needed, but safe to import always)
from isaaclab.envs.mdp.recorders.recorders_cfg import ActionStateRecorderManagerCfg
from isaaclab.managers import DatasetExportMode


# ---------------------------------------------------------------------------
# Dual WASD / IJKL keyboard handler
# ---------------------------------------------------------------------------

class DualArmKeyboard:
    """Subscribes to carb keyboard events and maintains a 6-vector command.

    Command layout: [vx_L, vy_L, vz_L, vx_R, vy_R, vz_R] in world frame.
    Screen orientation (top-down, +Y = up, +X = right):
        W/I → +vy (+Y, screen-up)
        S/K → -vy (-Y, screen-down)
        A/J → -vx (-X, screen-left)
        D/L → +vx (+X, screen-right)
        Q   → +vz (both arms up)
        E   → -vz (both arms down)
    """

    def __init__(self, sensitivity: float = 0.6, sim_device: str = "cpu"):
        self._sensitivity = sensitivity
        self._sim_device = sim_device

        self._appwindow = omni.appwindow.get_default_app_window()
        self._input = carb.input.acquire_input_interface()
        self._keyboard = self._appwindow.get_keyboard()
        self._keyboard_sub = self._input.subscribe_to_keyboard_events(
            self._keyboard,
            lambda event, *args, obj=weakref.proxy(self): obj._on_keyboard_event(event, *args),
        )

        # 6-vector: [vx_L, vy_L, vz_L, vx_R, vy_R, vz_R]
        self._cmd = np.zeros(6, dtype=np.float32)

        # Additional one-shot callbacks (KEY_PRESS only), keyed by carb key name.
        self._callbacks: dict[str, callable] = {}

        self._create_key_bindings()

    def __del__(self):
        self._input.unsubscribe_to_keyboard_events(self._keyboard, self._keyboard_sub)
        self._keyboard_sub = None

    def _create_key_bindings(self):
        s = self._sensitivity
        self._KEY_DELTA: dict[str, np.ndarray] = {
            # Left arm XY (WASD)
            "W": np.array([0.0,  s, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
            "S": np.array([0.0, -s, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
            "A": np.array([-s, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
            "D": np.array([ s, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
            # Right arm XY (IJKL)
            "I": np.array([0.0, 0.0, 0.0, 0.0,  s, 0.0], dtype=np.float32),
            "K": np.array([0.0, 0.0, 0.0, 0.0, -s, 0.0], dtype=np.float32),
            "J": np.array([0.0, 0.0, 0.0, -s, 0.0, 0.0], dtype=np.float32),
            "L": np.array([0.0, 0.0, 0.0,  s, 0.0, 0.0], dtype=np.float32),
            # Both arms Z (Q/E)
            "Q": np.array([0.0, 0.0,  s, 0.0, 0.0,  s], dtype=np.float32),
            "E": np.array([0.0, 0.0, -s, 0.0, 0.0, -s], dtype=np.float32),
        }

    def reset(self):
        self._cmd.fill(0.0)

    def add_callback(self, key: str, func):
        """Register a one-shot callback fired on KEY_PRESS."""
        self._callbacks[key] = func

    def advance(self) -> torch.Tensor:
        """Return the current 4-dim command as a float32 tensor."""
        return torch.tensor(self._cmd, dtype=torch.float32, device=self._sim_device)

    def _on_keyboard_event(self, event, *args, **kwargs):
        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            name = event.input.name
            if name in self._KEY_DELTA:
                self._cmd += self._KEY_DELTA[name]
            if name in self._callbacks:
                self._callbacks[name]()
        elif event.type == carb.input.KeyboardEventType.KEY_RELEASE:
            name = event.input.name
            if name in self._KEY_DELTA:
                self._cmd -= self._KEY_DELTA[name]
        return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    recording_enabled = args_cli.dataset_file is not None

    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.episode_length_s = 1.0e9
    env_cfg.terminations.time_out = None

    # ---- Configure recording if requested --------------------------------
    if recording_enabled:
        output_dir = os.path.dirname(os.path.abspath(args_cli.dataset_file))
        output_file_name = os.path.splitext(os.path.basename(args_cli.dataset_file))[0]
        os.makedirs(output_dir, exist_ok=True)

        env_cfg.recorders = ActionStateRecorderManagerCfg()
        env_cfg.recorders.dataset_export_dir_path = output_dir
        env_cfg.recorders.dataset_filename = output_file_name
        env_cfg.recorders.dataset_export_mode = DatasetExportMode.EXPORT_ALL

        # Disable surprise auto-resets so the operator controls episode boundaries.
        env_cfg.terminations.success = None
        env_cfg.terminations.out_of_bounds = None
        # Per-term observations needed by ActionStateRecorderManagerCfg.
        env_cfg.observations.policy.concatenate_terms = False

    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped

    teleop = DualArmKeyboard(sensitivity=args_cli.sensitivity, sim_device=env.device)

    # -----------------------------------------------------------------------
    # State shared with callbacks
    # -----------------------------------------------------------------------
    speed_scale = [args_cli.speed]
    SPEED_DN, SPEED_UP = 0.8, 1.25
    SPEED_MIN, SPEED_MAX = 0.1, 4.0

    # Lifecycle flags
    should_reset = [False]  # R: reset arm poses (always)
    should_save = [False]   # O: save take (recording mode only)
    running = [False]       # whether a recording take is in progress
    saved_count = [0]

    # UI label references — assigned after the window is created below;
    # closures capture by name so they see the final values at call time.
    status_label = None
    saves_label = None

    # Speed control helpers (defined before the action terms are resolved)
    def speed_down():
        speed_scale[0] = max(SPEED_MIN, speed_scale[0] * SPEED_DN)
        left_term.set_speed_scale(speed_scale[0])
        right_term.set_speed_scale(speed_scale[0])
        print(f"[Push-T Bimanual] speed scale → {speed_scale[0]:.2f}×")

    def speed_up():
        speed_scale[0] = min(SPEED_MAX, speed_scale[0] * SPEED_UP)
        left_term.set_speed_scale(speed_scale[0])
        right_term.set_speed_scale(speed_scale[0])
        print(f"[Push-T Bimanual] speed scale → {speed_scale[0]:.2f}×")

    # ---- Callbacks -------------------------------------------------------
    # R always resets (discards recording if in progress)
    def request_reset():
        if recording_enabled and running[0]:
            print("[Push-T Bimanual] ✕  Discarding current take and resetting…")
        should_reset[0] = True
        running[0] = False

    teleop.add_callback("R", request_reset)

    if recording_enabled:
        def request_start():
            if not running[0]:
                # Clear any pre-P free-move steps, then snapshot the current
                # arm/scene position as the episode's initial state.
                env.recorder_manager.reset([0])
                env.recorder_manager.record_post_reset([0])
                running[0] = True
                if status_label is not None:
                    status_label.text = "●  RECORDING"
                    status_label.style = {"font_size": 22, "color": 0xFF22CC22}
                print("[Push-T Bimanual] ▶  Recording STARTED — move the arms, then press O to save.")

        def request_save():
            if running[0]:
                should_save[0] = True
            else:
                print("[Push-T Bimanual] Not recording — press P first to start a take.")

        teleop.add_callback("P", request_start)
        teleop.add_callback("O", request_save)

    teleop.add_callback("LEFT_BRACKET", speed_down)
    teleop.add_callback("RIGHT_BRACKET", speed_up)

    # -----------------------------------------------------------------------
    # Start
    # -----------------------------------------------------------------------
    env.reset()
    teleop.reset()

    left_term = env.action_manager.get_term("pusher_left")
    right_term = env.action_manager.get_term("pusher_right")
    left_term.set_speed_scale(speed_scale[0])
    right_term.set_speed_scale(speed_scale[0])

    # ---- Viewport overlay: T-block + goal contour via debug_draw ----------
    _draw_iface = None

    # Cached static overlay geometry (filled when overlay is enabled)
    _overlay_goal_pts_w = None    # (1, 8, 3) goal T world contour
    _overlay_table_segs = None    # (starts, ends) for 4 table-edge segments

    if args_cli.overlay:
        try:
            import isaacsim.util.debug_draw._debug_draw as _omni_debug_draw
            _draw_iface = _omni_debug_draw.acquire_debug_draw_interface()

            from isaaclab_tasks.manager_based.manipulation.pusht.pusht_bimanual_env_cfg import (
                GOAL_XY,
                GOAL_YAW,
            )
            _outline_local = _get_outline_local(env.device)
            _origin = env.scene.env_origins[0]  # (3,) — single env

            # Goal T contour (lifted 3 cm above table surface for viewport visibility)
            _goal_pos_xy = _origin[:2].unsqueeze(0) + torch.tensor(
                GOAL_XY, dtype=torch.float32, device=env.device
            )
            _goal_yaw_t = torch.tensor([GOAL_YAW], device=env.device)
            _overlay_goal_pts_w = t_outline_world(
                _goal_pos_xy, _goal_yaw_t, z=float(_origin[2].item()) + 0.03, outline_local=_outline_local
            )  # (1, 8, 3)

            # Table boundary (4 corners of the 1 m × 1 m top face, lifted 1 cm)
            _table_z = float(_origin[2].item()) + 0.01
            _table_corners = [
                [float(_origin[0].item()) - 0.5, float(_origin[1].item()) - 0.5, _table_z],
                [float(_origin[0].item()) + 0.5, float(_origin[1].item()) - 0.5, _table_z],
                [float(_origin[0].item()) + 0.5, float(_origin[1].item()) + 0.5, _table_z],
                [float(_origin[0].item()) - 0.5, float(_origin[1].item()) + 0.5, _table_z],
            ]
            _overlay_table_segs = (
                _table_corners,
                _table_corners[1:] + [_table_corners[0]],
            )

        except Exception as e:
            print(f"[Push-T Bimanual] Overlay unavailable: {e}")
            _draw_iface = None

    # ---- Recording status window (Isaac Sim UI) --------------------------
    if recording_enabled:
        import omni.ui as ui
        from isaaclab.envs.ui import EmptyWindow

        _rec_window = EmptyWindow(env, "Teleop Recording")
        with _rec_window.ui_window_elements["main_vstack"]:
            ui.Label(
                f"Dataset:  {args_cli.dataset_file}",
                style={"font_size": 13, "color": 0xFFAAAAAA},
            )
            ui.Spacer(height=6)
            status_label = ui.Label(
                "○  Not recording",
                style={"font_size": 22, "color": 0xFFAAAAAA},
            )
            saves_label = ui.Label(
                "Saved: 0 takes",
                style={"font_size": 14, "color": 0xFFCCCCCC},
            )
            ui.Spacer(height=6)
            ui.Label("P — start  |  O — save  |  R — reset", style={"font_size": 12, "color": 0xFF888888})

    _recording_hint = (
        "  P               — START recording the current take\n"
        "  O               — STOP & SAVE take to dataset (arms stay in place)\n"
        "  R               — DISCARD take and reset arm poses to home\n"
    ) if recording_enabled else (
        "  R               — reset arm poses to home\n"
    )
    print(
        f"\nBimanual Push-T teleoperation started"
        + (f" [RECORDING → {args_cli.dataset_file}]" if recording_enabled else "")
        + ":\n"
        "  W / S / A / D   — move LEFT  arm EE  (+Y / -Y / -X / +X)\n"
        "  I / K / J / L   — move RIGHT arm EE  (+Y / -Y / -X / +X)\n"
        "  Q / E           — raise / lower BOTH arm EEs  (+Z / -Z)\n"
        + _recording_hint
        + "  [ / ]           — decrease / increase EE speed (both arms)\n"
        f"  Initial speed scale: {speed_scale[0]:.2f}×\n"
        + ("\nMove arms to starting position, then press P to begin recording.\n"
           if recording_enabled else "")
        + f"  Viewport overlay: {'ON (--no-overlay to disable)' if args_cli.overlay else 'OFF (--overlay to enable)'}\n"
    )

    step_dt = 1.0 / args_cli.step_hz
    next_step_time = time.time()

    with torch.inference_mode():
        while simulation_app.is_running():

            # ---- Step (arms always movable) ------------------------------
            action = teleop.advance().unsqueeze(0).repeat(env.num_envs, 1)
            env.step(action)

            # ---- Viewport overlay: block + goal + table --------
            if _draw_iface is not None:
                try:
                    _block = env.scene["t_block"]
                    _outline_local = _get_outline_local(env.device)

                    # T-block contour (dynamic)
                    _b_pos = _block.data.root_pos_w[0:1, :2]
                    _b_yaw = _t_yaw_from_quat(_block.data.root_quat_w)[0:1]
                    _b_z = _block.data.root_pos_w[0, 2].item() + 0.02 + 0.03
                    _b_pts = t_outline_world(_b_pos, _b_yaw, z=_b_z, outline_local=_outline_local)
                    _b_np = _b_pts[0].cpu().numpy().tolist()   # 8 × [x,y,z]

                    # Goal T contour (static, cached)
                    _g_np = _overlay_goal_pts_w[0].cpu().numpy().tolist()

                    # Table boundary (static, cached)
                    _t_starts, _t_ends = _overlay_table_segs

                    # Concatenate all segments: block(8) + goal(8) + table(4)
                    _starts = _b_np + _g_np + _t_starts
                    _ends = _b_np[1:] + [_b_np[0]] + _g_np[1:] + [_g_np[0]] + _t_ends
                    _colors = (
                        [[1.0, 1.0, 1.0, 1.0]] * 8     # block: white
                        + [[0.0, 0.8, 0.0, 1.0]] * 8   # goal: green
                        + [[0.7, 0.7, 0.7, 1.0]] * 4   # table: bright gray
                    )
                    _thicks = [3.0] * 8 + [2.0] * 8 + [2.0] * 4

                    _draw_iface.clear_lines()
                    _draw_iface.draw_lines(_starts, _ends, _colors, _thicks)
                except Exception:
                    pass  # silently skip overlay errors mid-loop

            # ---- Save: export take, arms stay in place -------------------
            if should_save[0]:
                # record_pre_reset finalises the buffer and auto-exports
                # (export_in_record_pre_reset=True, EXPORT_ALL).
                env.recorder_manager.record_pre_reset([0])
                running[0] = False
                should_save[0] = False
                saved_count[0] += 1
                if status_label is not None:
                    status_label.text = "○  Not recording"
                    status_label.style = {"font_size": 22, "color": 0xFFAAAAAA}
                if saves_label is not None:
                    saves_label.text = f"Saved: {saved_count[0]} take{'s' if saved_count[0] != 1 else ''}"
                print(
                    f"[Push-T Bimanual] ✔  Take saved  "
                    f"(total saved: {saved_count[0]}).  "
                    "Move arms and press P to start next take."
                )

            # ---- Reset: discard buffer and go home -----------------------
            elif should_reset[0]:
                # Always discard any buffered data (pre-P free-move or in-progress take).
                if recording_enabled:
                    env.recorder_manager.reset([0])
                env.reset()
                teleop.reset()
                left_term.set_speed_scale(speed_scale[0])
                right_term.set_speed_scale(speed_scale[0])
                should_reset[0] = False
                if status_label is not None:
                    status_label.text = "○  Not recording"
                    status_label.style = {"font_size": 22, "color": 0xFFAAAAAA}
                if recording_enabled:
                    print("[Push-T Bimanual] Arms reset.  Move arms and press P to start a take.")

            # ---- Pace the loop -------------------------------------------
            next_step_time += step_dt
            while time.time() < next_step_time and simulation_app.is_running():
                time.sleep(min(0.005, next_step_time - time.time()))
                env.sim.render()
            if env.sim.is_stopped():
                break

    env.close()

    if recording_enabled:
        print(
            f"\nRecording session ended.  "
            f"Saved {saved_count[0]} take(s) to: {args_cli.dataset_file}"
        )


if __name__ == "__main__":
    main()
    simulation_app.close()
