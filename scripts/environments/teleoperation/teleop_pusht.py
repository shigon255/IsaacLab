# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Keyboard teleoperation for the Push-T environment.

Key bindings
------------
Arrow / Numpad 8/2/4/6   Move Franka EE in the table plane (camera-aligned).
Z / Numpad 7             EE up.
X / Numpad 9             EE down.
R                        Reset episode (arm + T-block).
N / M                    Orbit camera left / right.
, / .                    Orbit camera down / up (elevation).
Page Up / Down           Zoom camera in / out.
[ / ]                    Decrease / increase robot moving speed.
"""

from __future__ import annotations

import argparse
import math
import time

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Teleoperate the Push-T tabletop task.")
parser.add_argument("--task", type=str, default="Isaac-PushT-v0", help="Task name.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments.")
parser.add_argument("--step_hz", type=int, default=30, help="Teleoperation step rate.")
parser.add_argument("--sensitivity", type=float, default=0.6, help="Keyboard motion sensitivity.")
parser.add_argument("--speed", type=float, default=1.0, help="Initial EE speed multiplier (applied on top of velocity_scale).")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

from isaaclab.devices.keyboard import Se2Keyboard, Se2KeyboardCfg

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg


# ---------------------------------------------------------------------------
# Orbit-state helpers — used for keyboard camera control
# ---------------------------------------------------------------------------

def _eye_from_orbit(azimuth: float, elevation: float, radius: float) -> tuple[float, float, float]:
    """Convert spherical orbit params (all in radians/metres) to an env-local eye position.

    Lookat is always (0, 0, 0) in env-local coords (table centre).
    """
    cos_el = math.cos(elevation)
    x = radius * cos_el * math.cos(azimuth)
    y = radius * cos_el * math.sin(azimuth)
    z = radius * math.sin(elevation)
    return (x, y, z)


def _orbit_from_eye(eye: tuple[float, float, float]) -> tuple[float, float, float]:
    """Decompose an env-local eye position into (azimuth, elevation, radius)."""
    x, y, z = eye
    radius = math.sqrt(x * x + y * y + z * z)
    if radius < 1e-6:
        return (0.0, math.pi / 4, 1.5)
    azimuth = math.atan2(y, x)
    elevation = math.asin(max(-1.0, min(1.0, z / radius)))
    return (azimuth, elevation, radius)


# ---------------------------------------------------------------------------
# Live viewport-camera azimuth reader
# ---------------------------------------------------------------------------

def _read_viewport_azimuth() -> float | None:
    """Return the horizontal azimuth (rad) the active viewport camera is looking toward.

    Returns None if the viewport / camera isn't accessible (headless run etc.).
    """
    try:
        from omni.kit.viewport.utility import get_active_viewport
        import omni.usd
        from pxr import Usd, UsdGeom, Gf  # noqa: F401 (Gf for type)

        vp = get_active_viewport()
        if vp is None:
            return None
        stage = omni.usd.get_context().get_stage()
        cam_path = vp.get_active_camera()
        cam_prim = stage.GetPrimAtPath(cam_path)
        if not cam_prim.IsValid():
            return None

        xform = UsdGeom.Xformable(cam_prim)
        m = xform.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        # Camera looks down its local -Z axis; transform that into world space
        fwd = m.TransformDir(Gf.Vec3d(0.0, 0.0, -1.0))
        xy_len = math.hypot(fwd[0], fwd[1])
        if xy_len < 1e-4:
            # Near-vertical (top-down) — skip; keep last yaw
            return None
        return math.atan2(fwd[1], fwd[0])
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.episode_length_s = 1.0e9
    env_cfg.terminations.time_out = None
    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped

    teleop = Se2Keyboard(
        Se2KeyboardCfg(
            v_x_sensitivity=args_cli.sensitivity,
            v_y_sensitivity=args_cli.sensitivity,
            omega_z_sensitivity=args_cli.sensitivity,  # Z/X keys → EE up/down
            sim_device=env.device,
        )
    )

    # -----------------------------------------------------------------------
    # State shared between callbacks and the main loop
    # -----------------------------------------------------------------------
    speed_scale = [args_cli.speed]  # mutable via callbacks

    # Initial orbit parameters derived from the ViewerCfg eye position.
    # Lookat is always env-local (0, 0, 0) = table centre.
    viewer_eye = tuple(env_cfg.viewer.eye)  # e.g. (-1.3, -0.9, 1.0)
    orbit = list(_orbit_from_eye(viewer_eye))  # [azimuth, elevation, radius]
    # orbit[0] = azimuth, orbit[1] = elevation, orbit[2] = radius

    ORBIT_STEP_DEG = 15.0
    ORBIT_STEP = math.radians(ORBIT_STEP_DEG)
    ELEV_STEP = math.radians(10.0)
    ELEV_MIN = math.radians(5.0)
    ELEV_MAX = math.radians(85.0)
    ZOOM_IN = 0.85
    ZOOM_OUT = 1.0 / ZOOM_IN  # ~1.176
    RADIUS_MIN = 0.5
    RADIUS_MAX = 4.0
    SPEED_DN = 0.8
    SPEED_UP = 1.25
    SPEED_MIN = 0.1
    SPEED_MAX = 4.0

    def _apply_camera():
        eye = _eye_from_orbit(orbit[0], orbit[1], orbit[2])
        env.viewport_camera_controller.update_view_location(eye=eye, lookat=(0.0, 0.0, 0.0))

    # ---- Keyboard callbacks -----------------------------------------------

    should_reset = [False]

    def request_reset():
        should_reset[0] = True

    def orbit_left():
        orbit[0] -= ORBIT_STEP
        _apply_camera()

    def orbit_right():
        orbit[0] += ORBIT_STEP
        _apply_camera()

    def elev_down():
        orbit[1] = max(ELEV_MIN, orbit[1] - ELEV_STEP)
        _apply_camera()

    def elev_up():
        orbit[1] = min(ELEV_MAX, orbit[1] + ELEV_STEP)
        _apply_camera()

    def zoom_in():
        orbit[2] = max(RADIUS_MIN, orbit[2] * ZOOM_IN)
        _apply_camera()

    def zoom_out():
        orbit[2] = min(RADIUS_MAX, orbit[2] * ZOOM_OUT)
        _apply_camera()

    def speed_down():
        speed_scale[0] = max(SPEED_MIN, speed_scale[0] * SPEED_DN)
        pusher_term.set_speed_scale(speed_scale[0])
        print(f"[Push-T] speed scale → {speed_scale[0]:.2f}×")

    def speed_up():
        speed_scale[0] = min(SPEED_MAX, speed_scale[0] * SPEED_UP)
        pusher_term.set_speed_scale(speed_scale[0])
        print(f"[Push-T] speed scale → {speed_scale[0]:.2f}×")

    teleop.add_callback("R", request_reset)
    teleop.add_callback("N", orbit_left)
    teleop.add_callback("M", orbit_right)
    teleop.add_callback("COMMA", elev_down)
    teleop.add_callback("PERIOD", elev_up)
    teleop.add_callback("PAGE_UP", zoom_in)
    teleop.add_callback("PAGE_DOWN", zoom_out)
    teleop.add_callback("LEFT_BRACKET", speed_down)
    teleop.add_callback("RIGHT_BRACKET", speed_up)

    # -----------------------------------------------------------------------
    # Start
    # -----------------------------------------------------------------------
    env.reset()
    teleop.reset()

    # Grab the action term so we can push live yaw + speed into it each frame.
    pusher_term = env.action_manager.get_term("pusher")
    pusher_term.set_speed_scale(speed_scale[0])

    print(teleop)
    print(
        "\nPush-T teleoperation started.\n"
        "  Arrow / Numpad  — move Franka EE in table plane (camera-aligned)\n"
        "  Z / Numpad 7    — EE up\n"
        "  X / Numpad 9    — EE down\n"
        "  R               — reset episode\n"
        "  N / M           — orbit camera left / right\n"
        "  , / .           — orbit camera down / up\n"
        "  PgUp / PgDn     — zoom in / out\n"
        "  [ / ]           — decrease / increase EE speed\n"
        f"  Initial speed scale: {speed_scale[0]:.2f}×\n"
    )

    step_dt = 1.0 / args_cli.step_hz
    next_step_time = time.time()

    with torch.inference_mode():
        while simulation_app.is_running():

            # ---- Live camera-azimuth → direction alignment ----------------
            az = _read_viewport_azimuth()
            if az is not None:
                pusher_term.set_control_yaw(az)

            # ---- Step the environment ------------------------------------
            action = teleop.advance().repeat(env.num_envs, 1)
            env.step(action)

            # ---- Handle reset request ------------------------------------
            if should_reset[0]:
                env.reset()
                teleop.reset()
                pusher_term.set_speed_scale(speed_scale[0])
                should_reset[0] = False

            # ---- Pace the loop -------------------------------------------
            next_step_time += step_dt
            while time.time() < next_step_time and simulation_app.is_running():
                time.sleep(min(0.005, next_step_time - time.time()))
                env.sim.render()
            if env.sim.is_stopped():
                break

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
