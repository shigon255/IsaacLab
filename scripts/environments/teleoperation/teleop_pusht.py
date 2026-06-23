# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Keyboard teleoperation for the Push-T environment."""

from __future__ import annotations

import argparse
import time

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Teleoperate the Push-T tabletop task.")
parser.add_argument("--task", type=str, default="Isaac-PushT-v0", help="Task name.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments.")
parser.add_argument("--step_hz", type=int, default=30, help="Teleoperation step rate.")
parser.add_argument("--sensitivity", type=float, default=0.6, help="Keyboard motion sensitivity.")
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


def _set_viewport_camera():
    try:
        from omni.kit.viewport.utility import get_viewport_from_window_name

        viewport = get_viewport_from_window_name("Viewport")
        if viewport is not None:
            viewport.set_active_camera("/World/envs/env_0/FixedCamera")
    except Exception:
        pass


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
            omega_z_sensitivity=0.0,
            sim_device=env.device,
        )
    )

    should_reset = False

    def request_reset():
        nonlocal should_reset
        should_reset = True

    teleop.add_callback("R", request_reset)

    env.reset()
    teleop.reset()
    _set_viewport_camera()
    print(teleop)
    print("Push-T teleoperation started. Use arrow keys or numpad to move the blue pusher. Press R to reset.")

    step_dt = 1.0 / args_cli.step_hz
    next_step_time = time.time()
    with torch.inference_mode():
        while simulation_app.is_running():
            action = teleop.advance().repeat(env.num_envs, 1)
            env.step(action)

            if should_reset:
                env.reset()
                teleop.reset()
                should_reset = False
                _set_viewport_camera()

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
