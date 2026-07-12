# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Single-arm Franka cube-stacking environment (phys-vidsim isaac-lab-scene-expansion, exp 20)."""

import gymnasium as gym

gym.register(
    id="Isaac-FrankaStack-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.franka_stack_env_cfg:FrankaStackEnvCfg",
    },
)
