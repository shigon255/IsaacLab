# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Bimanual ARX X5 Hang-Mugs environment (phys-vidsim robodojo-scene-expansion #29, M2)."""

import gymnasium as gym


gym.register(
    id="Isaac-RobodojoHangMugs-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.robodojo_hang_mugs_env_cfg:RobodojoHangMugsEnvCfg",
    },
)
