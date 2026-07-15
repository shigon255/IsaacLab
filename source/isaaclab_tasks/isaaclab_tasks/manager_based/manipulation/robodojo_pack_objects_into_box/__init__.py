# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Bimanual ARX X5 Pack-Objects-Into-Box environment (phys-vidsim robodojo-scene-expansion #29)."""

import gymnasium as gym


gym.register(
    id="Isaac-RobodojoPackObjectsIntoBox-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.robodojo_pack_objects_into_box_env_cfg:RobodojoPackObjectsIntoBoxEnvCfg",
    },
)
