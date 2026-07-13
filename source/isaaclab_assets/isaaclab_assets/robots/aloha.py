# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the ALOHA bimanual robot (single-arm ViperX 300s).

The following configurations are available:

* :obj:`VX300S_CFG`: Single ViperX 300s arm (the ALOHA follower arm) without a gripper actuator.
* :obj:`VX300S_HIGH_PD_CFG`: Same robot with stiffer PD control for differential-IK use.

The bimanual ALOHA rig uses two of these arms facing each other. Each arm is instantiated
separately so that IsaacLab's Articulation API can address them independently.

Joint names (from MuJoCo Menagerie ``trossen_vx300s/vx300s.xml``):
  Arm  : waist, shoulder, elbow, forearm_roll, wrist_angle, wrist_rotate
  Finger: left_finger, right_finger

End-effector body name: ``gripper_link``

Reference:
  https://github.com/google-deepmind/mujoco_menagerie/tree/main/trossen_vx300s
  https://aloha-2.github.io/
"""

from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

# Path to the locally-converted ViperX 300s USD, resolved relative to THIS repo checkout
# (not a hardcoded machine-specific absolute path -- that was a real portability bug: it
# silently worked only on the one machine/clone where it was first written, and would
# resolve to nothing on a fresh clone or a different checkout location, e.g.
# phys-vidsim's submodules/IsaacLab vs. this repo's own dev clone). _aloha_usd/ is
# gitignored (not committed, matching this repo's own **/*.usd convention) -- run
# phys-vidsim's scripts/setup_aloha_assets.sh (or manually: scripts/tools/convert_mjcf.py
# against a MuJoCo Menagerie checkout) to populate it at THIS repo's root before using
# VX300S_CFG/VX300S_HIGH_PD_CFG.
_REPO_ROOT = Path(__file__).resolve().parents[4]
_VX300S_USD_PATH = str(_REPO_ROOT / "_aloha_usd" / "vx300s" / "vx300s.usd")

VX300S_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=_VX300S_USD_PATH,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=0,
            fix_root_link=True,
        ),
        activate_contact_sensors=False,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            "waist": 0.0,
            "shoulder": -0.96,
            "elbow": 1.16,
            "forearm_roll": 0.0,
            "wrist_angle": -0.3,
            "wrist_rotate": 0.0,
            "left_finger": 0.035,
            "right_finger": -0.035,
        },
    ),
    actuators={
        "arm": ImplicitActuatorCfg(
            joint_names_expr=["waist", "shoulder", "elbow", "forearm_roll", "wrist_angle", "wrist_rotate"],
            effort_limit_sim={
                "waist": 35.0,
                "shoulder": 144.0,
                "elbow": 59.0,
                "forearm_roll": 22.0,
                "wrist_angle": 22.0,
                "wrist_rotate": 22.0,
            },
            stiffness=80.0,
            damping=4.0,
        ),
        "gripper": ImplicitActuatorCfg(
            joint_names_expr=["left_finger", "right_finger"],
            effort_limit_sim=5.0,
            stiffness=200.0,
            damping=10.0,
        ),
    },
)
"""Configuration of a single ViperX 300s arm (ALOHA follower arm)."""


VX300S_HIGH_PD_CFG = VX300S_CFG.copy()
VX300S_HIGH_PD_CFG.spawn.rigid_props.disable_gravity = True
VX300S_HIGH_PD_CFG.actuators["arm"].stiffness = 400.0
VX300S_HIGH_PD_CFG.actuators["arm"].damping = 80.0
"""ViperX 300s with stiffer PD control for task-space differential-IK."""
