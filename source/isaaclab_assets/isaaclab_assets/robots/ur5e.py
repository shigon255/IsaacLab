# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the Universal Robots UR5e (phys-vidsim isaac-lab-multi-scene, #21, M2).

The following configurations are available:

* :obj:`UR5E_CFG`: UR5e with its own MJCF-sourced position-actuator gains.
* :obj:`UR5E_HIGH_PD_CFG`: Same robot with stiffer PD control for differential-IK use
  (matches this repo's own VX300S_HIGH_PD_CFG/FRANKA_PANDA_HIGH_PD_CFG convention for
  every robot driven by FrankaEEPusherActionCfg).

Converted from Genesis's OWN bundled UR5e MJCF (NOT MuJoCo Menagerie's upstream copy,
which uses different joint names with a "_joint" suffix and a different actuator/gain
scheme -- see phys-vidsim's scripts/setup_ur5e_assets.sh for the full rationale and the
conversion recipe). Joint names/gains below are read directly from that source MJCF
(assets/xml/universal_robots_ur5e/ur5e.xml, bundled with the installed genesis-world
pip package) so this config matches EXACTLY what phys-vidsim's Genesis-side
simulation/ur5e_pickplace/scene.py already assumes -- both backends load the same
joint names, and gains are the file's own tuned values (its "position"/"position_small"
default actuator classes: kp=2000/kv=100/effort=150 for shoulder_pan/shoulder_lift/
elbow, kp=500/kv=25/effort=28 for wrist_1/wrist_2/wrist_3), not guessed.

No gripper -- the bundled MJCF has none (consistent with every scene in this codebase
using this robot; the bare `ee_virtual_link`, a purpose-built tool-center-point frame,
is the contact point, matching simulation/ur5e_pickplace/scene.py's own ArmSpec).

Reference: Genesis's own bundled asset (genesis-world pip package,
assets/xml/universal_robots_ur5e/ur5e.xml).
"""

from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

# Path to the locally-converted UR5e USD, resolved relative to THIS repo checkout --
# same portability precedent as aloha.py's _VX300S_USD_PATH (never a hardcoded
# machine-specific absolute path). _ur5e_usd/ is gitignored (matching this repo's own
# **/*.usd convention) -- run phys-vidsim's scripts/setup_ur5e_assets.sh to populate it
# at THIS repo's root before using UR5E_CFG/UR5E_HIGH_PD_CFG.
_REPO_ROOT = Path(__file__).resolve().parents[4]
_UR5E_USD_PATH = str(_REPO_ROOT / "_ur5e_usd" / "ur5e.usd")

# Home pose -- matches simulation/ur5e_pickplace/scene.py's own _UR5E_HOME_QPOS exactly
# (a non-singular, centrally-reaching pose), so both backends' teleop starts from the
# same joint configuration.
_HOME_QPOS = {
    "shoulder_pan": -1.5708,
    "shoulder_lift": -1.5708,
    "elbow": 1.5708,
    "wrist_1": -1.5708,
    "wrist_2": -1.5708,
    "wrist_3": 0.0,
}

UR5E_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=_UR5E_USD_PATH,
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
    init_state=ArticulationCfg.InitialStateCfg(joint_pos=_HOME_QPOS),
    actuators={
        "shoulder": ImplicitActuatorCfg(
            joint_names_expr=["shoulder_pan", "shoulder_lift", "elbow"],
            effort_limit_sim=150.0,
            stiffness=2000.0,
            damping=100.0,
        ),
        "wrist": ImplicitActuatorCfg(
            joint_names_expr=["wrist_1", "wrist_2", "wrist_3"],
            effort_limit_sim=28.0,
            stiffness=500.0,
            damping=25.0,
        ),
    },
)
"""Configuration of the UR5e with its own MJCF-sourced actuator gains."""


UR5E_HIGH_PD_CFG = UR5E_CFG.copy()
UR5E_HIGH_PD_CFG.spawn.rigid_props.disable_gravity = True
UR5E_HIGH_PD_CFG.actuators["shoulder"].stiffness = 4000.0
UR5E_HIGH_PD_CFG.actuators["shoulder"].damping = 200.0
UR5E_HIGH_PD_CFG.actuators["wrist"].stiffness = 1000.0
UR5E_HIGH_PD_CFG.actuators["wrist"].damping = 50.0
"""UR5e with stiffer PD control for task-space differential-IK."""
