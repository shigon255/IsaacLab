# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the bare KUKA LBR iiwa14 (phys-vidsim isaac-lab-multi-scene, #21, M3).

The following configurations are available:

* :obj:`KUKA_IIWA14_CFG`: Bare iiwa14, no gripper.
* :obj:`KUKA_IIWA14_HIGH_PD_CFG`: Same robot with stiffer PD control for
  differential-IK use (matches this repo's own VX300S_HIGH_PD_CFG/UR5E_HIGH_PD_CFG
  convention for every robot driven by FrankaEEPusherActionCfg).

IsaacLab ships NO built-in bare-iiwa14 config -- only `isaaclab_assets.robots.
kuka_allegro` (iiwa7 + an Allegro hand, entirely different joint names:
`iiwa7_joint_*`, not this robot). Converted from Genesis's OWN bundled KUKA URDF
(assets/urdf/kuka_iiwa/model.urdf, installed alongside genesis-world) via
phys-vidsim's scripts/setup_kuka_assets.sh -- joint names (`lbr_iiwa_joint_1..7`) and
EE link (`lbr_iiwa_link_7`) match simulation/kuka_nutassembly/scene.py's own
`_KUKA_JOINTS`/ee_link_name exactly, since both backends load the same source file.

No gripper -- the bundled URDF has none (consistent with every scene in this codebase
using this robot). Stiff PD gains (kp=2000/kv=100, effort_limit=300 -- the URDF's own
per-joint `<limit effort="300">`) are baked into the converted USD's own PhysX joint
drive at conversion time (scripts/setup_kuka_assets.sh's own --stiffness/--damping
flags to convert_urdf_to_usd.py) -- matching Genesis's own kuka_nutassembly/scene.py,
which sets the SAME kp=2000/kv=100 post-load via set_dofs_kp/set_dofs_kv because a
bare URDF (unlike a hand-authored MJCF) carries no tuned actuator gains at all. The
ImplicitActuatorCfg below repeats the same numbers explicitly (not relying on the
baked-in USD values alone) to match every other robot config in this file for
clarity/consistency.

Reference: Genesis's own bundled asset (genesis-world pip package,
assets/urdf/kuka_iiwa/model.urdf).
"""

from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

# Path to the locally-converted KUKA iiwa14 USD, resolved relative to THIS repo
# checkout -- same portability precedent as aloha.py's _VX300S_USD_PATH /
# ur5e.py's _UR5E_USD_PATH. _kuka_usd/ is gitignored -- run phys-vidsim's
# scripts/setup_kuka_assets.sh to populate it before using KUKA_IIWA14_CFG/
# KUKA_IIWA14_HIGH_PD_CFG.
_REPO_ROOT = Path(__file__).resolve().parents[4]
_KUKA_USD_PATH = str(_REPO_ROOT / "_kuka_usd" / "kuka_iiwa14.usd")

_KUKA_JOINTS = [f"lbr_iiwa_joint_{i}" for i in range(1, 8)]
# Matches simulation/kuka_nutassembly/scene.py's own _KUKA_HOME_QPOS exactly (a mild
# elbow bend, non-singular reach pose) -- both backends' teleop starts from the same
# joint configuration.
_HOME_QPOS = {
    "lbr_iiwa_joint_1": 0.0, "lbr_iiwa_joint_2": 0.3, "lbr_iiwa_joint_3": 0.0,
    "lbr_iiwa_joint_4": -1.2, "lbr_iiwa_joint_5": 0.0, "lbr_iiwa_joint_6": 1.0,
    "lbr_iiwa_joint_7": 0.0,
}

KUKA_IIWA14_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=_KUKA_USD_PATH,
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
        "arm": ImplicitActuatorCfg(
            joint_names_expr=_KUKA_JOINTS,
            effort_limit_sim=300.0,
            stiffness=2000.0,
            damping=100.0,
        ),
    },
)
"""Configuration of the bare KUKA iiwa14, matching Genesis's own set_dofs_kp/kv gains."""


KUKA_IIWA14_HIGH_PD_CFG = KUKA_IIWA14_CFG.copy()
KUKA_IIWA14_HIGH_PD_CFG.spawn.rigid_props.disable_gravity = True
KUKA_IIWA14_HIGH_PD_CFG.actuators["arm"].stiffness = 4000.0
KUKA_IIWA14_HIGH_PD_CFG.actuators["arm"].damping = 200.0
"""KUKA iiwa14 with stiffer PD control for task-space differential-IK."""
