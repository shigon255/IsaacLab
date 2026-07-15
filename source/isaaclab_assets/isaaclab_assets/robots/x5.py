# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the ARX X5 dual-arm robot (RoboDojo's native robot, phys-vidsim
exp robodojo-scene-collection #18).

The following configurations are available:

* :obj:`X5_CFG`: Single X5 arm, default PD gains.
* :obj:`X5_HIGH_PD_CFG`: Same robot with stiffer PD control for differential-IK use
  (mirrors isaaclab_assets.robots.aloha's VX300S_HIGH_PD_CFG / .franka's
  FRANKA_PANDA_HIGH_PD_CFG -- every scene wired so far in this fork drives its arm via
  FrankaEEPusherActionCfg's "pose"-mode differential IK, which needs this stiffer
  variant, not the base one).

The RoboDojo push_T-analogue rig (phys-vidsim's robodojo_pusht, in progress) uses two of
these arms facing each other, same pattern as ALOHA's bimanual rig -- each arm is
instantiated separately so IsaacLab's Articulation API can address them independently.

Joint names (from Assets/Robots/x5/X5A.urdf, the HF dataset RoboDojo-Benchmark/RoboDojo's
own robot bundle -- NOT real-stanford/arx5-sdk's X5.urdf, which turned out to be arm-only,
its gripper mount a *fixed* joint with no actuated DOF; see phys-vidsim's CLAUDE.md
External Dependencies entry and exps/current/robodojo-scene-collection/status.md for the
full discovery writeup):
  Arm   : joint1, joint2, joint3, joint4, joint5, joint6
  Finger: joint7, joint8 (prismatic, symmetric, stroke [0, 0.044])

End-effector body name: ``link6`` (no dedicated gripper_link/EE body -- joint7/joint8's
fingers attach directly to link6, same convention phys-vidsim's Genesis-side
simulation/sim_common/robodojo_x5.py uses for its own IK target link; this is the
single-source-of-truth URDF feeding BOTH backends, so joint/link naming here must stay in
sync with that module).

CAVEATS (real, unresolved as of 2026-07-15 -- read before using in a scene):
- home joint_pos below matches phys-vidsim's Genesis-side X5_HOME_QPOS exactly (same
  physical starting pose on both backends), but that value is itself an unverified first
  guess (mild elbow bend) -- needs empirical tuning once teleoperated for real, on
  either backend.
- effort_limit_sim values are carried over from X5A.urdf's own placeholder
  ``effort="100"`` on every joint (a SolidWorks-exporter default, not a real physical
  torque limit) -- same caveat as that URDF's joint lower/upper limits.
- stiffness/damping chosen to match phys-vidsim's Genesis-side tuning exactly (arm:
  X5_ARM_KP/KV = 2000/100 in simulation/sim_common/robodojo_x5.py; gripper: ArmSpec's
  default grip_kp/grip_kv = 2500/80), both first-guess-but-verified-stable values (the
  Genesis smoke test confirmed no NaN/inf under IK+grip stepping at these gains), not
  independently retuned for Isaac's own PhysX solver.

Reference:
  https://huggingface.co/datasets/RoboDojo-Benchmark/RoboDojo
  https://github.com/real-stanford/arx5-sdk
"""

from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

# Path to the locally-converted X5 USD, resolved relative to THIS repo checkout (same
# portability reasoning as aloha.py's _VX300S_USD_PATH -- never a hardcoded
# machine-specific absolute path). _x5_usd/ is gitignored (not committed, matching this
# repo's own **/*.usd convention) -- run phys-vidsim's scripts/setup_x5_usd.sh to
# populate it at THIS repo's root before using X5_CFG/X5_HIGH_PD_CFG.
_REPO_ROOT = Path(__file__).resolve().parents[4]
_X5_USD_PATH = str(_REPO_ROOT / "_x5_usd" / "x5.usd")

X5_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=_X5_USD_PATH,
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
            "joint1": 0.0,
            "joint2": 0.3,
            "joint3": -1.0,
            "joint4": 0.5,
            "joint5": 0.0,
            "joint6": 0.0,
            "joint7": 0.022,
            "joint8": 0.022,
        },
    ),
    actuators={
        "arm": ImplicitActuatorCfg(
            joint_names_expr=["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"],
            effort_limit_sim=100.0,
            stiffness=2000.0,
            damping=100.0,
        ),
        "gripper": ImplicitActuatorCfg(
            joint_names_expr=["joint7", "joint8"],
            effort_limit_sim=100.0,
            stiffness=2500.0,
            damping=80.0,
        ),
    },
)
"""Configuration of a single ARX X5 arm."""


X5_HIGH_PD_CFG = X5_CFG.copy()
X5_HIGH_PD_CFG.spawn.rigid_props.disable_gravity = True
X5_HIGH_PD_CFG.actuators["arm"].stiffness = 4000.0
X5_HIGH_PD_CFG.actuators["arm"].damping = 200.0
"""ARX X5 with stiffer PD control for task-space differential-IK."""
