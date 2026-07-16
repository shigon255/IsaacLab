# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Single-arm Franka cube-stacking environment: reproduces the shape of phys-vidsim's
Genesis-side simulation/franka_stack/scene.py::FrankaStackScene (single Panda + cubeA/
cubeB/flipbox) as an Isaac Lab manager-based env, for isaac-lab-scene-expansion (#20)'s
M3 scene generalization.

Robot: Isaac Lab's own built-in FRANKA_PANDA_CFG (isaaclab_assets.robots.franka) -- a
real, tested, Nucleus-hosted USD asset. Unlike the fork's ALOHA VX300S asset (_aloha_usd/,
committed under this repo's own gitignore exception), this one needs NO local asset
provisioning at all -- it downloads from Isaac Sim's Nucleus server like any other
built-in IsaacLab robot.

Arm control reuses ../pusht/mdp/actions.py::FrankaEEPusherActionCfg AS-IS (same
position-only differential-IK EE-velocity term already live-verified for bimanual
Push-T, enable_z=True for full XYZ control -- see that module's docstring). Gripper
control reuses IsaacLab's own core BinaryJointPositionActionCfg AS-IS. Deliberately NOT
adding wrist-yaw control this pass (Genesis's ArmSpec.enable_yaw, needed for flipbox's
flip task): that needs either full pose-mode differential IK or a hand joint-target
override on top of position-only IK, and either one is new, unverified control-law code
this repo has no live way to test yet -- see isaac-lab-scene-expansion (#20)'s status.md.
cubeA/cubeB stacking is driveable through this action set now; flipbox's flip is not.

Camera: fixed_cam mirrors ../pusht/pusht_bimanual_env_cfg.py's CameraCfg EXACTLY (same
4 data_types, update_latest_camera_pose=True, ROS convention) -- only `offset.pos` is
repositioned, reusing the SAME already-proven top-down rot=(0,1,0,0) rather than
deriving a new oblique-angle quaternion by hand (unverifiable without a live render;
isaac_backend's viewpoint-orbit teleop feature can reach an oblique angle live once
this scene is wired into collect_demos_isaac.py, the same way it does for Push-T).

Objects: cubeA/cubeB/flipbox are plain procedural CuboidCfg boxes (no external USD
asset, unlike the stack task's own blue_block.usd/red_block.usd), sized/colored to
match simulation/franka_stack/scene.py's own constants exactly (cubeA 0.02 half-size
red, cubeB 0.025 half-size green, flipbox 0.025 half-size blue). flipbox is a SOLID
blue box here, not the two-tone yellow/red-patch marked box Genesis's
marked_box_mjcf() produces -- USD per-face materials need more than a single
PreviewSurfaceCfg on one CuboidCfg prim; deferred alongside wrist-yaw (both are needed
together for the flip task to mean anything visually).
"""

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.envs import mdp as base_mdp
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import CameraCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg
from isaaclab.utils import configclass
from isaaclab_assets.robots.franka import FRANKA_PANDA_HIGH_PD_CFG

from ..pusht.mdp.actions import FrankaEEPusherActionCfg

# ---- Shared constants -- mirror simulation/franka_stack/scene.py's own values exactly
# (kept in sync manually; see simulation/isaac_backend/scenes/franka_stack.py's matching
# comment -- there's no Genesis-free import path to that module). ----
_CUBE_HALF_SIZE = {"cubeA": 0.02, "cubeB": 0.025, "flipbox": 0.025}
_CUBE_RGBA = {
    "cubeA": (0.85, 0.15, 0.15), "cubeB": (0.15, 0.75, 0.20), "flipbox": (0.2, 0.3, 0.85),
}
_CUBE_MASS = {"cubeA": 0.02, "cubeB": 0.03, "flipbox": 0.03}
# Table top at world Z=0 (matches Genesis's own franka_stack table convention); Panda
# base at env origin, home pose reaches +X -- cubes placed in front of it along +X,
# same layout logic as Genesis's own _CUBE_INIT_POS (exact numbers are Isaac's own,
# not copied -- see module docstring / scenes/pusht.py's established precedent).
_CUBE_INIT_POS = {
    "cubeA": (0.40, -0.05, _CUBE_HALF_SIZE["cubeA"]),
    "cubeB": (0.40, 0.05, _CUBE_HALF_SIZE["cubeB"]),
    "flipbox": (0.55, 0.0, _CUBE_HALF_SIZE["flipbox"]),
}
_CUBE_INIT_ROT = (1.0, 0.0, 0.0, 0.0)  # identity
# str.capitalize() would mangle "cubeA" -> "Cubea" (lowercases everything after the
# first char) -- explicit prim names instead.
_CUBE_PRIM_NAME = {"cubeA": "CubeA", "cubeB": "CubeB", "flipbox": "Flipbox"}


def _cube_cfg(name: str) -> RigidObjectCfg:
    half = _CUBE_HALF_SIZE[name]
    return RigidObjectCfg(
        prim_path=f"{{ENV_REGEX_NS}}/{_CUBE_PRIM_NAME[name]}",
        init_state=RigidObjectCfg.InitialStateCfg(pos=_CUBE_INIT_POS[name], rot=_CUBE_INIT_ROT),
        spawn=sim_utils.CuboidCfg(
            size=(half * 2, half * 2, half * 2),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False, max_depenetration_velocity=5.0,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
            mass_props=sim_utils.MassPropertiesCfg(mass=_CUBE_MASS[name]),
            physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=0.9, dynamic_friction=0.7),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=_CUBE_RGBA[name], roughness=0.6),
        ),
    )


@configclass
class FrankaStackSceneCfg(InteractiveSceneCfg):
    """Single Franka Panda + table + cubeA/cubeB/flipbox + fixed_cam."""

    robot = FRANKA_PANDA_HIGH_PD_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.35, 0.0, -0.025)),
        spawn=sim_utils.CuboidCfg(
            size=(1.0, 1.0, 0.05),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
            physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=1.0, dynamic_friction=0.8),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.42, 0.42, 0.38), roughness=0.8),
        ),
    )

    cubeA = _cube_cfg("cubeA")
    cubeB = _cube_cfg("cubeB")
    flipbox = _cube_cfg("flipbox")

    fixed_cam = CameraCfg(
        prim_path="{ENV_REGEX_NS}/FixedCamera",
        update_period=0.0,
        # See ../pusht/pusht_bimanual_env_cfg.py's identical field for the full
        # rationale -- required for isaac_backend's runtime camera repositioning
        # (viewpoint orbit / per-demo replay pose) to actually be readable back.
        update_latest_camera_pose=True,
        # 512x512, matching Genesis's convention (see ../pusht/pusht_bimanual_env_cfg.py's
        # identical field for the full rationale -- found live 2026-07-13 via run_eval.py's
        # GT/gen shape assertion against Wan's always-512 output).
        height=512,
        width=512,
        data_types=["rgb", "instance_id_segmentation_fast", "distance_to_image_plane", "normals"],
        colorize_instance_id_segmentation=False,
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=18.0,
            focus_distance=1.0,
            horizontal_aperture=20.955,
            clipping_range=(0.05, 3.0),
        ),
        # Top-down over the cube workspace (Isaac's own default, not Genesis's oblique
        # 3/4 view -- see module docstring for why an oblique quaternion isn't hand-derived
        # here). rot=(0,1,0,0) is the SAME proven top-down rotation pusht_bimanual_env_cfg.py
        # uses -- only the position is scene-specific.
        offset=CameraCfg.OffsetCfg(pos=(0.40, 0.0, 0.9), rot=(0.0, 1.0, 0.0, 0.0), convention="ros"),
    )

    ground = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -0.08)),
        spawn=GroundPlaneCfg(),
    )

    light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(color=(0.8, 0.8, 0.8), intensity=2500.0),
    )


@configclass
class ActionsCfg:
    """Arm EE-velocity pusher (reused as-is from Push-T, enable_z=True for full XYZ) +
    binary gripper (reused as-is from IsaacLab core). NO wrist-yaw term yet -- see
    module docstring."""

    arm = FrankaEEPusherActionCfg(
        asset_name="robot",
        arm_joint_names=[f"panda_joint{i}" for i in range(1, 8)],
        ee_body_name="panda_hand",
        ee_z_height=0.25,
        velocity_scale=0.035,
        z_velocity_scale=0.020,
        z_workspace=(0.02, 0.5),
        workspace=((0.15, -0.3), (0.6, 0.3)),
        control_yaw_offset=0.0,
        default_target_xy=(0.35, 0.0),
        enable_z=True,
    )

    gripper = base_mdp.BinaryJointPositionActionCfg(
        asset_name="robot",
        joint_names=["panda_finger_joint.*"],
        open_command_expr={"panda_finger_joint.*": 0.04},
        close_command_expr={"panda_finger_joint.*": 0.0},
    )


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        fixed_cam = ObsTerm(
            func=base_mdp.image,
            params={"sensor_cfg": SceneEntityCfg("fixed_cam"), "data_type": "rgb", "normalize": False},
        )
        actions = ObsTerm(func=base_mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    reset_robot = EventTerm(
        func=base_mdp.reset_joints_by_scale,
        mode="reset",
        params={"position_range": (1.0, 1.0), "velocity_range": (0.0, 0.0), "asset_cfg": SceneEntityCfg("robot")},
    )
    reset_cubeA = EventTerm(
        func=base_mdp.reset_root_state_uniform,
        mode="reset",
        params={"pose_range": {}, "velocity_range": {}, "asset_cfg": SceneEntityCfg("cubeA")},
    )
    reset_cubeB = EventTerm(
        func=base_mdp.reset_root_state_uniform,
        mode="reset",
        params={"pose_range": {}, "velocity_range": {}, "asset_cfg": SceneEntityCfg("cubeB")},
    )
    reset_flipbox = EventTerm(
        func=base_mdp.reset_root_state_uniform,
        mode="reset",
        params={"pose_range": {}, "velocity_range": {}, "asset_cfg": SceneEntityCfg("flipbox")},
    )


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=base_mdp.time_out, time_out=True)


@configclass
class FrankaStackEnvCfg(ManagerBasedRLEnvCfg):
    scene: FrankaStackSceneCfg = FrankaStackSceneCfg(num_envs=1, env_spacing=1.5, replicate_physics=True)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    commands = None
    rewards = None
    curriculum = None

    def __post_init__(self):
        # Matches phys-vidsim's Genesis-side cadence exactly -- see
        # ../pusht/pusht_bimanual_env_cfg.py's identical override for the full derivation
        # (dt=1/500, decimation=10 -> 50 Hz control, so frame_steps=3 gives 16.667 fps).
        self.decimation = 10
        self.episode_length_s = 10.0
        self.sim.dt = 1.0 / 500.0
        self.sim.render_interval = self.decimation
        self.sim.physx.bounce_threshold_velocity = 0.01
        self.sim.physx.friction_correlation_distance = 0.00625
