# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Single-arm UR5e push-clutter environment: reproduces the shape of phys-vidsim's
Genesis-side simulation/ur5e_pickplace/scene.py::Ur5ePickPlaceScene (UR5e + real
robosuite objects cereal/can/milk/bread, no gripper, push-only) as an Isaac Lab
manager-based env, for isaac-lab-multi-scene (#21)'s M2.

Robot: UR5E_HIGH_PD_CFG (isaaclab_assets.robots.ur5e, new this pass -- converted from
Genesis's OWN bundled UR5e MJCF via scripts/setup_ur5e_assets.sh; see that script's and
ur5e.py's own docstrings for why Genesis's bundled copy, not MuJoCo Menagerie's, is the
conversion source). Arm control reuses ../pusht/mdp/actions.py::FrankaEEPusherActionCfg
AS-IS (enable_z=True, lock_orientation=True for a fixed top-down approach) -- NO
wrist-yaw control this pass despite Genesis's own ArmSpec.enable_yaw=True: confirmed
live (2026-07-14) that FrankaEEPusherAction has no yaw action channel at all
(action_dim is only ever 2 or 3; control_yaw_offset is a fixed translation-frame
rotation constant, not a runtime orientation command) -- a pre-existing, documented Isaac-
side gap (see exps/archived/isaac-lab-scene-expansion/status.md's franka_stack
flipbox deferral for the precedent), not something newly introduced here. No gripper
term either -- the bundled UR5e MJCF has none, matching Genesis's own no-gripper
ArmSpec exactly (bare `ee_virtual_link` contact point).

Objects: cereal/can/milk/bread -- robosuite's REAL meshed objects (not procedural
primitives), converted via scripts/setup_robosuite_object_assets.sh's MeshConverter
path (IsaacLab's dedicated OBJ->USD converter -- NOT the MJCF path, which cannot
ingest these meshes at all; see convert_mesh_to_usd.py's docstring for the full
root-cause writeup). Each is a plain RigidObjectCfg wrapping the converted USD via
sim_utils.UsdFileCfg, with physics supplied here (rigid/collision/mass props), same
pattern franka_stack_env_cfg.py's procedural CuboidCfg objects use for their own
physics config -- just swapping the spawn source from a procedural primitive to a
converted mesh file. Object masses are REASONED approximations (grocery-item scale),
not measured -- robosuite's own fragment only specifies a material density, not an
absolute mass, and this repo has no live way to weigh the actual mesh volume yet;
flag for a live sanity check (does each object rest/settle plausibly, not sink through
the table or fly off it).

Object init X/Y: Isaac's own layout (2x2 grid in front of the arm, table centered at
X=0.35 -- same table-placement precedent franka_stack_env_cfg.py established), NOT
copied from Genesis's own (0,0)-centered layout -- physically distinct rigs, not
pixel-identical twins (see scenes/pusht.py's module docstring for this precedent).
Object init Z, however, IS copied directly from Genesis's own simulation/
ur5e_pickplace/scene.py::_OBJECT_INIT_POS -- those values are each object's own
intrinsic resting height (verified there via live get_AABB() calls against the SAME
underlying mesh this file also loads), independent of which world frame the scene
places the table in.

Camera: fixed_cam mirrors ../pusht/pusht_bimanual_env_cfg.py's CameraCfg EXACTLY (same
top-down rot=(0,1,0,0) static default) -- same "don't hand-derive an oblique
quaternion, tune the teleop viewpoint-orbit default instead" precedent
franka_stack_env_cfg.py/aloha_blocks_env_cfg.py both established.
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
from isaaclab_assets.robots.ur5e import UR5E_HIGH_PD_CFG

from ..pusht.mdp.actions import FrankaEEPusherActionCfg

# ---- Shared constants -- mirror simulation/ur5e_pickplace/scene.py's own OBJECT_NAMES
# and Z-heights exactly (kept in sync manually; there's no Genesis-free import path to
# that module since it pulls in `genesis` at module level -- see
# simulation/isaac_backend/scenes/ur5e_pickplace.py's matching comment). ----
_OBJECT_NAMES: tuple[str, ...] = ("cereal", "can", "milk", "bread")
_OBJECT_USD_NAME: dict[str, str] = {name: name for name in _OBJECT_NAMES}
_OBJECT_PRIM_NAME: dict[str, str] = {"cereal": "Cereal", "can": "Can", "milk": "Milk", "bread": "Bread"}
# Z copied from Genesis's own _OBJECT_INIT_POS (each object's own intrinsic resting
# height, verified there via get_AABB() against the same mesh) -- X/Y are Isaac's own
# 2x2-grid layout in front of the arm (see module docstring).
_OBJECT_INIT_POS: dict[str, tuple[float, float, float]] = {
    "cereal": (0.40, -0.10, 0.078),
    "can":    (0.40,  0.05, 0.0433),
    "milk":   (0.50, -0.10, 0.064),
    "bread":  (0.50,  0.05, 0.0263),
}
_OBJECT_INIT_ROT = (1.0, 0.0, 0.0, 0.0)  # identity
# Reasoned grocery-item-scale masses (kg) -- NOT measured, see module docstring.
_OBJECT_MASS: dict[str, float] = {"cereal": 0.3, "can": 0.15, "milk": 0.5, "bread": 0.1}

_REPO_ROOT_USD_DIR = "_robosuite_usd"


def _object_cfg(name: str) -> RigidObjectCfg:
    from pathlib import Path

    repo_root = Path(__file__).resolve().parents[6]  # .../submodules/IsaacLab (or the dev clone root)
    # Per-object SUBDIRECTORY (_robosuite_usd/<name>/<name>.usd), not a shared flat
    # _robosuite_usd/<name>.usd -- see scripts/setup_robosuite_object_assets.sh's own
    # comment for why: MeshConverter writes its instanceable-geometry sublayer to a
    # FIXED relative path inside whatever usd_dir it's given, so objects sharing one
    # usd_dir silently clobber each other's geometry (confirmed live via a real
    # teleop crash -- RigidBodyAPI resolution failed for every object except the
    # last one converted).
    usd_path = str(repo_root / _REPO_ROOT_USD_DIR / _OBJECT_USD_NAME[name] / f"{_OBJECT_USD_NAME[name]}.usd")
    return RigidObjectCfg(
        prim_path=f"{{ENV_REGEX_NS}}/{_OBJECT_PRIM_NAME[name]}",
        init_state=RigidObjectCfg.InitialStateCfg(pos=_OBJECT_INIT_POS[name], rot=_OBJECT_INIT_ROT),
        spawn=sim_utils.UsdFileCfg(
            usd_path=usd_path,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False, max_depenetration_velocity=5.0,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
            mass_props=sim_utils.MassPropertiesCfg(mass=_OBJECT_MASS[name]),
        ),
    )


@configclass
class Ur5ePickPlaceSceneCfg(InteractiveSceneCfg):
    """Single UR5e + table + cereal/can/milk/bread + fixed_cam."""

    robot = UR5E_HIGH_PD_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

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

    cereal = _object_cfg("cereal")
    can = _object_cfg("can")
    milk = _object_cfg("milk")
    bread = _object_cfg("bread")

    fixed_cam = CameraCfg(
        prim_path="{ENV_REGEX_NS}/FixedCamera",
        update_period=0.0,
        update_latest_camera_pose=True,
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
    """Single-arm EE-velocity pusher, no gripper -- matches Genesis's own no-gripper
    UR5e ArmSpec exactly."""

    arm = FrankaEEPusherActionCfg(
        asset_name="robot",
        arm_joint_names=["shoulder_pan", "shoulder_lift", "elbow", "wrist_1", "wrist_2", "wrist_3"],
        # NOT "ee_virtual_link" -- Genesis's own EE body (source MJCF: a real <body>,
        # geometry-less, a child of wrist_3_link with a +0.1m/quat offset) -- confirmed
        # live (2026-07-14) the MJCF importer DROPS it entirely from the converted USD
        # (matching a conversion-time warning: "Neither inertial nor geometries where
        # specified for ee_virtual_link"; absent from a full Stage.Traverse() of the
        # result). "wrist_3_link" is the nearest REAL body -- a substitute, not exact:
        # the actual EE frame is offset ~0.1m from it (see ur5e.xml's own
        # pos="0 0.1 0" quat="-1 1 1 1" on ee_virtual_link). Isaac's EE-position
        # readout/pusher target will be off by that amount until this is revisited
        # (e.g. via a FrameTransformer offset, once one is needed for a task that
        # cares about absolute reach precision).
        ee_body_name="wrist_3_link",
        ee_z_height=0.25,
        velocity_scale=0.35,
        z_velocity_scale=0.20,
        z_workspace=(0.02, 0.5),
        workspace=((0.15, -0.35), (0.55, 0.35)),
        control_yaw_offset=0.0,
        default_target_xy=(0.35, 0.0),
        enable_z=True,
        lock_orientation=True,
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
    reset_cereal = EventTerm(
        func=base_mdp.reset_root_state_uniform, mode="reset",
        params={"pose_range": {}, "velocity_range": {}, "asset_cfg": SceneEntityCfg("cereal")},
    )
    reset_can = EventTerm(
        func=base_mdp.reset_root_state_uniform, mode="reset",
        params={"pose_range": {}, "velocity_range": {}, "asset_cfg": SceneEntityCfg("can")},
    )
    reset_milk = EventTerm(
        func=base_mdp.reset_root_state_uniform, mode="reset",
        params={"pose_range": {}, "velocity_range": {}, "asset_cfg": SceneEntityCfg("milk")},
    )
    reset_bread = EventTerm(
        func=base_mdp.reset_root_state_uniform, mode="reset",
        params={"pose_range": {}, "velocity_range": {}, "asset_cfg": SceneEntityCfg("bread")},
    )


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=base_mdp.time_out, time_out=True)


@configclass
class Ur5ePickPlaceEnvCfg(ManagerBasedRLEnvCfg):
    scene: Ur5ePickPlaceSceneCfg = Ur5ePickPlaceSceneCfg(num_envs=1, env_spacing=1.5, replicate_physics=True)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    commands = None
    rewards = None
    curriculum = None

    def __post_init__(self):
        # Matches phys-vidsim's Genesis-side cadence exactly -- see
        # ../pusht/pusht_bimanual_env_cfg.py's identical override for the full derivation.
        self.decimation = 10
        self.episode_length_s = 10.0
        self.sim.dt = 1.0 / 500.0
        self.sim.render_interval = self.decimation
        self.sim.physx.bounce_threshold_velocity = 0.01
        self.sim.physx.friction_correlation_distance = 0.00625
