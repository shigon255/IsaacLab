# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Bimanual ARX X5 Play-Tic-Tac-Toe environment (phys-vidsim robodojo-scene-expansion #29,
M2) -- RoboDojo's `play_tic_tac_toe` task. Reuses ../robodojo_pusht/robodojo_pusht_env_cfg.py's
exact robot/table/camera setup unchanged (same dual-X5 rig, same mount poses), matching
every other robodojo_* env cfg in this fork; a different shape family from every other scene
in this pass -- a flat static board + 9 small game pieces (5 player "O" rings, 4 opponent
"X" pieces, drawn from chessman index 0/1 respectively).

The checkerboard is the FIRST `Geometry`-typed (fixed, non-free-jointed) real RoboDojo mesh
in this fork -- mounted as `AssetBaseCfg` + `UsdFileCfg` (like the ground plane / table),
NOT `RigidObjectCfg` (which every other, free-jointed `Rigid` object in this fork uses).

Object assets staged into this repo's gitignored `_robodojo_object_usd/`. RoboDojo's export
pipeline ships PhysicsRigidBodyAPI + PhysicsCollisionAPI baked into every object.usdz
(confirmed for the bowl in #18, holds for every RoboDojo asset since they share one export
pipeline) -- no MeshConverter physics-baking step needed for the movable pieces, same plain
UsdFileCfg(usd_path=...) as every other robodojo_* object; the static board doesn't need
rigid-body physics at all (a fixed AssetBaseCfg, like the table) -- and correspondingly
doesn't need a `mass_props` override either.

Each piece's `mass_props` (2026-07-17, phys-vidsim `physics-time-calibration` #33 deliverable
4) is a real-world target mass, matching `simulation/sim_common/physics_defaults.py`'s
`ROBODOJO_TARGET_MASS_KG["chessman"]` -- applied uniformly to both piece types (player/
opponent are the same real-world identity, just different chessman mesh indices). See
`robodojo_stack_bowls_env_cfg.py`'s docstring for why this works on an already-baked
referenced prim.

Object initial positions match the Genesis side exactly
(`simulation/robodojo_play_tic_tac_toe/scene.py`'s `BOARD_POS`/`PIECE_INIT_POS`) -- taken
directly from `play_tic_tac_toe.yml`'s own per-piece xlim/ylim (already-fixed single-value
ranges, not randomized pools).

No custom success/out-of-bounds termination -- construction + condition-rendering scope
only, matching every other robodojo_* env cfg in this fork.
"""

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.devices import DevicesCfg
from isaaclab.devices.keyboard import Se2KeyboardCfg
from isaaclab.envs import ManagerBasedRLEnvCfg, ViewerCfg
from isaaclab.envs import mdp as base_mdp
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import CameraCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg
from isaaclab.utils import configclass
from isaaclab_assets.robots.x5 import X5_HIGH_PD_CFG

from ..pusht.mdp.actions import FrankaEEPusherActionCfg

# 180 deg yaw around Z, (w,x,y,z) -- same as robodojo_pusht_env_cfg.py.
_ROT_180_Z = (0.0, 0.0, 0.0, 1.0)

_ARM_JOINT_NAMES = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]
_FINGER_JOINT_NAMES = ["joint7", "joint8"]
_EE_BODY_NAME = "link6"

_REPO_ROOT_USD_DIR = "_robodojo_object_usd"
_BOARD_STAGED_DIR = "checkerboard_00000"
_BOARD_POS = (0.0, -0.075, 0.0)

# Matches simulation/robodojo_play_tic_tac_toe/scene.py's PIECE_INDEX/PIECE_INIT_POS
# exactly -- (staged chessman index dir, prim name, init pos).
_PIECE_SPECS: dict[str, tuple[str, str, tuple[float, float, float]]] = {
    "player_piece0":   ("chessman_00000", "PlayerPiece0",   (-0.2, -0.25, 0.02)),
    "player_piece1":   ("chessman_00000", "PlayerPiece1",   (-0.1, -0.25, 0.02)),
    "player_piece2":   ("chessman_00000", "PlayerPiece2",   (0.0,  -0.25, 0.02)),
    "player_piece3":   ("chessman_00000", "PlayerPiece3",   (0.1,  -0.25, 0.02)),
    "player_piece4":   ("chessman_00000", "PlayerPiece4",   (0.2,  -0.25, 0.02)),
    "opponent_piece0": ("chessman_00001", "OpponentPiece0", (-0.15, 0.2,  0.02)),
    "opponent_piece1": ("chessman_00001", "OpponentPiece1", (-0.05, 0.2,  0.02)),
    "opponent_piece2": ("chessman_00001", "OpponentPiece2", (0.05,  0.2,  0.02)),
    "opponent_piece3": ("chessman_00001", "OpponentPiece3", (0.15,  0.2,  0.02)),
}
_PIECE_INIT_ROT = (1.0, 0.0, 0.0, 0.0)

# Real-world target mass (kg) -- matches phys-vidsim's sim_common/physics_defaults.py
# ROBODOJO_TARGET_MASS_KG["chessman"] exactly (a small game piece / peg).
_PIECE_MASS_KG = 0.02


def _usd_path(staged_dir: str) -> str:
    from pathlib import Path

    repo_root = Path(__file__).resolve().parents[6]  # .../submodules/IsaacLab (or the dev clone root)
    return str(repo_root / _REPO_ROOT_USD_DIR / staged_dir / "object.usdz")


def _piece_cfg(name: str) -> RigidObjectCfg:
    staged_dir, prim_name, pos = _PIECE_SPECS[name]
    return RigidObjectCfg(
        prim_path=f"{{ENV_REGEX_NS}}/{prim_name}",
        init_state=RigidObjectCfg.InitialStateCfg(pos=pos, rot=_PIECE_INIT_ROT),
        spawn=sim_utils.UsdFileCfg(
            usd_path=_usd_path(staged_dir),
            mass_props=sim_utils.MassPropertiesCfg(mass=_PIECE_MASS_KG),
        ),
    )


@configclass
class RobodojoPlayTicTacToeSceneCfg(InteractiveSceneCfg):
    """Tabletop bimanual Play-Tic-Tac-Toe scene using two ARX X5 arms + a static checkerboard
    + 9 movable game pieces drawn from 2 distinct real chessman meshes."""

    robot_left: ArticulationCfg = X5_HIGH_PD_CFG.replace(
        prim_path="{ENV_REGEX_NS}/RobotLeft",
        init_state=X5_HIGH_PD_CFG.init_state.replace(
            pos=(-0.35, 0.0, 0.0),
            rot=_ROT_180_Z,
        ),
    )

    robot_right: ArticulationCfg = X5_HIGH_PD_CFG.replace(
        prim_path="{ENV_REGEX_NS}/RobotRight",
        init_state=X5_HIGH_PD_CFG.init_state.replace(
            pos=(0.35, 0.0, 0.0),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )

    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -0.025)),
        spawn=sim_utils.CuboidCfg(
            size=(1.0, 1.0, 0.05),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
            physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=1.0, dynamic_friction=0.8),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.42, 0.42, 0.38), roughness=0.8),
        ),
    )

    # Static Geometry-typed board -- fixed AssetBaseCfg + UsdFileCfg, like the table itself,
    # NOT RigidObjectCfg (no free-body physics needed -- see module docstring).
    board = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Board",
        init_state=AssetBaseCfg.InitialStateCfg(pos=_BOARD_POS),
        spawn=sim_utils.UsdFileCfg(usd_path=_usd_path(_BOARD_STAGED_DIR)),
    )

    player_piece0 = _piece_cfg("player_piece0")
    player_piece1 = _piece_cfg("player_piece1")
    player_piece2 = _piece_cfg("player_piece2")
    player_piece3 = _piece_cfg("player_piece3")
    player_piece4 = _piece_cfg("player_piece4")
    opponent_piece0 = _piece_cfg("opponent_piece0")
    opponent_piece1 = _piece_cfg("opponent_piece1")
    opponent_piece2 = _piece_cfg("opponent_piece2")
    opponent_piece3 = _piece_cfg("opponent_piece3")

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
        offset=CameraCfg.OffsetCfg(pos=(0.0, 0.0, 0.90), rot=(0.0, 1.0, 0.0, 0.0), convention="ros"),
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
    """Identical wiring to robodojo_pusht_env_cfg.py's ActionsCfg -- same rig, same
    EE-velocity pusher + binary gripper per side."""

    pusher_left = FrankaEEPusherActionCfg(
        asset_name="robot_left",
        arm_joint_names=_ARM_JOINT_NAMES,
        ee_body_name=_EE_BODY_NAME,
        ee_z_height=0.03,
        velocity_scale=0.035,
        z_velocity_scale=0.020,
        z_workspace=(0.01, 0.40),
        workspace=((-0.35, -0.25), (0.15, 0.25)),
        control_yaw_offset=0.0,
        default_target_xy=(-0.15, 0.15),
        enable_z=True,
        lock_orientation=True,
    )

    gripper_left = base_mdp.BinaryJointPositionActionCfg(
        asset_name="robot_left",
        joint_names=_FINGER_JOINT_NAMES,
        open_command_expr={"joint7": 0.044, "joint8": 0.044},
        close_command_expr={"joint7": 0.0, "joint8": 0.0},
    )

    pusher_right = FrankaEEPusherActionCfg(
        asset_name="robot_right",
        arm_joint_names=_ARM_JOINT_NAMES,
        ee_body_name=_EE_BODY_NAME,
        ee_z_height=0.03,
        velocity_scale=0.035,
        z_velocity_scale=0.020,
        z_workspace=(0.01, 0.40),
        workspace=((-0.15, -0.25), (0.35, 0.25)),
        control_yaw_offset=0.0,
        default_target_xy=(0.15, 0.15),
        enable_z=True,
        lock_orientation=True,
    )

    gripper_right = base_mdp.BinaryJointPositionActionCfg(
        asset_name="robot_right",
        joint_names=_FINGER_JOINT_NAMES,
        open_command_expr={"joint7": 0.044, "joint8": 0.044},
        close_command_expr={"joint7": 0.0, "joint8": 0.0},
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
    reset_robot_left = EventTerm(
        func=base_mdp.reset_joints_by_scale,
        mode="reset",
        params={"position_range": (1.0, 1.0), "velocity_range": (0.0, 0.0), "asset_cfg": SceneEntityCfg("robot_left")},
    )
    reset_robot_right = EventTerm(
        func=base_mdp.reset_joints_by_scale,
        mode="reset",
        params={"position_range": (1.0, 1.0), "velocity_range": (0.0, 0.0), "asset_cfg": SceneEntityCfg("robot_right")},
    )
    reset_player_piece0 = EventTerm(
        func=base_mdp.reset_root_state_uniform,
        mode="reset",
        params={"pose_range": {}, "velocity_range": {}, "asset_cfg": SceneEntityCfg("player_piece0")},
    )
    reset_player_piece1 = EventTerm(
        func=base_mdp.reset_root_state_uniform,
        mode="reset",
        params={"pose_range": {}, "velocity_range": {}, "asset_cfg": SceneEntityCfg("player_piece1")},
    )
    reset_player_piece2 = EventTerm(
        func=base_mdp.reset_root_state_uniform,
        mode="reset",
        params={"pose_range": {}, "velocity_range": {}, "asset_cfg": SceneEntityCfg("player_piece2")},
    )
    reset_player_piece3 = EventTerm(
        func=base_mdp.reset_root_state_uniform,
        mode="reset",
        params={"pose_range": {}, "velocity_range": {}, "asset_cfg": SceneEntityCfg("player_piece3")},
    )
    reset_player_piece4 = EventTerm(
        func=base_mdp.reset_root_state_uniform,
        mode="reset",
        params={"pose_range": {}, "velocity_range": {}, "asset_cfg": SceneEntityCfg("player_piece4")},
    )
    reset_opponent_piece0 = EventTerm(
        func=base_mdp.reset_root_state_uniform,
        mode="reset",
        params={"pose_range": {}, "velocity_range": {}, "asset_cfg": SceneEntityCfg("opponent_piece0")},
    )
    reset_opponent_piece1 = EventTerm(
        func=base_mdp.reset_root_state_uniform,
        mode="reset",
        params={"pose_range": {}, "velocity_range": {}, "asset_cfg": SceneEntityCfg("opponent_piece1")},
    )
    reset_opponent_piece2 = EventTerm(
        func=base_mdp.reset_root_state_uniform,
        mode="reset",
        params={"pose_range": {}, "velocity_range": {}, "asset_cfg": SceneEntityCfg("opponent_piece2")},
    )
    reset_opponent_piece3 = EventTerm(
        func=base_mdp.reset_root_state_uniform,
        mode="reset",
        params={"pose_range": {}, "velocity_range": {}, "asset_cfg": SceneEntityCfg("opponent_piece3")},
    )


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=base_mdp.time_out, time_out=True)


@configclass
class RobodojoPlayTicTacToeEnvCfg(ManagerBasedRLEnvCfg):
    scene: RobodojoPlayTicTacToeSceneCfg = RobodojoPlayTicTacToeSceneCfg(
        num_envs=1, env_spacing=1.5, replicate_physics=True
    )
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    commands = None
    rewards = None
    curriculum = None

    def __post_init__(self):
        # Matches robodojo_pusht_env_cfg.py's cadence exactly.
        self.decimation = 10
        self.episode_length_s = 10.0
        self.sim.dt = 1.0 / 500.0
        self.sim.render_interval = self.decimation
        self.sim.physx.bounce_threshold_velocity = 0.01
        self.sim.physx.friction_correlation_distance = 0.00625
        self.num_rerenders_on_reset = 2
        self.image_obs_list = ["fixed_cam"]
        self.viewer = ViewerCfg(
            eye=(0.0, -0.001, 1.8),
            lookat=(0.0, 0.0, 0.0),
            origin_type="env",
            env_index=0,
        )
        self.teleop_devices = DevicesCfg(
            devices={
                "keyboard": Se2KeyboardCfg(
                    v_x_sensitivity=0.6,
                    v_y_sensitivity=0.6,
                    omega_z_sensitivity=0.0,
                    sim_device=self.sim.device,
                )
            }
        )
