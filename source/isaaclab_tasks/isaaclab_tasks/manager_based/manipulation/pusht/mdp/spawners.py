# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from collections.abc import Callable

from pxr import Usd

from isaaclab.sim import schemas
from isaaclab.sim.spawners import materials
from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
from isaaclab.sim.utils import bind_physics_material, bind_visual_material, clone, create_prim, get_current_stage
from isaaclab.utils import configclass


@configclass
class PushTShapeCfg(RigidObjectSpawnerCfg):
    """Spawner config for a T shape made from two rectangular cuboids."""

    func: Callable = None

    bar_size: tuple[float, float, float] = (0.24, 0.08, 0.04)
    stem_size: tuple[float, float, float] = (0.08, 0.24, 0.04)
    bar_offset: tuple[float, float, float] = (0.0, 0.08, 0.0)
    stem_offset: tuple[float, float, float] = (0.0, -0.04, 0.0)

    visual_material_path: str = "material"
    visual_material: materials.VisualMaterialCfg | None = None
    physics_material_path: str = "material"
    physics_material: materials.PhysicsMaterialCfg | None = None


def _spawn_box(
    prim_path: str,
    size: tuple[float, float, float],
    offset: tuple[float, float, float],
    cfg: PushTShapeCfg,
    stage: Usd.Stage,
    visual_material_path: str | None,
    physics_material_path: str | None,
):
    size_min = min(size)
    scale = tuple(dim / size_min for dim in size)
    create_prim(
        prim_path,
        "Cube",
        translation=offset,
        scale=scale,
        attributes={"size": size_min},
        stage=stage,
    )

    if cfg.collision_props is not None:
        schemas.define_collision_properties(prim_path, cfg.collision_props, stage=stage)
    if visual_material_path is not None:
        bind_visual_material(prim_path, visual_material_path, stage=stage)
    if physics_material_path is not None:
        bind_physics_material(prim_path, physics_material_path, stage=stage)


@clone
def spawn_t_shape(
    prim_path: str,
    cfg: PushTShapeCfg,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    **kwargs,
) -> Usd.Prim:
    """Spawn a compound T shape with collision geometry on child cuboids."""

    stage = get_current_stage()
    if stage.GetPrimAtPath(prim_path).IsValid():
        raise ValueError(f"A prim already exists at path: '{prim_path}'.")

    create_prim(prim_path, prim_type="Xform", translation=translation, orientation=orientation, stage=stage)

    visual_material_path = None
    if cfg.visual_material is not None:
        visual_material_path = (
            f"{prim_path}/{cfg.visual_material_path}"
            if not cfg.visual_material_path.startswith("/")
            else cfg.visual_material_path
        )
        cfg.visual_material.func(visual_material_path, cfg.visual_material)

    physics_material_path = None
    if cfg.physics_material is not None:
        physics_material_path = (
            f"{prim_path}/{cfg.physics_material_path}"
            if not cfg.physics_material_path.startswith("/")
            else cfg.physics_material_path
        )
        cfg.physics_material.func(physics_material_path, cfg.physics_material)

    _spawn_box(
        f"{prim_path}/bar",
        cfg.bar_size,
        cfg.bar_offset,
        cfg,
        stage,
        visual_material_path,
        physics_material_path,
    )
    _spawn_box(
        f"{prim_path}/stem",
        cfg.stem_size,
        cfg.stem_offset,
        cfg,
        stage,
        visual_material_path,
        physics_material_path,
    )

    if cfg.mass_props is not None:
        schemas.define_mass_properties(prim_path, cfg.mass_props, stage=stage)
    if cfg.rigid_props is not None:
        schemas.define_rigid_body_properties(prim_path, cfg.rigid_props, stage=stage)

    return stage.GetPrimAtPath(prim_path)


PushTShapeCfg.func = spawn_t_shape
