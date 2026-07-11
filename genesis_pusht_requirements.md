# Bimanual Push-T in Genesis — Requirements Document

## Scene Setup

**Physics world**
- Flat table surface, ~1 m × 1 m working area
- Two 6-DOF robot arms (e.g. ViperX 300s / SO-ARM100 equivalent) mounted on opposite ends of the table, facing each other
- T-shaped rigid block (~0.24 m bar × 0.08 m stem, ~40 mm tall) resting on the table surface
- Static "goal" T-shape marker at a fixed pose, visually distinct (semi-transparent green)

**Camera**
- Single fixed top-down camera, orthographically framed to the table surface
- Resolution 128 × 128, RGB output
- Additionally outputs a per-primitive instance segmentation pass (raw integer prim IDs, not colorized) — needed for arm contour extraction

**Coordinate convention**
- Table surface = Z = 0; arms extend from ±X ends; screen-up = +Y, screen-right = +X
- Both arms use planar (XY) end-effector velocity control with optional Z

---

## Teleoperation Controls

| Key(s) | Action |
|--------|--------|
| W / S / A / D | Move **left** arm EE: +Y / −Y / −X / +X |
| I / K / J / L | Move **right** arm EE: +Y / −Y / −X / +X |
| Q / E | Raise / lower **both** arms (+Z / −Z) |
| [ / ] | Decrease / increase EE speed (both arms) |
| R | Reset both arms to home pose |
| P | Start recording current take |
| O | Stop & save current take |

**Speed scaling**: multiplicative, range ~0.1× – 4×, applied to a base velocity. Both arms scale together.

---

## Recording

**Format**: HDF5, one group per demonstration (`data/demo_0`, `data/demo_1`, …)

**Per-step data stored** (all arrays shape `(T, ...)`):

| HDF5 path | Shape | Description |
|-----------|-------|-------------|
| `obs/fixed_cam` | `(T, 128, 128, 3)` uint8 | RGB top-down camera |
| `obs/edge_cam` | `(T, 128, 128, 3)` uint8 | Synthetic edge image |
| `obs/state` | `(T, ~20)` float32 | Low-dim state vector |
| `obs/actions` | `(T, 6)` float32 | Applied EE velocity |
| `actions` | `(T, 6)` float32 | Raw action |
| `states/...` | nested | Full articulation & rigid-body state |

**Recording flow**: operator presses P → system snapshots the current state as episode start → every step is appended → O saves and closes the episode. R discards the current buffer.

---

## Synthetic Edge Image (`edge_cam`)

Produced every step from known scene state + camera segmentation. Layers drawn back-to-front into a black 128 × 128 RGB canvas:

| Layer | Color | Source | Notes |
|-------|-------|--------|-------|
| Table boundary | Gray (120,120,120) | Project 4 table corners via camera intrinsics | Clamp to frame border so edge always visible |
| Goal T outline | Dim green | Fixed goal pose → rotate T polygon → project | Static, computed once |
| T-block outline | White, thickness 2 | Current block pose → rotate T polygon → project | 8-vertex T polygon |
| Left arm contour | Cyan (0,220,255) | Camera instance segmentation mask for left arm | `cv2.findContours` on prim-ID mask |
| Right arm contour | Orange (255,140,0) | Camera instance segmentation mask for right arm | Same method |

**T polygon** (local frame, 8 vertices):
```
(-0.12, 0.12) → (0.12, 0.12) → (0.12, 0.04) → (0.04, 0.04)
→ (0.04,-0.16) → (-0.04,-0.16) → (-0.04, 0.04) → (-0.12, 0.04)
```

**Camera projection**: standard pinhole — transform world points to camera frame, then `u = fx·X/Z + cx`, `v = fy·Y/Z + cy`.

**Arm contour method**: the camera's instance-segmentation pass assigns each pixel an integer prim ID. Filter IDs whose associated prim path contains "RobotLeft" / "RobotRight", build a binary mask, run contour detection, draw outlines.

---

## Viewport Overlay (Live Debug Draw)

Drawn in 3D world space every step using the simulator's debug-draw API (line segments):

| Layer | Color | Thickness |
|-------|-------|-----------|
| T-block outline | White | 3 px |
| Goal T outline | Green | 2 px |
| Table boundary | Light gray | 2 px |

Arms are **not** in the overlay — the 3D arm meshes are already visible in the viewport. The overlay lifts lines ~3 cm above the table surface for visibility.

---

## UI Panel (Recording Mode)

Small status window showing:
- Dataset file path
- Recording state: `● RECORDING` / `○ Not recording`
- Number of takes saved
- Key hint: `P — start | O — save | R — reset`

---

## Video Extraction Script

Offline tool that reads an HDF5 file and writes one MP4 per demo:
- `--obs_key obs/fixed_cam` → RGB video (default)
- `--obs_key obs/edge_cam` → synthetic edge video
- `--fps` (default 30), `--out_dir`, `--demo` (single demo or all)
