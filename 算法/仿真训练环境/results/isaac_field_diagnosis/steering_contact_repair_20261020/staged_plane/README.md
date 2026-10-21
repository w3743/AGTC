# Staged flat-ground plane hook (draft, not executed)

This folder contains an isolated diagnostic hook only. It has not been imported,
executed, tested, or used to create an environment. It does not change the
production scene, training contract, old checkpoint, or prior results.

## Why this is the smallest available seam

The installed Isaac Lab Newton integration exposes
`isaaclab_newton.cloner.newton_builder_world_hook`. Its callback receives the
per-world `ModelBuilder` after the environment's USD sources have been copied
into that world and before finalization. Isaac Lab's own Franka/Newton MPM task
uses this hook to add a native plane with `builder.add_shape_plane`; this avoids
editing Isaac Lab/Newton vendor code or creating another simulation path.

For this project the imported ground shape is labelled
`/World/envs/env_0/Field/Ground`. `terrain_asset()` authors it as a triangulated
USD mesh at z=0 even when `terrain_enabled=False`. The hook retains that mesh and
its `VISIBLE` flag, clears only its `COLLIDE_SHAPES` flag, then adds one static
Newton plane at the same world transform. It copies the ground's `mu`, `kf`,
`ke`, `kd`, margin, gap, and its effective `mujoco:solref_mode` to the plane.
The v4 runtime dump shows that the ground and all 14 wheel shapes use mode `2`
(`SOLREF_MODE_MJCF_DEFAULT`). The public builder attribute stores sparse
overrides only, so if the ground has no explicit entry the hook pins the plane
to this v4-observed mode `2`. The v6 final-model audit fails before reset or
motion unless the retained ground, plane, and wheel shapes all actually report
mode `2`; this does not assume that the plane's builder default is equivalent.
Solver configuration and the robot builder remain owned by the v4-derived
runner.

## Ray and render semantics

Newton's `NewtonRaycastSensor` requests a BVH containing `COLLIDE_SHAPES`. The
mesh remains a visible Newton shape for rendering but is no longer in that
raycast/contact set; lidar/ray reads therefore hit the replacement plane. In
the flat test region, both surfaces are z=0 with +Z normal, so hit distance and
normal should agree there. The plane is infinite while the original USD mesh is
finite (100 m square), so rays outside the mesh boundary can now hit the plane.
This comparison is only for a plant-free, flat, in-field vehicle-motion
diagnostic. It does not preserve boundary, rough-terrain, plant, or full sensor
contract behavior and is not a production candidate by itself. The USD ground
mesh remains the rendered ground surface.

## v6 result-only integration

The hook is integrated only around the copied probe's single
`IsaacFieldEnv(cfg)` constructor in
`../contact_ground_plane_v6/contact_ground_plane_v6.py`. The copy preserves the
v4 seed, initial pose, command, PI drive, contact sensor, 2 ms / elliptic /
impratio-1 solver settings, and policy/native sampling. It pins
`plants=0`, `terrain_enabled=False`, and `num_envs=1`; the helper fails closed
otherwise. The full and fixed comparison contracts include
`geometry_treatment="flat_mesh_to_native_plane"`, so the old source-mesh USD
hash cannot make this run appear geometrically identical to v4.

```python
from isaaclab_newton.cloner import newton_builder_world_hook
from isaaclab_newton.physics import NewtonManager

# staged_dir is this directory, resolved from the runner's repository ROOT.
sys.path.insert(0, str(staged_dir))
from staged_plane_hook import (
    FlatGroundPlaneHook,
    assert_flat_single_env_cfg,
    validate_final_model,
)

assert_flat_single_env_cfg(cfg)
plane_hook = FlatGroundPlaneHook()
with newton_builder_world_hook(plane_hook):
    env = IsaacFieldEnv(cfg)

if len(plane_hook.worlds) != 1:
    raise RuntimeError(f"plane hook ran for unexpected worlds: {plane_hook.worlds}")
model_audit = validate_final_model(
    NewtonManager.get_model(),
    wheel_joint_names,
    wheel_body_names,
)
```

The v6 runner records `plane_hook.worlds` and `model_audit` alongside its
existing runtime shape-material and solver-option dump. Its final-model checks
require exactly one retained `/Field/Ground` mesh and one
`/Field/GroundPlaneDiagnostic`, confirm mesh-visible/mesh-noncolliding and
plane-colliding/plane-not-rendered flags, and require one unique basename match
for each of the original 14 wheel joints and wheel-link bodies. The wheel links
are nested below `Robot/base_link/`, so matching is path-depth independent. Keep
the full model labels or at least these selected labels, flags, and copied
material values in the raw result so the geometry change is auditable. The
runner also compares all six finalized material fields between retained ground
mesh and plane, checks wheel/ground/plane `mu=.8`, `kf=1000`, and fails unless
ground, plane, and wheel shapes all retain the v4-observed solref mode 2.

## Source seam evidence

- `external/IsaacLab/source/isaaclab_newton/isaaclab_newton/cloner/replicate.py:71` defines the public context manager and callback contract.
- `external/IsaacLab/source/isaaclab_newton/isaaclab_newton/cloner/newton_clone_utils.py:380` calls world hooks after adding the USD world builder.
- `external/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/contrib/franka_pour/pour_env.py:115` and `:247` show the supported context and native-plane pattern.
- `external/IsaacLab/source/isaaclab_newton/isaaclab_newton/sensors/ray_caster/newton_raycast_sensor.py:289` adds `COLLIDE_SHAPES` to the ray BVH flags.
- `isaac_field/scene_assets.py:25` and `:44` author the finite `/Terrain/Ground` mesh.
- `isaac_field/env.py:137` makes flat terrain when disabled; `:154` binds the USD to `{ENV_REGEX_NS}/Field`.
- Existing v3 probe records the imported ground label as `/World/envs/env_0/Field/Ground` and already checks the 14 wheel joints/bodies.

## Status

The v6 directory is a prepared, unexecuted diagnostic. Python syntax and CPU
fake-model assertions are the only planned checks before the parent decides
whether to run it; no environment was constructed and no GPU was used. Treat any
future result as a controlled flat-geometry comparison with the same Newton
solver and drive, not as a full-contract match or production recommendation.
