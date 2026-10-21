# Physics/contact child for a future `drive_config` (read-only proposal)

This directory is an isolated insertion fragment, not production code. It has
not been imported or run, and it does not edit the frozen 41-source trial or any
existing result/checkpoint. The proposed metadata is descriptive; it does not
alter the observation/action contract or policy weights.

## Compatibility finding

Adding one JSON-safe nested child such as `drive_config["physics_contact"]` is
compatible with the current metadata path:

- `scripts/train_isaac_foundation_fresh.py:107-117` accepts any drive keys as
  long as the existing `interface` remains `independent_left_right_track_speed_explicit_pi_v6`;
  it shallow-copies the mapping, so the nested child is retained. The same
  object is written in initialization metadata (`:498`) and the fresh runtime
  contract is JSON-written by `_json()` (`:67-69`).
- `isaac_field/training.py:175-176` and `isaac_field/learning.py:615-616` copy
  the full drive mapping into `model.field_drive_config`; SB3 checkpoints keep
  the nested Python mapping. `learning.py:445-446` includes both current and
  checkpoint drive mappings in held-out evaluation results.
- `isaac_field/viewer.py:163-170` recursively normalizes mappings/lists, and
  `:291-299` compares the full current and checkpoint drive mappings. A
  same-shape checkpoint with old recorded drive metadata will load if its
  action/encoder contracts still pass, then display a drive mismatch. A
  checkpoint with no drive metadata displays `legacy/unrecorded`, not
  `mismatch`.
- `scripts/improve_isaac_policy.py:339-343` compares the current drive contract
  in before/after evaluation reports. Equal physics metadata remains
  comparable; different physics metadata becomes
  `incomparable_environment_contract`. It intentionally does not require an
  old checkpoint's descriptive metadata to match before re-evaluation.
- `scripts/train_isaac_foundation.py:590-591` checks the final and baseline
  checkpoint drive metadata against the runtime mapping using normal recursive
  Python dict equality. A new initialized baseline and its final model will
  match if both were created after adding the child.

All nested values must be ordinary JSON-safe strings, booleans, `int`, `float`,
`None`, lists, and dictionaries. Avoid enum objects, classes, Warp arrays, and
NumPy scalar/array values in the serialized mapping.

## Field names and the `kf` caveat

Current `FieldEnvCfg` uses `dt=.005`, `decimation=10`, Newton physics,
`MJWarpSolverCfg(njmax=256, nconmax=128, iterations=40, ls_iterations=30,
integrator='implicitfast', use_mujoco_contacts=False)`, Newton's
`num_substeps=1`/`collision_decimation=0`, and
`NewtonShapeCfg(margin=.002, gap=.005, mu=.8)` (`isaac_field/env.py:71-79`).
`MJWarpSolverCfg` also exposes `solver_type`, `solver`, `cone`, `impratio`,
`disable_contacts`, and `tolerance` (see `external/IsaacLab/source/isaaclab_newton/isaaclab_newton/physics/mjwarp_manager_cfg.py:22-145`).
`NewtonCfg` separately provides `num_substeps` and `collision_decimation`
(`external/IsaacLab/source/isaaclab_newton/isaaclab_newton/physics/newton_manager_cfg.py:171-180`).
The effective current policy period is `float(env.step_dt)`, nominally
`physics_dt * decimation` = 0.05 s.

`NewtonShapeCfg` exposes `mu`, `ke`, `kd`, `margin`, and `gap`, but the installed
base class has no `kf` field (`external/IsaacLab/source/isaaclab_newton/isaaclab_newton/physics/newton_manager_cfg.py:106-143`). The fragment records `kf` only when the actual cfg exposes it; otherwise cfg `kf` stays null and the source limitation is explicit. A local `NewtonShapeCfgWithKF` subclass can provide the configured value without conflating it with finalized runtime materials. Record cfg defaults separately from finalized `model.shape_material_*` values for the selected wheel and ground shapes. Existing v3 runtime evidence reports wheel/ground
`mu=.8, kf=1000, ke=2500, kd=100, margin=.002, gap=.005` (for example,
`contact_friction_v3_summary.json:1453,1514,1530-1548,1566-1613`). The fragment
stores them under canonical `ground` and `wheel` roles, with a sorted list of
unique finite values for each material field. It deliberately omits full
per-environment shape paths and shape counts, so 1-env viewer/evaluation and
4-env training metadata compare equal when materials match; genuine per-shape
material variation remains visible as multiple values. Missing roles and
non-finite actual values fail closed.

With `use_mujoco_contacts=False`, Newton's CollisionPipeline generates the
contacts and MJWarp resolves the constraints; `True` selects MuJoCo Warp's
internal contact path. Store both the boolean and an explicit descriptive
`contact_generation` value. `physics_manager` is the runtime Newton manager
class (currently `NewtonMJWarpManager`), while `solver_type` is the configured
`mujoco_warp` backend and `solver` is the configured `newton` algorithm.

## Minimal future insertion

Use `drive_physics_contact_fragment.py` as a helper in a future unfrozen source
revision. The only production additions should be: import the helper, compute
and cache its result once after `super().__init__` has finalized the model, and
add `physics_contact=copy.deepcopy(self._physics_contact_metadata)` to the
existing `drive_config` return. Caching matters because reading Newton model
material arrays can synchronize GPU state; the viewer may refresh metadata
repeatedly. `copy.deepcopy` keeps callers from mutating cached nested values.

## Most relevant checks if this proceeds

- Extend `tests/test_env_integration.py::test_native_gimbal_imu_and_drive_feedback_use_physical_signals` to assert the actual timestep/contact fields and JSON serialization. This test is opt-in and requires `RUN_ISAAC_PHYSICS=1`; the existing top-level drive assertions remain unaffected by an additive child.
- Extend CPU-only `tests/test_isaac_field_viewer.py::test_metrics_show_current_and_checkpoint_drive_observation_metadata` with one nested physics value changed in the checkpoint mapping and assert the existing drive status becomes “不一致”. Its recursive comparison already supports the nested mapping.
- Existing `tests/test_fresh_foundation.py::test_live_environment_contract_requires_pi_v6_drive_interface_and_v6_observation` validates the interface/schema only, so an additive nested mapping should pass without weakening those checks. Existing evaluation report equality assertions should also preserve the child unchanged.

Do not add this metadata to the currently frozen PI trial. If a physical
candidate is selected, introduce it before the next trial's prepare/preflight
so the candidate snapshot, initialization record, checkpoints, and evaluation
reports all describe the same runtime configuration. It does not create a new
physics protocol or require any NN contract change.
