# Front-guide same-side speed coupling

This record summarizes the guide-wheel backend integration and the retained CPU/native regression evidence for 2026-10-20. It covers implementation and physics-contract checks only; no new model training or learning result is claimed.

## Implementation path

1. `isaac_field/assets.py` defines the frozen `FrontGuideSpec` and its two `FrontGuideWheelSpec` instances. The current trial defaults are x=0.42 m, guide axis 0.05 m above the support-wheel axes, radius 0.045 m, width 0.06 m, and 0.25 kg per guide. These are proxy assumptions. The asset digest includes the complete guide spec, including `rotation_coupling`, and guide assets use a dedicated `front_guides_<sha256>/field_robot.usda` directory. The no-guide digest and asset path retain their legacy form. The two guide masses are taken from the chassis allocation so the native model remains 30 kg. Guide joints have no independent `DriveAPI`.
2. `make_env(..., front_guide_spec=...)` serializes the dataclass as a plain dict in `FieldEnvCfg`; `IsaacFieldEnv` restores the typed spec before asset creation and runtime setup. This keeps Isaac's config wrapper away from recursively copying the frozen dataclass.
3. In `isaac_field/env.py`, the runtime creates a `SameSideMimicBuilderHook`. `isaac_field/newton_drive.py` resolves each joint inside its actual Newton world and appends 12 existing support-wheel relations plus two guide relations when coupling is enabled. Each guide follows its same-side wheel-0 leader with `q_guide = (R_main/R_guide) * q_wheel0`; the angular-speed ratio is the same. At the current equal 45 mm radii, the ratio is 1.0. The hook checks the requested per-world row counts and the final Newton/MuJoCo relation mapping.
4. Coupling adds equality rows, not actuators. The original 14 driven support-wheel joints, per-wheel 2 Nm effort cap, PI state, gimbal controls, and 37-dimensional feedback remain. Native checks found zero actuator generalized force at guide DOFs while the driven wheels were commanded; equality constraint force may still transfer load to a guide.
5. `drive_config` reports 14 same-side relations per world as 12 support-wheel plus two guide relations. `front_guide` records the coupled model name, geometry, mode, same-side wheel-0 leaders, radius ratios, zero guide motors, zero added PID, and zero added drive torque. Contact metadata records the guide bodies and their actual material values.
6. The fresh runner freezes the guide spec, derived wheel records, guide coupling contract, per-world relation counts, robot asset hash, and a canonical profile-contract SHA in `profile_physics_contract`. It passes that same spec to each environment and validates the live drive contract against the frozen profile before training/evaluation. This documents the runner interface; this integration task did not launch a training trial.

## Supported modes and limits

- `rotation_coupling="same_side_track_speed"` is the default. It creates 14 same-side relations per world: 12 support-wheel follower relations and two guide-to-wheel-0 relations.
- `rotation_coupling="free"` is retained as a diagnostic setting. Guide wheels have no independent motors and no guide equalities; the existing support-wheel relation count stays 12 per world.
- `front_guide_spec=None` preserves the prior no-guide asset hash/path and the prior same-side metadata shape. With the default support coupling enabled, it remains 12 same-side relations per world and the original 14 driven support wheels.
- The rollers are separate proxy wheels. This is not a continuous track, a detailed track-contact model, a real-vehicle calibration, or evidence of improved learned navigation.

## Verification

The final coupled native run is [attempt8](attempt8_native_full_pass/). Its recorded command was `RUN_ISAAC_PHYSICS=1 ./scripts/isaaclab_python.sh -m pytest -q -s tests/test_front_guide_native.py`; the pytest and actual process exit codes are both 0. It reports 3 passed and 39 upstream warnings. The run covers coupled and free guide modes plus the live adapter terrain-seed check. In two worlds it sampled 20 settling steps followed by 40 policy steps: one straight command and one opposing-track pivot. The original residual limits were retained (position max 0.02 rad, velocity p95 0.0926 rad/s, velocity max 0.463 rad/s).

Measured guide-to-same-side-wheel-0 residuals over the 40 commanded policy steps were:

| Side | q absolute max (rad) | qd absolute p95 (rad/s) | qd absolute max (rad/s) | Radius ratio |
|---|---:|---:|---:|---:|
| Left | 0.0000495911 | 0.0128595 | 0.0161705 | 1.0 |
| Right | 0.0000419617 | 0.0103928 | 0.0132980 | 1.0 |

The same run checked 30 kg total mass, the 14 existing driven support-wheel joints, no guide actuator or guide actuator torque, 12+2 equality rows per world with no cross-world pairs, the unchanged 2 Nm wheel cap and 37-element feedback, and guide contact material metadata (`mu=0.8`, `kf=1000`). The complete source is in `tests/test_front_guide_native.py`; raw output and exit record remain in attempt8.

The legacy regression is [attempt9](attempt9_native_legacy14/). Its recorded command was `RUN_ISAAC_PHYSICS=1 ./scripts/isaaclab_python.sh -m pytest -q -s tests/test_env_integration.py`; the pytest and actual process exit codes are both 0. It reports 14 passed, 55 upstream warnings, in 86.65 s.

Targeted CPU metadata tests and compile checks also passed in [attempt1](attempt1_cpu/) and [attempt5](attempt5_cpu_finalized_contract/) (12 passed, compileall exit 0); the additional static compile check in [attempt2](attempt2_static/) exited 0.

## Retained failed attempts

Attempts 3, 4, 6, and 7 have actual exit code 1 and remain untouched alongside their logs. These were test-audit/API mistakes, not a failed training or an observed guide-motion threshold regression:

- [attempt3](attempt3_native_guides_motion/) and [attempt4](attempt4_native_mapping_diagnosis/) used an extra MuJoCo `eq_objtype` filter that excluded valid Newton-generated joint equalities (`eq_type` identified joint equality while `eq_objtype` retained its default value). Attempt3 also used direct unprefixed host joint-name lookup in the free-guide test. The corrected test resolves joint IDs through the solver's per-world mapping and filters on the joint equality type; attempt4 preserves the raw host mapping diagnosis.
- [attempt6](attempt6_native_coupling_final/) queried a nonexistent `MjModel.eq_polycoef`; the supported joint equality coefficients are read from `MjModel.eq_data[row, :5]`.
- [attempt7](attempt7_native_motion_residuals/) expected the non-reset world to retain only one stability sample after 60 updates. The corrected partial-reset assertion checks that the reset world returns to zero and the untouched world retains 60 samples. The guide q/qd residual checks had already passed in that attempt.

Every attempt directory retains its own command wrapper, raw pytest log, and actual exit-code record. `source_before/sha256.json` records the pre-integration source backups. No failed log was overwritten.
