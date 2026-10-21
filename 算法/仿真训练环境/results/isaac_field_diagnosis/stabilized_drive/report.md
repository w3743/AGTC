# Stabilized-drive Isaac Lab integration check

Date: 2026-10-17 (Asia/Shanghai)

Command executed in the Ubuntu-24.04 WSL Isaac Lab runtime:

```sh
RUN_ISAAC_PHYSICS=1 bash scripts/isaaclab_python.sh -m pytest -q -s tests/test_env_integration.py
```

The command was run from the project root with output captured in `test_env_integration_20261017_01.log`. Pytest completed successfully: **5 passed, 44 warnings in 35.82s**. This includes `test_native_low_speed_drive_and_braking_have_bounded_chassis_motion`; all its chassis-speed, displacement, braking-drift, and settled-speed assertions passed at their existing thresholds.

Static source values checked for this run: `drive_velocity_gain = 0.5`; `track_ground_contact_length_m = 0.595`. No production environment or asset files were changed. No other suites, benchmarks, or training were run.

Warnings were deprecation/future warnings from TorchScript, Newton actuator naming, Newton shape-color replacement, and the SolverMuJoCo standalone-root fallback for plant anchors; none caused a test failure.
