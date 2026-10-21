# Native motion-response probe draft

`staged_motion_response_probe.py` records an eight-case response matrix for the current Newton drive path. It uses the existing `_step` helper in `tests/test_env_integration.py` and `record_wheel_applied_effort` in `tests/newton_effort_capture.py`. The original source used for the 5 ms baseline is preserved under `sources/` with the SHA recorded in that run's summary.

Each case resets the same one-environment flat Field configuration with seed `90231`, start `[50, 50, 0]`, and goal `[90, 50]`. It then runs 1 s of zero-action settle, 2 s of commanded motion, and 5 s of zero-action braking. The last 0.3 s of braking is explicitly zero command. The installed PI controller remains active as configured by the env; there is no test-side PI overlay or manual integral/feedforward edit. The eight cases are:

| Case | Left, right normalized track action | Response sign |
|---|---:|---|
| Forward | `[0.12, 0.12]` | `v > 0` |
| Reverse | `[-0.12, -0.12]` | `v < 0` |
| Low pivot left | `[-0.10, 0.10]` | `yaw > 0` |
| Low pivot right | `[0.10, -0.10]` | `yaw < 0` |
| Left arc | `[0.12, 0.24]` | `v > 0`, `yaw > 0` |
| Right arc | `[0.24, 0.12]` | `v > 0`, `yaw < 0` |
| High pivot right | `[0.40, -0.40]` | `yaw < 0` |
| High pivot left | `[-0.40, 0.40]` | `yaw > 0` |

The output contains one policy-step trajectory record with chassis pose and velocity, roll/pitch/root height, wheel joint speeds and targets, track commands, PI state, damage/collision counters, and native-effort slot bounds. A second JSONL records all 14 actual post-actuator wheel efforts at every Newton physics substep. The summary includes reset-state hashes for each case, config fields and live Newton shape-material arrays indexed alongside runtime shape labels (including `shape_material_kf` when exposed), solver options, mean signed motion, pivot drift, braking-tail `v/w`, and opposite-motion rebound integrals.

The reported reference checks reuse the existing integration gates: forward/reverse displacement of 0.7–1.3 m in the commanded direction, last-second signed speed above 0.05 m/s, high-pivot yaw change of at least 0.1 rad, pivot center drift below 0.75 m, final braking `|v| < 0.02 m/s`, and the task's existing `|w| < 0.3 rad/s` hold bound. The low-pivot 0.1-rad yaw test is retained as an informational result only, since that case uses the older weak response command `±0.10`. Both final braking samples and the maximum absolute `v/w` over the last 0.3 s are recorded and checked against the same limits. The probe also checks finite state/effort samples, the 2 Nm native effort cap, zero task damage/collision counters, no task termination, and the environment's 1.2 rad roll/pitch overturn limit. Arc cases record signed yaw and translation response without imposing a new calibrated threshold. With plants and rocks absent, zero damage/collision counters are instrumentation checks, not an obstacle-contact validation.

These checks are response screens, not a physics-repair verdict. A candidate repair must be compared against the old physics using matching pre-settle reset-state hashes for every action; do not treat a weak baseline's small correctly signed yaw as proof of repair. The probe fixes policy dt at 0.05 s and accepts physics dt 0.005/0.002/0.001 s (decimation 10/25/50); cone and impratio are explicit command-line values. All runs use the existing flat mesh and Newton contacts (`use_mujoco_contacts=False`).

The pinned current-physics baseline runner is `run_baseline_5ms_pyramidal_1_kf1000.sh`. It fixes `physics_dt=0.005`, policy `dt=0.05`, decimation `10`, pyramidal cone, impratio `1`, and production materials. The run completed with real exit 0, matching pre-settle reset hashes across all eight cases, 12,800/12,800 captured Newton physics substeps, and max wheel effort 1.8781 Nm. Runtime `shape_material_kf` was 1000. Its raw data, summary, log, and `.exit.json` are in `runs/baseline_5ms_pyramidal_1_kf1000/` and the `runs/` directory.

The baseline has already run. The pinned runner refuses to reuse any prior output, log, or exit record.

For the friction-coefficient isolation, `--shape-mu` optionally sets `cfg.sim.physics.default_shape_cfg.mu`; omitting it retains the config default. The summary records whether `mu` was explicit and validates the live Newton `shape_material_mu` for exactly one `/Field/Ground` shape and all 14 wheel collision shapes before collecting motion. `kf`, solver cone, impratio, timestep, reset, and eight actions remain fixed. The prepared runners are `run_mu_0_4_5ms_pyramidal_1_kf1000.sh` and `run_mu_1_2_5ms_pyramidal_1_kf1000.sh`; each has a distinct fresh output path and requires parent authorization plus `RUN_STAGED_RESPONSE_MU=1`.

Run these only after the parent confirms the GPU is idle and authorizes the treatment:

```bash
wsl.exe -d Ubuntu-24.04 -- bash -lc 'cd "/mnt/e/Pioneer/先进通用履带底盘工程/算法/仿真训练环境" && RUN_STAGED_RESPONSE_MU=1 bash results/isaac_field_diagnosis/steering_contact_repair_20261020/staged_response/run_mu_0_4_5ms_pyramidal_1_kf1000.sh'
wsl.exe -d Ubuntu-24.04 -- bash -lc 'cd "/mnt/e/Pioneer/先进通用履带底盘工程/算法/仿真训练环境" && RUN_STAGED_RESPONSE_MU=1 bash results/isaac_field_diagnosis/steering_contact_repair_20261020/staged_response/run_mu_1_2_5ms_pyramidal_1_kf1000.sh'
```

The probe refuses output directories outside `runs/`, refuses non-empty output directories, and verifies the frozen training and diagnostic source hashes before and after measurement. A change in response at `mu=0.4` or `mu=1.2` is evidence about this simulation treatment only; it does not calibrate real-world friction or establish a repair by itself.
