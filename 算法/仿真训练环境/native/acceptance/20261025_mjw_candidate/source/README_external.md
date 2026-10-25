# Newton MuJoCo Warp candidate (external prototype)

This folder contains an isolated candidate for a new `newton_mujoco_warp` backend. It is outside the repository and does not alter the active or frozen v1.1 source. The candidate temporarily reuses `NewtonXPBDGBackend` construction and world maps, then patches the finalized candidate model's chassis and pitch inertias plus inverse inertias, sets shape gap and margin to zero, and replaces the solver with Newton 1.7 `SolverMuJoCo` using MJWarp internal collision detection.

The candidate preserves the current action-target semantics: wheel `joint_target_qd` velocity control and gimbal `joint_target_q` position control. It does not apply the legacy side PI feed-forward torque. Wheel and gimbal effort values come from `QFRC_ACTUATOR` observables.

## CPU-only probe

Run from WSL with the isolated matched MuJoCo 3.15 environment:

```sh
CUDA_VISIBLE_DEVICES= /opt/venv_mujoco315_smoke/bin/python \
  /mnt/c/pioneer/agtc-native-acceptance-backup/audit_stack/mjw_candidate/probe_cpu.py
```

The probe explicitly uses `device="cpu"`, builds one world, checks the requested MuJoCo solver settings, actual exported gap/margin and material/inertia values, performs one deliberately over-commanded physical substep to inspect actual actuator-force observables, and checks that masked `reset(..., flags=0)` preserves a requested pose. It does not run PPO or use a GPU.

The prototype intentionally patches inertia after model finalization and therefore updates both `body_inertia` and `body_inv_inertia`. For eventual repository integration, move the corrected inertia literals into the builder before model finalization and share the constructor rather than retaining this prototype patch.
