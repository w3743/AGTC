# Staged multi-world same-side mimic candidate

This directory is diagnostic staging only. Nothing here is imported by the
production environment or registered as an integration test. The USD, 14
wheel actuators, current drive controller, and effort limit remain untouched.

## CPU builder and solver check

Run from the repository root in the configured Isaac Lab WSL environment:

```bash
bash results/isaac_field_diagnosis/steering_contact_repair_20261020/staged_coupling/run_cpu_two_world_preflight.sh
```

The recorded CPU-only run completed with actual exit code 0. On Newton 1.6.1,
`ModelBuilder.joint_world`, `ModelBuilder.current_world`,
`add_constraint_mimic`, and `add_custom_values` are available. The preflight
builds and finalizes one- and two-world models, then constructs the real
`SolverMuJoCo` model on CPU. It found 12 mimic/equality rows for one world and
24 for two, with all joint pairs and equality world fields confined to their
own `current_world`. For two Newton worlds the solver keeps a single-world
MuJoCo topology (`neq=12`, `nq=14`, `nv=14`) and batches it with
`mjw_data.nworld=2`, `qpos/qvel=(2,14)`. Thus `neq` is per-world topology, not
the 24 total Newton-world custom rows. One- and two-world canonical mechanism
metadata is identical.

## Minimal production hookup if the candidate is selected

The public callback context manager is
`external/IsaacLab/source/isaaclab_newton/isaaclab_newton/cloner/replicate.py`
(`newton_builder_world_hook`). In `isaac_field/env.py`, the narrow construction
window is the `super().__init__(cfg, ...)` call in `IsaacFieldEnv.__init__`
(currently around line 205): Newton scene/model replication occurs there.
Construct the 14 basename list from `ROBOT_SPEC.wheel_specs`, create one
`MultiWorldSameSideMimicHook(expected_world_count=cfg.scene.num_envs)`, and wrap
only that `super().__init__` call in the public context manager. Do not hold the
global hook across `make_env` use, stepping, or resets. After scene construction,
check that the hook visited exactly the requested worlds and optionally run a
single final model audit; then cache the canonical, environment-count-agnostic
mechanism metadata in `drive_config`. Keep the model audit out of the per-frame
path. The hook itself rejects callback/world mismatches and duplicate callbacks.

This is an extension to the live Newton build and still needs a production
patch review before adoption. The callback registry is process-global, so two
simultaneous environment constructions must not share/register the same hook.
The hook's basename lookup is intentionally scoped to `joint_world ==
current_world`; a global basename uniqueness assumption is invalid when cloned
worlds repeat joint names.

## Native regression draft

`native_two_env_test_draft.py` is a proposed opt-in case for
`tests/test_env_integration.py`, after the production hook is added. It creates
two real Newton worlds with distinct left/right commands, verifies 12
relations per world and distinct side-mean wheel rates plus body twist, checks
the original wheel mimic residual gates over the final settled second after a
1 s settle and 2 s differential command. A Warp-only tape records Newton
`joint_qd` in the existing post-actuator/pre-solver callback once per 2 ms
physics step; the final 1 s slice uses the original qd gates, while policy-step
joint position residuals use that same final-second window. It then partially resets world 0. The
reset world starts from all-zero wheel position/velocity, which satisfies the
linear mimic relation. Before/after reset snapshots demand exact equality in
world 1's wheel `q/qd`, root pose/twist, PI state, track command, and actuator
target/applied effort. One subsequent step checks finite state only; settled
residual thresholds are not applied to that transient.

The existing opt-in integration suite is guarded by `RUN_ISAAC_PHYSICS=1` in
`tests/test_env_integration.py`. Relevant regression coverage already includes
native actuator effort limits and two-environment partial PI/feedforward
reset; the proposed test adds actual mimic isolation and reset compatibility.
This draft has not been executed against a GPU scene and is not registered in
the test suite.
