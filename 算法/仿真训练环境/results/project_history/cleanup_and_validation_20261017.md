# Isaac Lab cleanup and validation — 2026-10-17

This cleanup archived only the two completed high-gain drive diagnostic scripts:
- scripts/diagnose_isaac_drive_v2.py → results/project_history/drive_tuning_20261017/diagnose_isaac_drive_v2.py
- scripts/diagnose_isaac_drive_v2_root_state.py → results/project_history/drive_tuning_20261017/diagnose_isaac_drive_v2_root_state.py

The associated diagnostic JSON files, training/evaluation results, and checkpoints were left unchanged. The production motion/braking regression remains tests/test_env_integration.py and requires RUN_ISAAC_PHYSICS=1; the optional A/B/C/D contact probe remains scripts/diagnose_isaac_ground_contact.py. Documentation now identifies the archived probes as historical and points to the current regression entry.

The staged-goto comparison used 32,768 additional training steps (training seeds 46001 and 46002), with fixed held-out seeds 76001–76012 on 16-plant flat scenarios. Goal arrivals changed from 0/12 to 1/12; mean final distance changed from 1.89916 m to 1.42249 m; mean damage stayed 0.5. The sole arrival (seed 76008) incurred four damage events and no collisions, so zero-damage, collision-free arrivals remained 0/12 in both evaluations. This is a small increase in goal arrivals, not evidence of reliable navigation, learned obstacle avoidance, or full task capability. The final checkpoint was CPU-verified at 53,248 total steps; original evaluation records and model files remain preserved.

Validation performed with the installed Isaac Lab WSL runtime:
- Pure current test suite: 42 passed (task, assets, layouts, policy, learning, viewer, adversary, adversarial training).
- python -m compileall -q scripts isaac_field: passed.
- bash -n scripts/install_isaaclab.sh scripts/isaaclab_python.sh: passed.
- Native physics tests were not rerun during this cleanup; the five native regression tests had already passed. No training or GPU test was started.
