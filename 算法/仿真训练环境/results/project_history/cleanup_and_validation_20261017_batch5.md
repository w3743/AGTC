# Fifth-batch cleanup and validation — 2026-10-17

This cleanup covers task reward/trajectory diagnostics, the explicit safe-goto profile, pure-turn and crop-ray regression evidence, and current Isaac Lab documentation. It did not modify runtime, training, or physics source files.

## Cleanup and documentation

- Moved `scripts/diagnose_plant_pose_consistency.py` and `scripts/diagnose_plant_sensor_hits.py` into `results/project_history/plant_diagnostics_20261017/`. A repository search found no production references outside the scripts. Existing JSON evidence under `results/isaac_field_diagnosis/` was left in place; no result JSON or training checkpoint was moved or overwritten.
- Updated `ISAAC_LAB.md` and `使用说明.md` with the opt-in `safe-goto` profile (`max_time=20 s`, `damage_penalty=20`, `goal_progress_budget=5`, `distance_cost=0`), while documenting that legacy `staged-goto` remains the default profile with its previous settings.
- Recorded the safe-goto before held-out baseline: 3/24 arrivals, 6/24 damaged episodes, mean damage 0.416667, and zero collisions. The 65,536-step continuation in `results/isaac_field_improvement/safe_goto_20261017_reward_v1/` and its after evaluation were still in progress at report time. Recent training-rollout counts are not held-out results. No capability-improvement claim is made.
- Recorded that the latest completed checkpoint remains the 53,248-step `field_policy_after.zip`. On development seeds 76001–76004, all four 20-second episodes timed out without entering the 0.4 m goal tolerance; hold stayed at zero. The observed weak yaw response and the analysis limits are documented alongside the raw trajectory files.
- Linked the existing native regressions: flat-ground left/right pivot test `test_native_pure_differential_turn_has_measurable_yaw_without_forward_command` and nearby crop-stem ray test `test_live_crop_stem_returns_native_range_without_a_leaf_collider`. These are individual physical/sensor checks, not policy-capability results.
- No old backend files, shared WSL runtime, upstream checkout, training output, or viewer process was touched.

## Validation

- Full CPU suite in the WSL Isaac Python runtime, with `RUN_ISAAC_PHYSICS` explicitly unset: **56 passed, 7 skipped in 4.14 s**. Native physics integration tests were skipped by default.
- `python -m compileall -q isaac_field scripts tests`: passed.
- `bash -n scripts/install_isaaclab.sh scripts/isaaclab_python.sh`: passed.
- The command output is preserved in `cleanup_and_validation_20261017_batch5.log`.
- No GPU test, simulator launch, training run, or throughput measurement was performed for this cleanup. The already-running safe-goto experiment was left untouched.
