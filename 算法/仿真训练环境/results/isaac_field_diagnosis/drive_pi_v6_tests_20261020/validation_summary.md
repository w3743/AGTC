# PI-v6 CPU / native validation and post-completion isolation

Recorded 2026-10-20. This note summarizes the already completed C validation and a read-only hash audit; tests and simulation were not rerun to produce this record.

## Validation evidence

- Full CPU suite: **244 passed, 13 skipped, 8 warnings**, actual exit 0. The warnings were existing upstream PyTorch/JIT tracing/deprecation warnings; no assertion failed.
- `compileall`, project-wrapper `bash -n`, and runner-wrapper `bash -n`: all exit 0, recorded in `validation_exit.txt`.
- New native PI motion/effort-cap regression: **1 passed**, actual exit 0.
- New native two-environment partial-reset/PI-state regression: **1 passed**, actual exit 0.
- Existing native integration selection: **11 passed, 2 deselected**, actual exit 0.
- Thus the two PI-specific native cases and eleven existing native integration checks all have saved successful exits. Native logs contain upstream Isaac Lab/Newton/Torch warnings; these are retained verbatim.

The saved logs and exit records are linked in this directory: `full_cpu_pytest.log`, `validation_exit.txt`, `native_pi_motion_cap.log` / `_exit.txt`, `native_pi_partial_reset.log` / `_exit.txt`, and `native_existing_integration_11.log` / `_exit.txt`. `current_change_hashes.json` records SHA-256 for the current PI-v6 source, test, and documentation scope and those evidence files.

## Old trial isolation and current trial

The completed `fresh_minimal_inputs_20261020` protocol remains SHA-256 `b5e8e9c3b0561bc73733528aa0c5be3120bb5aad3a54a1fb3418ad84f30a6d65`; its final `stage_04_turn90.zip` remains SHA-256 `2c03cc130977cc005adf56527bc10bc4719a8a9d9489fa82c5a1b41f9d3966e5`. All 40 frozen snapshots still match the old protocol. Of those 40 current paths, only `isaac_field/env.py` and `scripts/train_isaac_foundation_fresh.py` differ from the old freeze; the newly introduced `isaac_field/drive_control.py` was not part of the old 40-file contract. The prior `post_completion_effort_clamp_change.json` remains unchanged.

The independent `fresh_side_pi_20261020` protocol is still marked `prepared` as immutable protocol metadata; its live `run_status.json` is `running`, phase `training_near`, at 16,384 steps in the captured state. It has 41 frozen source files, including `drive_control.py`, and all 41 snapshots match current source at this audit. Runner PID 196317 / exec session 62137; protocol SHA-256 `0a356055c2ff08e13133ebfe12c71e90c63612a8efce7ea0db8b1187aafd10b8`. The official zero-step baseline SHA-256 is `bdb944d4dbba038536687810089448aecb132d07979dd5493c8a33ebbe9e0129`; the latest listed checkpoint is the 16,384-step near checkpoint, SHA-256 `e3bbe31db0c538c670902c9e4f65947de264fba3b88d77341c43dc7e4ef5ba2d`. Training and final evaluation are not complete.

The machine-readable post-completion record is `results/isaac_field_improvement/fresh_minimal_inputs_20261020/post_completion_side_pi_change.json`. It is separate from and does not rewrite the old protocol, snapshots, checkpoint, completion records, or evaluation raw data.

## Limits on interpretation

Ki=0.1 remains an uncalibrated simulation assumption. In the isolated straight-braking counterfactual, PI reached about −0.1300 m/s and accumulated about 0.07293 m of reverse-speed integral; side wheel speeds also showed noticeable fluctuations. The predeclared braking gate covers only the final 0.3 s, so passing its terminal low-speed threshold does not show that the full braking transient is non-degraded.

These CPU/native checks establish software and simulated physical-contract behavior only. They do not establish navigation capability, real hardware calibration, or field safety improvement. The currently running viewer remains the historical pre-PI v4 deployment until a separately validated PI-v6 deployment; it is not a PI-v6 viewer acceptance.

## Viewer deployment correction

The prior viewer sentence preserves the pre-restart v4 context; it was not a post-restart measurement. Root later confirmed that old PID 166936 received SIGTERM and exited. New viewer PID 198426 / exec session 1490 loaded the PI-v6 8,192-step checkpoint; port 8769 returned HTTP 200 while port 8770 was still starting. UI acceptance remains pending and will be recorded separately after visual verification. This deployment update does not change the completed P-only trial, its weights, or its evaluation results.
