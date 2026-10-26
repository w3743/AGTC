# MJW repo integration validation evidence

This folder archives the raw files found in C:\pioneer\agtc-native-acceptance-backup\audit_stack\repo_integration_before on 2026-10-25. The directory tree and file bytes were copied as-is. manifest.json lists each archived raw file's original relative path, archive relative path, byte length, and SHA-256; the copy was checked file by file. README.md and manifest.json are generated archive metadata and are not entries in the raw-file list.

## Recorded outcomes

- CPU raw logs: cpu_combined_v1.txt and cpu_graph_mode_fresh_flags_v1.txt each record 9 passing tests; cpu_new_tests_v1.txt records 3 passing tests. cpu_identity_canonical_v1.txt records a CPU device and canonical identity. cpu_reset_test_current.txt is an earlier failing attempt (Ran 1 test ... FAILED (errors=1)) and remains unmodified. The coordinator later reported that the final CPU suite passed 10 tests twice, but no raw transcript for those final runs was present in the source folder; this archive does not synthesize one.
- GPU candidate smoke v5 failed its final-contact acceptance check. The later v5b run is recorded as passed for a 3-world, 20-policy-step probe; it reports finite state, zero capacity overflow, and untouched-world preservation on partial reset.
- CUDA Graph v5 was explicitly terminated after 20 policy steps per backend with exit 143. The termination record preserves the exact PID and reason. CUDA Graph v6 did not reach simulation: it failed at startup because the output directory already existed (exit 1).
- CUDA Graph v6b completed 60 policy steps on each path but failed strict state equivalence beginning at policy step 25, including after partial reset. Its reported step-call timer ratio is about 9.78x; because the compared trajectories diverged, this is diagnostic timing only and is not a validated acceleration result.
- Eager A/B with the default non-deterministic mode completed 60 steps per path but failed repeated-run state equivalence at step 27. This demonstrates that the observed divergence is not unique to CUDA Graph.
- The RUN_TO_RUN eager attempt did not complete a policy step. The Python process was terminated with SIGTERM after high CPU/memory use while no policy progress had been recorded; the runner exit file says 143 and the original progress remains running. It is not evidence that RUN_TO_RUN initialized or passed.
- Quiet-mask attempt v1's probe result itself says passed for 20 policy steps, but the enclosing WSL runner returned exit 1 because Bash received a CRLF exit 0 line (0 followed by carriage return); runner_exit_observation.txt records this wrapper failure. Quiet-mask v1b is the corrected, successful wrapper run: outer exit 0, probe passed 20 policy steps and 1001 physics substeps. The actual warning mask was 3071 with only LS_ITERATIONS removed from the console-warning mask; the per-world line-search counters remained nonzero, while capacity and outer-iteration counters were zero. This is a logging profile change, not a solver-parameter change.

## Snapshot timing and provenance

- Files under each run's input_snapshot directory are the captured pre-launch inputs for that attempt. They remain as captured and are covered by the raw-file hashes. The Graph v5/v6/v6b, eager A/B, RUN_TO_RUN, and quiet-mask v1/v1b attempts have pre-launch snapshots in their respective folders.
- Graph v6b and eager A/B also retain post-run hash checks. Quiet-mask v1b retains its post-run source note and source hash. Those files are post-run evidence; they do not replace the pre-launch snapshot.
- The two deterministic_mode_v6_prechange files are a pre-change source backup, not a completed RUN_TO_RUN source snapshot.
- The GPU smoke v5/v5b folders contain their probe drivers, launch output and result records, but no complete separate input_snapshot bundle. No missing backend snapshot is reconstructed here. CPU logs likewise do not include a matching source bundle.
- Top-level .pre_* source files are historical backups, not claimed to be the active source at any particular run.

## Archive exclusions

Three .pyc files in __pycache__ directories were excluded. No model checkpoint, training result directory, dependency cache, or Warp compilation cache was copied. All other files in the source evidence tree, including small NPZ state captures, source snapshots, commands, logs, exit/status files and hashes, were copied and verified.
