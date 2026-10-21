# Production-candidate CPU validation (prepared)

This directory contains the full CPU-only verification runner. `attempt1` has completed after the parent trigger; no GPU was used.

The runner followed the project's established CPU mode: it unset `RUN_ISAAC_PHYSICS`, set `CUDA_VISIBLE_DEVICES` empty, ran full pytest discovery with the official `scripts/isaaclab_python.sh` runtime, compiled `isaac_field`, `scripts`, and `tests`, and ran `bash -n` on every saved shell script under `scripts/` and this diagnosis directory.

`attempt1` results: **244 passed, 14 skipped, 8 warnings** in 12.73 s; pytest, compileall, bash syntax and the WSL process all returned actual exit 0. The 14 skips are from this CPU-only invocation; the production native/GPU suite is a separate run. The warnings were one NumPy overflow warning from the extreme-value visualization test, two TorchScript tracing deprecations, and five `TracerWarning`s about tensor-to-Python boolean checks. All 39 discovered Bash scripts passed `bash -n`. WSL emitted its localhost/NAT warning before returning; the WSL process exit is recorded as 0 in [`attempt1/wsl_process_exit.json`](attempt1/wsl_process_exit.json).

The four production/test files in B's preflight source manifest still match their preflight SHA-256 values after CPU validation. The manifest also hashes the 39 Bash targets and the validation logs: [`attempt1/validation_hash_manifest.json`](attempt1/validation_hash_manifest.json). Raw pytest and syntax outputs are in [`attempt1/full_cpu_pytest.log`](attempt1/full_cpu_pytest.log) and [`attempt1/bash_syntax.log`](attempt1/bash_syntax.log); compileall completed with an empty log and exit 0.

For a later fresh validation attempt, run from the project root in WSL with a new attempt name:

```bash
RUN_STEERING_CONTACT_CPU_VALIDATION=1 bash results/isaac_field_diagnosis/steering_contact_repair_20261020/cpu_validation/run_full_cpu_validation.sh attempt2
```

The script is guarded by that explicit environment variable and refuses an existing output directory. It records per-stage exit codes in its `attemptN/validation_exit.json`; the actual Bash process exit must also be recorded by the caller. This CPU suite does not replace the production native/GPU suite.
