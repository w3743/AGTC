# Native V2 freeze validation archive

This archive preserves validation evidence for frozen protocol SHA 83db5a6e512d7d7244bc47ddf2059720c757504270e9bbf9129d0c7d8f92f87e. Files were copied from the external evidence directories listed in archive_manifest.json. Original bytes are retained; __pycache__, cache/.cache directories, and .pyc/.pyo files are excluded.

## Evidence groups

- 01_mjw_v6_physics_gate_prepared_stale_manifest: preserves the earlier physics-gate bundle, its logs/results, and stale prelaunch provenance. The raw world1/3/64 attempts show child exit code 0, but their input manifest predates the final driver/preregistration bytes and they are not accepted gate evidence. COMMAND_NOT_RUN.txt is a prepared command template; it does not negate the separately retained historical attempt files in runs/.
- 02_v6_gate_verifiedinputs_20261025_2230: the later preregistered 1/3/64 gate. Its three cases passed, each with 20 policy steps and 1,000 policy physics substeps plus diagnostic overhead. The final driver and preregistration inputs were SHA-verified before launch against manifest e623b12ce038165204d2ee79aee47a3847fc2c087557662875507ba452c59b50. This is a bounded physics gate, not a long training or capability acceptance.
- 03_v2_final_cpu_20261025_223526: CPU10 Newton backend tests passed (10 tests, exit 0), and the AST-extracted _write_live_state fake-environment call passed for four environments. It checked local goals, origin-translated body XY, and per-environment episode_time. The input source-set hash was unchanged before/after: e6e0f1e0031d0dec27bd632ec6f7b28e94a727d821323a698a2751c73635ff31. nonfinite_smoke.py and run_cpu_checks.py are retained but were NOT RUN in this CPU task.
- 04_root_docs_validation_native_v2_frozen: raw stdlib test transcripts from the coordinator: Windows 61 tests, exit 0; WSL 61 tests with 2 PowerShell-only skips, exit 0.

The CPU10 run explicitly used CPU. Warp's import emitted a no-CUDA-device diagnostic because CUDA_VISIBLE_DEVICES was empty; this was not a GPU test or a GPU failure. The AST smoke extracted the pure function from train_wsl.py and called it with a NumPy fake backend/exporter; it did not import the training module or launch training.

This archive does not include the separate live-view geometry JSON; the coordinator requested that artifact be archived with the viewer consumer once that integration is complete. No source, protocol, training run, or Git state was changed by this copy operation.