# Pilot06 rejected physical audit package

This package is an external, read-only audit snapshot of the completed pilot06 run and its four-seed evaluation. The raw run text artifacts and evaluation outputs were copied byte-for-byte; `artifact_copy_manifest.json` records each source and copy SHA256. `SHA256SUMS.txt` covers all package files except itself.

The trial manifest records `exit_code: 0` and `formal_eligible: true`, but the physical result is **rejected**. The raw training evidence records iteration 185 / step 15 with 64 non-finite environments and 0 non-finite actions. The complete `nonfinite_events.jsonl` is preserved without JSON parsing and reserialization. It includes a first-world `qpos_sample` of `[NaN, NaN, NaN]` and the other sampled worlds near z = -108.225 m.

A concurrent physics audit supplied by the root agent identified a world-0 wheel state read broadcast to all worlds, missing contacts (`None`), and per-world/control/reset defects. This package does not repeat those findings on GPU. It does not interpret the four-seed 0/4 result as a policy-quality result.

CPU-only checkpoint inspection used `/opt/venv/bin/python`, `weights_only=True`, `map_location=cpu`, one CPU thread, and an explicit safe allowlist for `torch.torch_version.TorchVersion`. CUDA remained uninitialized. The first safe-load attempt without that allowlist was rejected; its error output is retained, and no unsafe pickle fallback was used. Both final checkpoints are iteration 300, with finite policy and optimizer tensors and Adam step 1200.

The current `trial_checks.py preflight-checkpoint` command was run against the original `latest.pt` with a fresh external output target. It exited 5 because metrics row 185 reports `nonfinite_env_resets=64` and `nonfinite_events.jsonl` is non-empty. No preflight output file was created, and the source `trial_manifest.json` SHA256 was unchanged before and after.

The summary's implied nominal average of about 330.2 samples/s is retained only as arithmetic over the run counters and wall time. It is not accepted as usable physical throughput. No controlled 7.7x speedup claim is accepted.

No repository files, source run artifacts, Git state, or checkpoint files were modified. Raw run/evaluation artifacts were copied into this external package; no checkpoint bytes were copied. This task did not initialize CUDA or run a GPU benchmark.