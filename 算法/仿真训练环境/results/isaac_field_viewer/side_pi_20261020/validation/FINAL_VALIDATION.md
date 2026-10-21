# Final side PI checkpoint validation

This separate CPU-only entry point reuses the existing v6 checkpoint inspection helpers while pinning the current protocol SHA, 41-source snapshot, baseline SHA, and 196608 expected steps. It does not alter the original 8192-step validator. It reads baseline/final metadata and Adam state, checks finite weights and optimizer values, reports branch changes against baseline (including detector status without requiring detector updates), verifies checkpoint manifest and current/snapshotted source hashes, protects the older trial hashes, and compares relevant input-file signatures before and after reading. It does not construct an environment, evaluate seeds, train, or use CUDA.

After the final checkpoint is actually complete, run from the repository root:

```bash
bash results/isaac_field_viewer/side_pi_20261020/validation/run_final_checkpoint_validation.sh \
  "$PWD/results/isaac_field_improvement/fresh_side_pi_20261020" \
  "$PWD/results/isaac_field_improvement/fresh_side_pi_20261020/stage_04_turn90.zip" \
  final_196608
```

The runner writes `final_196608.json`, `final_196608.log`, and `final_196608.exit.txt` in this directory and refuses to overwrite existing outputs. The validator requires `--expected-steps 196608` and only writes the requested JSON under this validation directory.
