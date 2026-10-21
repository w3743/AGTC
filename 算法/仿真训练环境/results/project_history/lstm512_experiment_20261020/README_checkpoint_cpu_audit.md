# LSTM-512 checkpoint CPU audit

Run after the rough-terrain LSTM-512 trial has a prepared protocol, a formal zero-step `untrained_baseline.zip`, and a positive-timestep checkpoint. The helper loads both SB3 checkpoints on CPU, checks the protocol and live/snapshot source hashes before and after loading, and compares unique named policy parameters, finite weights and Adam state, and actual weight updates. It does not create an environment, train, or run simulation. It refuses to overwrite the report path.

From the project root in Ubuntu-24.04, set `CUDA_VISIBLE_DEVICES=''` and invoke `scripts/isaaclab_python.sh` with this helper plus:

```text
--trial results/isaac_field_improvement/random_global_rough_lstm512_20261020
--checkpoint stage_1_global_random/checkpoints/field_policy_global_random_8192_steps.zip
--output results/isaac_field_improvement/random_global_rough_lstm512_20261020/lstm512_checkpoint_cpu_audit.json
```

The expected unique parameter count is 3,065,801 for a shared 192-feature encoder, separate 512-unit one-layer actor/critic LSTMs, and 64-unit policy/value heads. The output path must not already exist. A passed report records the resolved checkpoint path, SHA-256 values, CPU-only state, architecture, optimizer and weight-comparison checks.

For the completed trial, use `--checkpoint stage_01_global_random.zip` and a new report path. The first-checkpoint report confirms updates and integrity, not learned navigation quality.
