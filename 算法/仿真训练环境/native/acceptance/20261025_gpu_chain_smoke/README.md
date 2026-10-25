# Native GPU chain smoke

This is a sequential runtime chain smoke in an external backup directory. It is not a benchmark, capability evaluation, or success-rate run.

1. CPU-only audit of the preflight latest checkpoint found iteration 2, Adam step 8, and finite policy/optimizer tensors.
2. One GPU resume iteration completed at iteration 3. The saved optimizer step advanced from 8 to 12 across four PPO epochs; checkpoint_3.pt and latest.pt are finite and retain the same v2 task identity SHA.
3. One GPU evaluation loaded that checkpoint with checkpoint_contract_validation strict_match and the same SHA. Seed 87010 reached the 32-step time limit with no runtime error or non-finite diagnostics. The outcome is recorded only as a short smoke.
4. A separate evaluation request using physics dt 0.005 / decimation 10 exited nonzero at checkpoint preflight. The requested output directory and evaluation.json were not created.

All GPU phases ran serially. The original preflight checkpoint remains untouched. Checkpoint audit is CPU-only and does not query or initialize CUDA.

See chain_summary.json for exact timings, row data, contract SHA, checkpoint hashes, validation details, and artifact inventory. argv and stdout/stderr captures are retained per phase. SHA256SUMS.txt covers all other files recursively.
