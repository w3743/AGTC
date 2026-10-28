# Native GPU preflight

This is one short CUDA training preflight, not a benchmark or throughput acceptance run.

The run used LSTM-512, 64 Newton XPBD CUDA environments, 32 policy steps, two PPO iterations, and task_backend fast_stop_v2. It exited with code 0 and produced iteration-2 checkpoints. A read-only post-run check confirmed the listed training source files match commit 953b355f.

checkpoint_validation.json records CPU-only safe deserialization and finite checks: policy and optimizer tensors are all finite for both checkpoint_2.pt and latest.pt.

The source recorded the v2 task identity SHA and actual observation backends. Capability metrics remain disabled because the depth and detections RTX domain gap has not been pixel-compared.

Keep the checkpoints in this external folder for the later smoke evaluation. Do not use the two iteration rates as a formal performance result.

argv.json records the exact WSL argument vector; exit.json records timing and exit status; rawlog.stdout.txt and rawlog.stderr.txt preserve the captured output channels. SHA256SUMS.txt covers every other file in this directory.
