# Host and pilot01 read-only probe

- Host: Windows 11 ARM64, WSL2 Ubuntu-24.04-Arm64 (aarch64), RTX Spark N1X.
- Python: `/opt/venv/bin/python` (3.12.3). Package versions and sanitized pip-index status are in `packages.json`; `psutil` is absent.
- An active pilot process was observed at 2026-10-25 16:15:16 +08 (PID 78806, parent 78798, shell wrapper 78791; command details are summarized, not copied). It then wrote 20 metric rows and checkpoint 20. The runner manifest says exit code 1 at 16:18:12 +08; no summary was written. This is not natural completion.
- Metrics snapshots captured at 16:19:32 and 16:20:40 +08 are byte-identical at iteration 20, so the pilot had already exited before the first saved snapshot. The earlier live process observation and final checkpoint establish that it ran and reached 20; the two saved snapshots do not show live growth.
- `checkpoint_20.pt` was CPU-loaded with `weights_only=True`; payload iteration is 20. Its SHA256 is in `checkpoint_inspection.json` and `source_hashes.json`.
- The old identity and observation contract do not record `task_backend`; the contract's reward note describes the simplified throughput-prototype task. All five observation channels are marked `real`, while `capability_metrics_allowed` is false because RTX domain gaps are not accepted.
- `llama-server.exe` PID 48968 was observed before the stop request. Non-force `taskkill` said force was required; five seconds later the PID was absent, and GPU memory fell from 27,183 MiB to about 5,140 MiB. No force termination was used. LM Studio and the unrelated `aria2c` download were left running.
- No benchmark, package installation, WSL restart, Git write, or modification of the original run was performed. After the server disappeared, no training/inference compute process appeared in the WDDM process table; desktop GPU use remained at about 5 GiB / 9%, so the machine was not fully idle.

All files here are newly created evidence snapshots. Original run files are referenced by path and SHA and were not changed by the probe.