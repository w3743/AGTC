# Drive controller validation record

This is an after-the-fact transcription from the tool output and task notes in this turn. The focused CPU run and the small CUDA equivalence run did not save separate stdout, stderr, JUnit, or process files. The result text below is copied as reported; no measurements were reconstructed. The saved CUDA reproduction script is reconstructed for future use and was not run during this documentation update.

## Focused CPU check

- Command: `C:/pioneer/isaac-training/.venv/Scripts/python.exe -B -m pytest -q -p no:cacheprovider --tb=short tests/test_drive_control.py`
- Working directory: `算法/仿真训练环境`
- Environment: `CUDA_VISIBLE_DEVICES=''`, `ISAAC_FIELD_DISABLE_NATIVE_RUNTIME=1`, `RUN_ISAAC_PHYSICS=0`
- Tool output: `28 passed in 3.27s`
- A separate numeric process exit code and raw output file were not preserved.

## CUDA equivalence check

The original check ran an inline Python body on `cuda:0` with native runtime and physics disabled. It compared the controller to the public-helper reference over 18 steps, including partial reset, saturation and unwinding, and topology cloning. The actual tool output was:

```text
CUDA equivalent: device=cuda:0, steps=18, partial_reset=true, saturation=true, topology_clone=true
```

The original inline body, separate numeric exit code, and raw process output were not preserved. [`drive_cuda_equivalence_repro.py`](drive_cuda_equivalence_repro.py) reconstructs the check; it has not been executed again. This was a correctness check, not a benchmark, and produced no FPS or speedup measurement.

## Full CPU gate

The later complete gate is preserved as the canonical regression evidence at [`cpu_gate_after_drive_20261024T1233Z/cpu_gate_after_drive.json`](cpu_gate_after_drive_20261024T1233Z/cpu_gate_after_drive.json), with saved command, process, stdout, stderr, JUnit, and before/after source hashes. It completed with actual exit code 0 and reported `546 passed, 4 skipped, 47 warnings, 10 subtests passed in 61.60s (0:01:01)`. It disabled CUDA and native physics, and reports that training and FPS benchmarking did not start.
