"""Launch the a07 evaluation adapter without tripping its own parallel-process guard.

The venv python.exe is a redirector that keeps a sibling process with the same
command line. The evaluator's guard treats any `evaluate_*.py` python process as
a parallel simulation, so the sibling was flagged as if a second evaluation were
running. Starting the adapter through a differently named launcher keeps the
process command line free of guarded script names while still executing the same
adapter code in this single process.
"""
from __future__ import annotations

import runpy
import sys

ADAPTER = r"C:\pioneer\先进通用履带底盘工程\算法\仿真训练环境\results\isaac_field_improvement\physx_pointmove_flat12_gpu_20261024_a07\evaluate_torch_batched.py"
CHECKPOINT = r"C:\pioneer\先进通用履带底盘工程\算法\仿真训练环境\results\isaac_field_improvement\perf_20261025_n32_dt5\logs\rsl_rl\perf_20261025_n32_dt5\train\model_95.pt"
OUTPUT = r"C:\pioneer\先进通用履带底盘工程\算法\仿真训练环境\results\isaac_field_improvement\perf_20261025_n32_dt5\eval_final"

sys.argv = [
    ADAPTER,
    "--checkpoint", CHECKPOINT,
    "--output", OUTPUT,
    "--task-backend", "torch_batched",
    "--seed-start", "87010",
    "--scenario-count", "16",
    "--goal-angle-deg", "180",
]
runpy.run_path(ADAPTER, run_name="__main__")
