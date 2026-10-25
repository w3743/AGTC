from __future__ import annotations
import json, sys, traceback
from pathlib import Path
proto = Path("/mnt/c/pioneer/先进通用履带底盘工程/算法/仿真训练环境/native/proto")
sys.path.insert(0, str(proto))
import test_nonfinite_guards as checks
results = {}
for name in ("test_env_recovers_injected_nan", "test_provider_guards_nonfinite_frames", "test_train_wsl_failure_path"):
    try:
        results[name] = checks.__dict__[name]()
    except BaseException as exc:
        results[name] = {"pass": False, "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()}
passed = all(
    results.get("test_env_recovers_injected_nan", {}).get("recovered", False)
    and results.get("test_train_wsl_failure_path", {}).get("pass", False)
    and "error" not in results.get("test_provider_guards_nonfinite_frames", {})
    for _ in (0,)
)
print("NONFINITE_GUARDS " + json.dumps({"pass": passed, "results": results}, ensure_ascii=False, sort_keys=True))
raise SystemExit(0 if passed else 1)