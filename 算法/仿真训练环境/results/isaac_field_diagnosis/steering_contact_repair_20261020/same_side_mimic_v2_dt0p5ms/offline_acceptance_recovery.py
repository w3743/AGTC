from __future__ import annotations
import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
CASE = HERE / "contact_probe_runs_dt0p5ms/dt-0p5ms__cone-elliptic__impratio-1__mu-0.8__kf-1000__equality-softening-v2__attempt1"
BASELINE = ROOT / "results/isaac_field_diagnosis/steering_contact_repair_20261020/same_side_mimic_v2_dt1ms/contact_probe_runs_dt1ms/dt-1ms__cone-elliptic__impratio-1__mu-0.8__kf-1000__equality-softening-v2__attempt2/same_side_mimic_v2_dt1ms_summary.json"
RUNNER = HERE / "contact_mimic_v2_dt0p5ms_runner.py"
OUTPUT = CASE / "offline_acceptance_recovery.json"
if OUTPUT.exists():
    raise SystemExit(f"Refusing to overwrite recovery result: {OUTPUT}")
spec = importlib.util.spec_from_file_location("contact_mimic_v2_dt0p5ms_postprocess", RUNNER)
if spec is None or spec.loader is None:
    raise RuntimeError(f"cannot load diagnostic runner: {RUNNER}")
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
if not BASELINE.is_file():
    raise FileNotFoundError(BASELINE)
module.BASELINE_SUMMARY_PATH = BASELINE
acceptance = module._post_run_coupling_acceptance(CASE)
summary_path = CASE / "same_side_mimic_v2_dt0p5ms_summary.json"
summary = json.loads(summary_path.read_text(encoding="utf-8"))
result = {
    "schema": "isaac-field-same-side-mimic-dt0p5ms-postprocess-recovery-v1",
    "motion_attempt": "attempt1",
    "motion_runner_actual_exit_code": 1,
    "motion_completed": summary.get("status") == "complete",
    "motion_runner_error": "post-run acceptance initially failed because baseline summary filename was mispointed; physics tape completed and was preserved",
    "offline_acceptance": acceptance,
    "baseline_summary": str(BASELINE),
    "output_summary": str(summary_path),
    "native_physics_substeps_expected": summary.get("native_physics_substeps_expected"),
    "native_physics_substeps_captured": summary.get("native_physics_substeps_captured"),
}
OUTPUT.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
print(json.dumps(result, ensure_ascii=False, allow_nan=False))
