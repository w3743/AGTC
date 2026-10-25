"""Merge the VQF bypass results into m1_verification.json and update capability_blockers."""
import json
from pathlib import Path

here = Path("/mnt/c/pioneer/先进通用履带底盘工程/算法/仿真训练环境/native/proto")
m1 = json.loads((here / "m1_verification.json").read_text(encoding="utf-8"))
vqf = json.loads((here / "m1_vqf_verification.json").read_text(encoding="utf-8"))
m1["feedback_vqf_bypass"] = vqf
m1["feedback"]["vqf_bypass_available"] = True
m1["feedback"]["vqf_bypass_residual"] = vqf["attitude_quaternion_max_abs_error"]
m1["feedback"]["direct_attitude_error"] = m1["feedback"]["vqf_attitude"]["attitude_quat_max_abs_error"]
m1["capability_blockers"] = [
    "depth placeholder (constant 0.5) - not equivalent to the RTX depth camera",
    "detections placeholder (zeros) - not equivalent to instance segmentation",
    "state inputs use simulator truth heading/gimbal; the reference uses the VQF-estimated pose "
    "(difference is small but not yet quantified with the VQF backend enabled) - follow-up ~0.25 d",
]
m1["vqf_note"] = ("VQF bypass verified: attitude quaternion residual 0.0 (<=1e-6 target) against an independent "
                  "SimFeedbackEstimator; CPU cost 0.46 ms/env/update (noise off) / 0.76 ms (noise on); "
                  "throughput impact measured separately (see EQUIVALENCE_REPORT.md)")
(here / "m1_verification.json").write_text(json.dumps(m1, indent=2, ensure_ascii=False), encoding="utf-8")
print("M1_JSON_UPDATED", list(m1.keys()))
print("blockers:", len(m1["capability_blockers"]))
