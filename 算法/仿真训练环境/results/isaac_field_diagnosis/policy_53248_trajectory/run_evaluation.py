from __future__ import annotations

import json
import math
from dataclasses import asdict
from pathlib import Path

from isaac_field.env import SingleFieldAdapter, make_env
from isaac_field.learning import evaluate_model
from isaac_field.task import Config

root = Path.cwd()
out = Path(__file__).resolve().parent
checkpoint = root / "results/isaac_field_improvement/staged_goto_20261017_drive_stable/field_policy_after.zip"
seeds = (76001, 76002, 76003, 76004)
goal_distance = (1.0, 3.0)
goal_angle = math.radians(60.0)
cfg = Config(plants=16, terrain_enabled=False, max_time=20.0)
protocol = {
    "purpose": "actual simulator diagnostic, not throughput benchmark",
    "checkpoint": str(checkpoint.resolve()),
    "checkpoint_training_steps": 53248,
    "seeds": list(seeds),
    "environment": asdict(cfg),
    "mode": "goto",
    "goal_distance_m": list(goal_distance),
    "goal_angle_rad": goal_angle,
    "distance_cost": 0.0,
    "deterministic": True,
    "device": "cuda:0",
    "record_trajectories": True,
    "trajectory_schema": "isaac-field-compact-trajectory-v1",
    "scene_seed": 80003,
}
(out / "protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
engine = make_env(
    cfg=cfg,
    mode="goto",
    num_envs=1,
    seed=80003,
    auto_reset=False,
    goal_distance=goal_distance,
    goal_angle=goal_angle,
    distance_cost=0.0,
)
try:
    adapter = SingleFieldAdapter(engine)
    report = evaluate_model(
        adapter,
        checkpoint,
        seeds,
        goal_distance=goal_distance,
        goal_angle=goal_angle,
        mode="goto",
        deterministic=True,
        device="cuda:0",
        record_trajectories=True,
    )
finally:
    engine.close()
(out / "evaluation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
print(json.dumps({"env_closed": True, "out": str(out), "summary": report["summary"], "rows": [{k: row[k] for k in ("seed", "success", "damage", "final_distance", "reason", "steps")} for row in report["rows"]]}, indent=2))
