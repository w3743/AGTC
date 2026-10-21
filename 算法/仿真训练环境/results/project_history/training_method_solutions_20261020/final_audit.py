"""Record current sources, old evidence integrity and prepared-trial status."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from scripts import train_isaac_mastery as runner

expected = {
    "results/isaac_field_improvement/fresh_coupled_tracks_20261020/protocol.json": "2735ca85917643e49e27088957d00b1576f37902ce9664203253ae698181d5b6",
    "results/isaac_field_improvement/continue_coupled_tracks_20261020/protocol.json": "fe54ce6233e10705a9bbc393784eeca8cbe4d459ac467af1b2fd3656fb4baa18",
    "results/isaac_field_improvement/fresh_coupled_tracks_20261020/stage_04_turn90.zip": "6b4c85884a51c2e23f557eb98bc04ea63905fc8f20d4379715fb9fe72b75cdc5",
    "results/isaac_field_improvement/continue_coupled_tracks_20261020/stage_04_turn90.zip": "9777af6addd30427026675c0fb9d577c555fd288294bc6129a80b53b1d633a70",
}
integrity = []
for name, digest in expected.items():
    actual = hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
    integrity.append({"path": name, "sha256": actual, "matches_historical_record": actual == digest})

prepared = []
for name in ("method_control_v2_20261020", "method_mastery_v2_20261020", "method_lambda099_v2_20261020"):
    output = ROOT / "results/isaac_field_improvement" / name
    try:
        runner.preflight(output)
        result = {"trial": name, "preflight_now": "passed"}
    except Exception as exc:
        result = {"trial": name, "preflight_now": "rejected", "error": str(exc)}
    result["formal_training_started_by_this_task"] = False
    prepared.append(result)

sources = [{"path": name, "sha256": hashlib.sha256((ROOT / name).read_bytes()).hexdigest()}
           for name in runner._sources()]
record = {"timestamp_utc": datetime.now(timezone.utc).isoformat(), "historical_integrity": integrity,
          "current_preflight_audit": prepared, "current_sources": sources,
          "independent_changes_this_task": ["training_reward_action_optimizer_diagnostics",
              "mastery_gates_and_actual_old_task_plant_sampling", "fixed_suite_safe_checkpoint_selection",
              "frozen_controlled_experiment_runner_and_gae_configuration"],
          "batch_count_this_task": "4/5 from documented 0/5; excludes concurrent front-guide/stability work",
          "formal_training_started": False, "new_learning_improvement_established": False,
          "concurrent_work": "env.py/newton_drive.py/front-guide and body-stability are owned by the other chat; this task has finished writing env.py and learning.py. Its optional body-pose/angular-velocity/termination detail fields are preserved.",
          "gpu_validation": "One 2-step Newton accounting/autoreset test passed before the coordination request; no additional GPU validation or training started after that request.",
          "automation": "no changes made; latest user-requested PAUSED state respected"}
runner._json(Path(__file__).parent / "final_audit.json", record)
print(json.dumps({"historical_integrity_all_match": all(row["matches_historical_record"] for row in integrity),
                  "prepared": prepared, "sources": len(sources)}, ensure_ascii=False))
if not all(row["matches_historical_record"] for row in integrity):
    raise SystemExit(1)
