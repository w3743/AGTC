"""Prepare three frozen protocols only; never instantiate a simulator."""
from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from scripts import train_isaac_mastery as runner

plans = (
    ("control", "method_control_v2_20261020", {"method": "fixed"}),
    ("mastery", "method_mastery_v2_20261020", {"method": "mastery"}),
    ("lambda099", "method_lambda099_v2_20261020", {"method": "fixed", "gae_lambda": 0.99}),
)
rows, protocols = [], {}
for name, directory, kwargs in plans:
    output = ROOT / "results/isaac_field_improvement" / directory
    prepared = runner.prepare(output, **kwargs)
    protocol = runner.preflight(output)
    protocols[name] = protocol
    rows.append({**prepared, "variant": name, "preflight": "passed",
                 "protocol_sha256": runner.foundation._sha256(output / "protocol.json")})

control = protocols["control"]
for name in ("mastery", "lambda099"):
    other = protocols[name]
    for key in ("config", "stages", "budget", "block_steps", "initialization", "mastery",
                "selection", "evaluation_environment_seed", "training_seed_base", "source_snapshot"):
        assert other[key] == control[key], (name, key)
assert protocols["mastery"]["hyperparameters"] == control["hyperparameters"]
assert [key for key in control["hyperparameters"]
        if protocols["lambda099"]["hyperparameters"][key] != control["hyperparameters"][key]] == ["gae_lambda"]
result = {"prepared_trials": rows, "protocol_controls_match": True,
          "training_started": False, "model_weights_initialized": False,
          "initial_policy_equal": None,
          "initial_policy_note": "After launch, require equal initialization.policy_tensor_sha256 before attributing paired effects.",
          "capability_improvement_established": False}
runner._json(Path(__file__).parent / "prepared_comparisons.json", result)
print(json.dumps(result, ensure_ascii=False))
