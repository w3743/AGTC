"""Reproduce the CPU-only direct-track PolicyInspector graph audit."""
from __future__ import annotations

import importlib.util
import json
import re
import runpy
import tempfile
from pathlib import Path

import numpy as np
from stable_baselines3.common.vec_env import DummyVecEnv

from isaac_field.action_contract import ACTION_LABELS, action_config
from isaac_field.policy import PolicyRunner
from isaac_field.visualization.inspection import PolicyInspector
from isaac_field.visualization.network_2d import _fields_for
from isaac_field.visualization.network_panel import PolicyNetworkPanel
import isaac_field.visualization.network_panel as network_panel_module


ROOT = Path(__file__).resolve().parents[3]


def _load_test_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def main() -> None:
    policy_fixtures = _load_test_module(
        "direct_tracks_policy_fixture", ROOT / "tests/test_policy_inspection.py"
    )
    panel_fixtures = runpy.run_path(str(ROOT / "tests/test_network_panel.py"))
    env = DummyVecEnv([policy_fixtures._InspectionEnv])
    original_panel_file = network_panel_module.__file__
    try:
        runner = PolicyRunner(env, device="cpu")
        runner.model.policy.eval()
        inspector = PolicyInspector(runner)
        inspector.action(policy_fixtures._observation(20261020), 0.0)

        layers = list(inspector.layers.values())
        layer_keys = [str(layer["key"]) for layer in layers]
        layer_key_set = set(layer_keys)
        action_layer = inspector.layers["action.applied"]
        action_shape = list(np.asarray(action_layer["values"]).shape)
        action_fields = list(_fields_for("action.applied"))
        unresolved_parents = [
            {"node": str(layer["key"]), "parent": str(parent)}
            for layer in layers
            for parent in layer.get("parents", ())
            if str(parent) not in layer_key_set
        ]
        physical_nodes = [
            key for key in layer_keys
            if key.startswith("actuator.") or key == "motion.actual"
        ]
        snapshot_layer_keys = [str(layer["key"]) for layer in inspector.snapshot["layers"]]
        current_action_config = action_config()

        panel = PolicyNetworkPanel(panel_fixtures["FakeGui"](), lambda *_args: None)
        panel.inspector = inspector
        panel.display_snapshot = inspector.snapshot
        with tempfile.TemporaryDirectory(prefix="direct_tracks_export_audit_") as temp_dir:
            network_panel_module.__file__ = str(
                Path(temp_dir) / "isaac_field/visualization/network_panel.py"
            )
            panel.export()
            export_root = Path(temp_dir) / "results/isaac_field_viewer/network_exports"
            sidecars = sorted(export_root.glob("*.json"))
            if len(sidecars) != 1:
                raise RuntimeError(f"expected one export sidecar, found {len(sidecars)}")
            sidecar = json.loads(sidecars[0].read_text(encoding="utf-8"))
            exported_layer_keys = [str(layer["key"]) for layer in sidecar["layers"]]
            sidecar_action_config_matches = sidecar.get("action_config") == current_action_config
            sidecar_layer_order_matches = exported_layer_keys == snapshot_layer_keys
            sidecar_physical_nodes = [
                key for key in exported_layer_keys
                if key.startswith("actuator.") or key == "motion.actual"
            ]

        html = (ROOT / "isaac_field/visualization/network_2d.html").read_text(encoding="utf-8")
        start = html.index("function overviewGraph(nodes) {")
        end = html.index("function detailedGraph(nodes)", start)
        overview_source = html[start:end]
        slot_match = re.search(r"const terminalSlots = \[(.*?)\];", overview_source, re.S)
        slot_names = re.findall(r"^\s*\[(\w+),", slot_match.group(1), re.M) if slot_match else []
        column_match = re.search(r"columns: \[(.*?)\], rowLabels:", overview_source, re.S)
        column_labels = re.findall(r'"([^\"]+)"', column_match.group(1)) if column_match else []
        overview_action_lookup = (
            'findNode(nodes, "action.applied") || findNode(nodes, "action_net")'
            in overview_source
        )
        overview_has_removed_outputs = (
            "motion.actual" in overview_source or "actuator." in overview_source
        )
        overview_ends_at_action = bool(
            overview_action_lookup
            and slot_names
            and slot_names[-1] == "action"
            and column_labels
            and column_labels[-1] == "四维动作"
            and not overview_has_removed_outputs
            and "action.applied" in layer_key_set
        )

        checks = {
            "action_shape_is_four": action_shape == [4],
            "action_field_order_matches_contract": (
                action_fields == list(ACTION_LABELS)
                and len(current_action_config["components"]) == 4
            ),
            "all_layer_parents_exist": not unresolved_parents,
            "no_physical_or_mechanical_graph_nodes": not physical_nodes,
            "snapshot_layer_order_matches_inspector": snapshot_layer_keys == layer_keys,
            "export_layer_order_matches_snapshot": sidecar_layer_order_matches,
            "export_sidecar_has_action_config": sidecar_action_config_matches,
            "export_has_no_physical_or_mechanical_layers": not sidecar_physical_nodes,
            "overview_endpoint_is_action_applied": overview_ends_at_action,
        }
        report = {
            "audit": "direct tracks policy graph",
            "date": "2026-10-20",
            "model_instance": (
                "PolicyRunner(make_model(...), device='cpu'); fresh random initialization; "
                "inference only, no training"
            ),
            "source_fixture": "tests/test_policy_inspection.py::_InspectionEnv",
            "action_config": current_action_config,
            "action_applied": {
                "shape": action_shape,
                "fields": action_fields,
                "parents": list(action_layer.get("parents", ())),
            },
            "graph": {
                "layer_count": len(layer_keys),
                "layer_keys": layer_keys,
                "physical_or_mechanical_nodes": physical_nodes,
                "unresolved_parents": unresolved_parents,
            },
            "snapshot_export": {
                "snapshot_layer_count": len(snapshot_layer_keys),
                "snapshot_layer_keys": snapshot_layer_keys,
                "sidecar_layer_keys": exported_layer_keys,
                "sidecar_action_config_matches": sidecar_action_config_matches,
                "physical_or_mechanical_nodes": sidecar_physical_nodes,
            },
            "overview_source_check": {
                "terminal_slots": slot_names,
                "column_labels": column_labels,
                "ends_at": "action.applied" if overview_ends_at_action else None,
                "method": (
                    "checked current overviewGraph terminal slots and final column; "
                    "node keys came from an actual PolicyInspector graph"
                ),
            },
            "checks": checks,
            "passed": all(checks.values()),
        }
        output = ROOT / "results/isaac_field_viewer/direct_tracks_20261020/graph_audit.json"
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(report, ensure_ascii=False, indent=2))
        if not report["passed"]:
            raise SystemExit("graph audit check failed; inspect graph_audit.json")
    finally:
        network_panel_module.__file__ = original_panel_file
        env.close()


if __name__ == "__main__":
    main()
