"""Compare the rendered board against its independent source netlist."""
from pathlib import Path
import json
import math
import re
import sys

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT.parents[2] / "检查/联合仿真/code_as_eda"))
from reference import verify_cli


def verify_passive_values():
    """Catch unit parsing errors independently from component and net matching."""
    reference = json.loads((PROJECT / "reference_268_netlist.json").read_bytes())
    circuit = json.loads((PROJECT / "exports/circuit.json").read_bytes())
    actual = {item["name"]: item for item in circuit if item["type"] == "source_component"}
    scale = {"": 1, "p": 1e-12, "n": 1e-9, "u": 1e-6, "m": 1e-3, "k": 1e3, "M": 1e6}
    failures = []
    checked = 0
    for part in reference["components"]:
        ref = part["ref"]
        if not ref.startswith(("R", "C", "L")) or set(part["pins"]) != {"1", "2"}:
            continue
        match = re.fullmatch(r"(\d+(?:\.\d+)?)([pnumkM]?)(Ω|F|H)", part["value"])
        if not match:
            failures.append(f"{ref}: missing normalized nominal value")
            continue
        expected = float(match.group(1)) * scale[match.group(2)]
        field = {"R": "resistance", "C": "capacitance", "L": "inductance"}[ref[0]]
        rendered = actual.get(ref, {}).get(field)
        checked += 1
        if not isinstance(rendered, (int, float)) or not math.isclose(rendered, expected, rel_tol=1e-9):
            failures.append(f"{ref}: expected {expected}, rendered {rendered}")
    if failures:
        raise ValueError("Passive value mismatches: " + "; ".join(failures))
    print(f"PASS: {checked} passive nominal values in SI units")


if __name__ == "__main__":
    verify_passive_values()
    verify_cli(PROJECT, 268)
