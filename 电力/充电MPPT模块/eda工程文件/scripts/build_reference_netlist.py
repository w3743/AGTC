"""Build an immutable Board 6 reference from the Stage 1 drawing contract."""
from pathlib import Path
from collections import Counter
import sys
PROJECT = Path(__file__).resolve().parents[1]
POWER = PROJECT.parents[1]
sys.path.insert(0, str(POWER / "检查/联合仿真/code_as_eda"))
from reference import read_json, require, source_record, apply_drawing_rows, write_json
CONTRACT = PROJECT.parent / "检查/充电MPPT施工合同.json"
SIM = PROJECT.parent / "检查/充电MPPT与自持仿真核算.json"
EXPECTED = {"contract": "3f927e35d878c54d64dd8b896dce2be237be7f4ba4a9807ed5c5a684b9e82af9", "simulation": "3fd45668f937bac1713197beba060fba34734bf6ef5d736abf284575bacdc01e"}
def build():
    contract, sim = read_json(CONTRACT), read_json(SIM)
    inputs = {"contract": source_record(CONTRACT), "simulation": source_record(SIM)}
    require(all(inputs[k]["sha256"] == v for k, v in EXPECTED.items()), "Stage 1 input hash mismatch")
    require(contract["status"] == "CONTRACT_COMPLETE_EDA_CAPTURE_PENDING", "Unexpected contract status")
    require(sim["status"] == "BOUNDARY_SIMULATION_COMPLETE_FINAL_SCHEMATIC_PENDING", "Unexpected simulation status")
    rows = contract["drawing_components"]
    parts = apply_drawing_rows({}, rows)
    require(len(parts) == len(rows) == contract["target_component_count"] == 433, "Unexpected object count")
    reference = {"contract_id": contract["contract_id"], "target_schematic_count": len(parts), "inputs": inputs,
                 "counts_by_placement": dict(sorted(Counter(p["placement"] for p in parts.values()).items())),
                 "notes": ["Captured from the Stage 1 contract; power connection, purchasing and bench qualification remain on hold."],
                 "components": sorted(parts.values(), key=lambda p: p["ref"])}
    write_json(PROJECT / f"reference_{len(parts)}_netlist.json", reference)
    write_json(PROJECT / "src/board_parts.json", {"components": reference["components"]})
    print(f"Wrote {len(parts)} objects")
if __name__ == "__main__":
    build()
