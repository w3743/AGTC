from pathlib import Path
import json
import sys

PROJECT = Path(__file__).resolve().parents[1]
POWER = PROJECT.parents[2]
ROOT = POWER.parent
sys.path.insert(0, str(POWER / "检查/联合仿真/code_as_eda"))
from reference import read_json, require, source_record, write_json


def main():
    scope_path = PROJECT / "CAN接口局部合同.json"
    scope = read_json(scope_path)
    source_path = ROOT / scope["source_reference"]
    source = read_json(source_path)
    requested = scope["refs"]
    original = {part["ref"]: part for part in source["components"]}
    require(len(requested) == len(set(requested)) == 10, "CAN 接口候选器件位号不唯一")
    require(set(requested) <= original.keys(), "水泵基准不含接口器件")
    parts = []
    for ref in requested:
        old = original[ref]
        pins = {pin: scope["net_renames"].get(net, net) for pin, net in old["pins"].items()}
        parts.append({
            "ref": ref, "mpn": old["mpn"], "manufacturer": old["manufacturer"],
            "lcsc": old["lcsc"], "footprint": old["footprint"], "value": old["value"],
            "dnp": old["dnp"], "placement": "board", "in_bom": True, "pins": pins,
            "pin_roles": old.get("pin_roles", {}),
            "source_layers": ["water_pump_reference", "mppt_interface_net_renames"],
        })
    by_ref = {part["ref"]: part for part in parts}
    require(by_ref["J_CAN"]["pins"] == scope["connector_pinout"], "CAN 连接器针序不一致")
    require(by_ref["U_CAN"]["pins"]["10"] == by_ref["U_CAN"]["pins"]["11"] == "CAN_ISO_5V",
            "VISOIN/VISOOUT 未连接")
    require(by_ref["C_CAN_ISOIN_1U"]["pins"] == {"1": "CAN_ISO_5V", "2": "GND_CAN"},
            "VISOIN 1µF 必须直接跨隔离电源和隔离地")
    reference = {
        "contract_id": scope["contract_id"],
        "status": scope["status"],
        "target_schematic_count": len(parts),
        "scope": scope["scope"],
        "inputs": {"source_reference": source_record(source_path), "scope_contract": source_record(scope_path)},
        "components": sorted(parts, key=lambda part: part["ref"]),
        "external_net_boundaries": scope["external_net_boundaries"],
        "limitations": scope["limitations"],
    }
    write_json(PROJECT / "reference_can_interface.json", reference)
    print(f"Reference: {len(parts)} interface components, {sum(len(x['pins']) for x in parts)} pins.")


if __name__ == "__main__":
    main()
