"""Build and verify a frozen contract overlay without modifying native inputs."""

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

POWER = Path(__file__).resolve().parents[3]
ROOT = POWER.parent
JOINT = POWER / "检查/联合仿真/画图前合同完整性校验.json"


def read_json(path):
    return json.loads(Path(path).read_bytes())


def require(condition, message):
    if not condition:
        raise ValueError(message)


def write_json(path, payload):
    Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                          encoding="utf-8", newline="\n")


def source_record(path):
    path = Path(path).resolve()
    return {
        "path": path.relative_to(ROOT).as_posix() if path.is_relative_to(ROOT) else str(path),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def pin_map(raw):
    return {str(pin): None if value in (None, "", "NC") else value
            for pin, value in raw.items()}


def lcsc_code(value):
    return value if isinstance(value, str) and re.fullmatch(r"C[1-9]\d*", value) else None


def load_baseline(enet):
    parts = {}
    for element in read_json(enet)["components"].values():
        props = element["props"]
        ref = props["Designator"]
        ports = element["pinInfoMap"]
        require(ref not in parts, f"重复底稿位号 {ref}")
        require(all(str(port["number"]) == pin for pin, port in ports.items()), f"底稿针号 {ref}")
        value = props.get("Name") or props.get("Value") or ""
        board = props.get("Convert to PCB") != "no"
        placement = ("board" if board else
                     "schematic-only" if "SYMBOL ONLY" in value.upper() else "off-board")
        parts[ref] = {
            "ref": ref, "mpn": props.get("Manufacturer Part") or None,
            "manufacturer": props.get("Manufacturer") or None,
            "lcsc": lcsc_code(props.get("Supplier Part")),
            "footprint": props.get("FootprintName") or None,
            "value": value, "dnp": bool(re.search(r"\bDNP\b", value, re.I)),
            "placement": placement, "in_bom": props.get("Add into BOM") != "no",
            "pins": pin_map({pin: port["net"] for pin, port in ports.items()}),
            "pin_roles": {pin: port["name"] for pin, port in ports.items()
                          if port.get("name") and pin != port["name"]},
            "source_layers": ["baseline_enet"],
        }
    return parts


def apply_drawing_rows(parts, rows, layer="drawing_components"):
    seen = set()
    for row in rows:
        ref = row["ref"]
        require(ref not in seen, f"{layer} 重复位号 {ref}")
        seen.add(ref)
        previous = parts.get(ref)
        part = dict(previous or {
            "ref": ref, "mpn": None, "lcsc": None, "manufacturer": None, "footprint": None,
            "dnp": False, "placement": "board", "in_bom": True, "pin_roles": {},
            "source_layers": [],
        })
        for field in ("mpn", "manufacturer", "lcsc", "footprint", "dnp", "placement", "in_bom"):
            if field in row:
                part[field] = row[field]
        if "new_mpn" in row and "mpn" not in row:
            part["mpn"] = row["new_mpn"]
        part["lcsc"] = lcsc_code(part["lcsc"])
        if "value" in row or previous is None:
            part["value"] = row.get("value") or part["mpn"] or ref
        if row.get("on_board") is False:
            part["placement"] = "off-board"
        if row.get("on_board") is True:
            part["placement"] = "board"
        if "pins" in row:
            part["pins"] = pin_map(row["pins"])
            part["pin_roles"] = {pin: role for pin, role in part["pin_roles"].items()
                                 if pin in part["pins"]}
        if "pin_roles" in row:
            part["pin_roles"] = {str(pin): role for pin, role in row["pin_roles"].items()}
        part["source_layers"] = [*part["source_layers"], layer]
        parts[ref] = part
    return parts


def write_reference(project, count, contract_path, enet_path, parts, notes=()):
    project = Path(project)
    contract_path = Path(contract_path).resolve()
    contract = read_json(contract_path)
    joint = read_json(JOINT)
    digest = source_record(contract_path)["sha256"]
    require(joint["status"] == "PASS" and
            joint["contract_sha256"][contract_path.relative_to(POWER).as_posix()] == digest,
            "联合校验状态或施工合同哈希不匹配")
    require(len(parts) == count, f"施工图对象应为 {count}，实际 {len(parts)}")
    for ref, data in parts.items():
        require(ref == data["ref"] and re.fullmatch(r"[A-Za-z][\w-]*", ref), f"非法位号 {ref}")
        require(data["pins"] and all(isinstance(pin, str) and pin and
                (net is None or isinstance(net, str) and net) for pin, net in data["pins"].items()),
                f"{ref} 引脚网表不完整")
        require(data["placement"] in ("board", "off-board", "schematic-only"), f"{ref} 装配分类错误")
    reference = {
        "contract_id": contract["contract_id"], "target_schematic_count": count,
        "inputs": {
            "enet": source_record(enet_path), "contract": source_record(contract_path),
            "joint": source_record(JOINT),
        },
        "counts_by_placement": dict(sorted(Counter(part["placement"] for part in parts.values()).items())),
        "notes": list(notes),
        "components": sorted(parts.values(), key=lambda row: row["ref"]),
    }
    write_json(project / f"reference_{count}_netlist.json", reference)
    print(f"Wrote {count} schematic objects to reference_{count}_netlist.json")
    return reference


def verify_cli(project, count):
    from verify import verify_circuit

    project = Path(project)
    parser = argparse.ArgumentParser(description="Compare Circuit JSON, schematic and BOM against the EDA contract overlay.")
    parser.add_argument("--reference", type=Path, default=project / f"reference_{count}_netlist.json")
    parser.add_argument("--exports", type=Path, default=project / "exports")
    options = parser.parse_args()
    result = verify_circuit(options.reference, options.exports, count)
    write_json(options.exports / "netlist_verify_report.json", result)
    pins = result["pin_net_comparison"]
    print(f"{result['status']}: {result['components']['observed']}/{count} objects, "
          f"{pins['equal']}/{pins['total']} pins, {pins['distinct_named_nets']} nets, "
          f"{len(result['findings'])} discrepancies.")
    raise SystemExit(result["status"] != "PASS")
