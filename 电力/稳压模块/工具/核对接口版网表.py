"""核对板外仅接口改版：剩余实例逐针守恒，新增保险接口不跨接两网。"""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("export", type=Path, help="官方DRC导出JSON或EasyEDA ENET")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    raw = args.export.read_bytes()
    exported = json.loads(raw.decode("utf-8-sig"))
    envelope = exported if "exports" in exported else None
    netlist = json.loads(exported["exports"]["netlist"]) if envelope else exported
    baseline = json.loads((ROOT / "原理图准备/板外仅接口迁移基线.json").read_text(encoding="utf-8"))
    contract = json.loads((ROOT / "原理图准备/最终逐针连接合同.json").read_text(encoding="utf-8"))
    actual, duplicates = {}, []
    for component in netlist["components"].values():
        ref = component["props"]["Designator"]
        if ref in actual:
            duplicates.append(ref)
        actual[ref] = {str(pin): row.get("net", "") for pin, row in component.get("pinInfoMap", {}).items()}
    expected = {ref: pins for ref, pins in baseline["component_pin_nets"].items()
                if ref not in baseline["removed_native_refs"]}
    expected.update(baseline["added_interface_pin_nets"])
    target = {c["ref"]: {str(p["number"]): p["net"] if p["connection"] == "NET" else ""
                         for p in c["pins"]} for c in contract["components"]}
    differences = []
    for ref in sorted(set(expected) | set(actual)):
        if expected.get(ref) != actual.get(ref):
            differences.append({"ref": ref, "expected": expected.get(ref), "actual": actual.get(ref)})
    drc = envelope.get("drc") if envelope else None
    drc_present = isinstance(drc, list) if envelope else None
    report = {
        "schema": "OFFBOARD_INTERFACE_NETLIST_CHECK_V1",
        "verification_scope": "NETLIST_AND_OFFICIAL_DRC" if envelope else "NETLIST_ONLY",
        "source": str(args.export),
        "source_sha256": hashlib.sha256(raw).hexdigest(),
        "component_count": len(actual),
        "pin_count": sum(map(len, actual.values())),
        "duplicate_refs": duplicates,
        "pin_net_differences": differences,
        "contract_matches_migration": target == expected,
        "official_drc": drc,
        "official_drc_present": drc_present,
        "pass": not duplicates and not differences and target == expected and (drc_present and drc == [] if envelope else True),
        "limits": "仅核对接口迁移后的针网与导出DRC；实际NC标志另由官方逐针验收核对，不代表实物或PCB验收。",
    }
    text = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k not in ["limits", "official_drc"]}, ensure_ascii=True))
    raise SystemExit(0 if report["pass"] else 1)


if __name__ == "__main__":
    main()
