"""Check the sixth-board deliverables and prove the previously frozen files stayed intact."""

import contextlib
import hashlib
import io
import json
from pathlib import Path
import runpy

HERE = Path(__file__).resolve().parent
POWER = HERE.parents[1]
ROOT = POWER.parent
OUT = HERE / "充电MPPT交付校验.json"


def read(path):
    return json.loads(path.read_bytes())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_documents():
    # Reuse the exact repository rule in-process, adding the sixth module to
    # this invocation while leaving the existing command and all four boards intact.
    namespace = runpy.run_path(str(POWER / ".agents/tools/统一目录说明.py"))
    main = namespace["main"]
    main.__globals__["MODULES"] = (*namespace["MODULES"], "充电MPPT模块")
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        status = main()
    return status, json.loads(buf.getvalue())


def main():
    expected = read(HERE / "冻结四板SHA256基线.json")
    changed = [name for name, old in expected.items()
               if not (ROOT / name).is_file() or digest(ROOT / name) != old]
    projects = {ROOT / name.split("/CodeAsEDA_tscircuit/")[0] / "CodeAsEDA_tscircuit"
                for name in expected if "/CodeAsEDA_tscircuit/" in name}
    exempt = {"node_modules", "dist", ".cache", "__pycache__"}
    added = sorted(
        p.relative_to(ROOT).as_posix()
        for project in projects for p in project.rglob("*") if p.is_file()
        and not exempt.intersection(p.relative_to(project).parts)
        and p.relative_to(ROOT).as_posix() not in expected
    )
    docs_code, docs = check_documents()
    contract = read(HERE / "充电MPPT施工合同.json")
    sim = read(HERE / "充电MPPT与自持仿真核算.json")
    can = read(HERE / contract["known_schematic_scope"]["deliverable"])
    pins = can["pin_net_comparison"]
    cases = sim["conditions"]
    check_frozen = not changed and not added and len(projects) == 4
    check_can = can["status"] == "PASS" and can["components"]["observed"] == 10 and pins["equal"] == pins["total"] == 36
    check_sim = (
        sim["engine"]["version"] == "ngspice-47"
        and sim["script_sha256"] == digest(HERE / "充电MPPT与自持仿真核算.py")
        and cases["solar_3_channel"]["status"] == "PASS_MODEL"
        and cases["dark_to_sun_bus_zero"]["local_bias"] == "PASS_MODEL"
        and cases["recovery_interface_charge_window"]["status"] == "PASS_MODEL"
    )
    safe_scope = (
        contract["target_component_count"] is None and contract["release_for_power_connection"] is False
        and contract["release_for_purchase"] is False
        and not (HERE / "CodeAsEDA_tscircuit/exports/netlist_verify_report.json").exists()
    )
    passed = not docs_code and check_frozen and check_can and check_sim and safe_scope
    data = {
        "status": "PASS_PRELIMINARY_DELIVERY" if passed else "FAIL",
        "full_board_construction": contract["status"],
        "frozen_assets": {"status": "PASS" if check_frozen else "FAIL", "checks_sha256": len(expected),
                          "changed_or_missing": changed, "added_in_frozen_projects": added},
        "document_gate": docs,
        "can_interface_only": {"status": "PASS" if check_can else "FAIL",
                               "objects": can["components"]["observed"], "pins": pins["equal"]},
        "simulation_evidence": {"status": "PASS" if check_sim else "FAIL",
                                "scope": "source hashes and boundary-model checks only"},
        "scope_enforced": safe_scope,
    }
    OUT.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(data["status"], data["frozen_assets"], data["document_gate"]["status"], data["can_interface_only"])
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
