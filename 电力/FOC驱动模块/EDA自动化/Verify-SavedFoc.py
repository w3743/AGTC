"""Run official checks in a new headless session on a copy of the saved project."""
import argparse
import base64
import csv
import hashlib
import io
import json
import shutil
import subprocess
import tempfile
import time
from collections import Counter
from contextlib import contextmanager
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NATIVE = ROOT / "eda工程文件/FOC_G431_Driver"
DELIVERY = ROOT / "eda工程文件/导出/当前交付"
CLI = Path("C:/Program Files/lceda-pro/lceda-pro.exe")
PAGE = "90aa6c46493624cf"


def call(*args):
    # The hidden renderer can outlive the CLI command. Files avoid waiting
    # on inherited stdout or stderr pipe handles after the command has exited.
    with tempfile.TemporaryFile() as stdout, tempfile.TemporaryFile() as stderr:
        process = subprocess.run(
            [CLI, *args], stdout=stdout, stderr=stderr, timeout=120,
            creationflags=subprocess.CREATE_NO_WINDOW,
        )
        stdout.seek(0)
        raw = stdout.read()
        if not raw:
            stderr.seek(0)
            raise ValueError(f"No CLI result ({process.returncode}): {stderr.read(1000)!r}")
    result = json.loads(raw.decode("utf-8-sig"))
    if not result["ok"]:
        raise RuntimeError(result)
    return result["value"]


def hashes(folder):
    return {
        file.relative_to(folder).as_posix(): hashlib.sha256(file.read_bytes()).hexdigest()
        for file in folder.rglob("*") if file.is_file()
    }


def pins(data):
    return {
        component["props"]["Designator"]: {
            pin: info.get("net") or "" for pin, info in component["pinInfoMap"].items()
        } for component in data["components"].values()
    }


def parts(text):
    text = text.lstrip("\ufeff")
    separator = "\t" if "\t" in text.splitlines()[0] else ","
    rows = csv.DictReader(io.StringIO(text), delimiter=separator)
    fields = rows.fieldnames or []
    ref_field = next((x for x in ("Designator", "位号") if x in fields), None)
    mpn_field = next((x for x in ("Manufacturer Part", "制造商型号", "制造商料号") if x in fields), None)
    if not (ref_field and mpn_field):
        raise ValueError(fields)
    items = {}
    for row in rows:
        for ref in row[ref_field].split(","):
            ref = ref.strip()
            if ref in items:
                raise ValueError(f"Duplicate designator {ref}")
            items[ref] = {"mpn": row[mpn_field], "supplierPart": row.get("Supplier Part", "")}
    return items


def compare(before, after):
    return {ref: [before.get(ref), after.get(ref)]
            for ref in sorted(before.keys() | after.keys()) if before.get(ref) != after.get(ref)}


@contextmanager
def headless(target):
    sessions = {s["sessionId"] for s in call("session", "list")}
    opened = call("open", "--path", str(target), "--headless", "true")
    session = opened["sessionId"]
    assert session not in sessions and opened["headless"] and Path(opened["path"]) == target

    def invoke(code, timeout="90000"):
        return call("invoke", "--session", session, "--ext-uuid", "eda",
                    "--timeout", timeout, "--code", code)

    try:
        for attempt in range(8):
            try:
                binding = invoke(f"""
const pages = await eda.dmt_Schematic.getAllSchematicPagesInfo();
let opened, failure;
try {{
    if (pages?.some(page => page.uuid === '{PAGE}')) opened = await eda.dmt_EditorControl.openDocument('{PAGE}');
}} catch (error) {{ failure = String(error); }}
const page = await eda.dmt_Schematic.getCurrentSchematicPageInfo();
const project = await eda.dmt_Project.getCurrentProjectInfo();
return {{opened, failure, page: page?.uuid, project: project?.uuid, name: project?.name, pages}};
""", timeout="5000")
            except RuntimeError as error:
                if error.args[0].get("error", {}).get("code") != "REQUEST_TIMEOUT":
                    raise
                binding = {"failure": "page binding timed out"}
            if binding.get("page") == PAGE and binding.get("name", "").lstrip("/") in ("FOC驱动模块", NATIVE.name):
                break
            time.sleep(1)
        else:
            raise RuntimeError(f"FOC copy did not open the target page: {opened}, {binding}")
        yield invoke, binding
    finally:
        call("session", "close", "--session", session, "--destroy")
    assert sessions <= {s["sessionId"] for s in call("session", "list")}


def exports(invoke):
    drc = invoke("return await eda.sch_Drc.check(true, false, true);")
    net = invoke("""const file = await eda.sch_ManufactureData.getNetlistFile('FOC_G431_P1','EasyEDA');
return file ? await file.text() : null;""")
    bom = invoke("""const file = await eda.sch_ManufactureData.getBomFile('FOC_G431_BOM','csv');
return file ? await file.text() : null;""")
    return drc, net, bom


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export-dir", type=Path, help="Stage fresh PDF, ENET and CSV from the verified session")
    parser.add_argument("--layout-trial", action="store_true", help="Apply P1 label cleanup to the temporary copy only")
    options = parser.parse_args()
    if options.layout_trial and not options.export_dir:
        parser.error("--layout-trial requires --export-dir")
    before = hashes(NATIVE)
    with tempfile.TemporaryDirectory(prefix="foc_readback_") as tmp:
        copy = Path(tmp) / NATIVE.name
        shutil.copytree(NATIVE, copy)
        assert before == hashes(copy)
        target_project = copy / (NATIVE.name + ".eprj3")
        edits = None
        if options.layout_trial:
            with headless(target_project) as (invoke, _):
                edits = invoke((ROOT / "EDA自动化/Improve-P1-Labels.js").read_text(encoding="utf-8-sig"))
        with headless(target_project) as (invoke, binding):
            drc, net, bom = exports(invoke)
            if options.export_dir:
                pages = [page["uuid"] for page in binding["pages"]]
                if pages != [PAGE]:
                    raise RuntimeError(f"Unexpected schematic pages: {pages}")
                pdf = invoke((ROOT / "EDA自动化/Export-CurrentPdf.js").read_text(encoding="utf-8-sig"))
        copy_hashes = hashes(copy)
        copy_unchanged = before == copy_hashes
        if options.layout_trial:
            staged_files = {name: (copy / name).read_bytes() for name in copy_hashes
                            if copy_hashes[name] != before.get(name)}
    before_net = pins(json.loads((DELIVERY / "FOC_G431_P1.enet").read_text(encoding="utf-8-sig")))
    after_net = pins(json.loads(net.lstrip("\ufeff")))
    before_bom = parts((DELIVERY / "FOC_G431_BOM.csv").read_text(encoding="utf-8-sig"))
    after_bom = parts(bom)
    current = json.loads(net.lstrip("\ufeff"))
    board_refs = {c["props"]["Designator"] for c in current["components"].values()
                  if c["props"].get("Convert to PCB") == "yes"}
    board = sorted(board_refs & after_bom.keys())
    external = sorted(after_bom.keys() - board_refs)
    contract = json.loads((ROOT / "原理图准备/当前原理图连接.json").read_text(encoding="utf-8-sig"))
    expected = {c["ref"]: {pin: value or "" for pin, value in c["pins"].items()}
                for c in contract["components"]}
    summary = {
        "edaVersion": "4.1.93",
        "method": "Official headless readback from a byte-identical copy of the saved native folder",
        "binding": binding,
        "nativeSha256": before,
        "copyUnchanged": copy_unchanged,
        "originalUnchanged": before == hashes(NATIVE),
        "existingSessionsStillPresent": True,
        "drcCounts": dict(Counter(row["type"] for row in drc)),
        "netlist": {"components": len(after_net), "pins": sum(map(len, after_net.values())),
                    "differences": compare(before_net, after_net), "contractDifferences": compare(expected, after_net)},
        "bom": {
            "populatedRefs": len(after_bom), "differences": compare(before_bom, after_bom),
            "boardCount": len(board), "externalCount": len(external), "externalRefs": external,
            "cadMpnLabels": len({row["mpn"] for row in after_bom.values()}),
        },
        "drc": drc,
    }
    if options.layout_trial:
        summary["layoutTrial"] = edits
        summary["reopenedTrialSha256"] = copy_hashes
        allowed = {NATIVE.name + ".eprj3", "sch/Schematic1/P1.esch2"}
        if (before.keys() != copy_hashes.keys() or not staged_files.keys() <= allowed
                or "sch/Schematic1/P1.esch2" not in staged_files):
            raise RuntimeError(f"Unexpected native change in visual trial: {list(staged_files)}")
        output = options.export_dir / "trial.json"
        options.export_dir.mkdir(parents=True, exist_ok=True)
    else:
        output = ROOT / "检查/当前保存源_独立官方复跑.json"
    output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print({key: value for key, value in summary.items() if key not in ("drc", "nativeSha256")})
    if (not summary["originalUnchanged"] or (not copy_unchanged and not options.layout_trial)
            or summary["netlist"]["differences"] or summary["netlist"]["contractDifferences"]
            or summary["bom"]["differences"] or any(row["type"] == "error" for row in drc)):
        raise SystemExit(1)
    if options.export_dir:
        target = options.export_dir
        target.mkdir(parents=True, exist_ok=True)
        pdf_bytes = base64.b64decode(pdf["base64"], validate=True)
        assert pdf_bytes.startswith(b"%PDF-") and len(pdf_bytes) == pdf["bytes"]
        artifacts = {"FOC_G431_单页原理图.pdf": pdf_bytes,
                     "FOC_G431_P1.enet": net.encode("utf-8"), "FOC_G431_BOM.csv": bom.encode("utf-8")}
        summary["freshOfficialExports"] = {
            name: {"sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}
            for name, data in artifacts.items()
        }
        for name, data in artifacts.items():
            (target / name).write_bytes(data)
        if options.layout_trial:
            for name, data in staged_files.items():
                staged_path = target / "native" / name
                staged_path.parent.mkdir(parents=True, exist_ok=True)
                staged_path.write_bytes(data)
        output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
