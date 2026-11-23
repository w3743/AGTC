"""Combine verified schematic BOMs and frozen vehicle accessory selections."""

import csv
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
POWER = HERE.parents[1]
BOARDS = {
    "BMS": ("BMS模块", ("BMS",), 301, "闭合候选_9S5P/BMS_9S5P施工合同.json"),
    "稳压": ("稳压模块", ("稳压",), 268, "闭合候选_P5Rev2/稳压P5Rev2施工合同.json"),
    "FOC": ("FOC驱动模块", ("FOC-A", "FOC-B"), 165, "闭合候选_164Rev2/FOC_164Rev2施工合同.json"),
    "水泵": ("水泵驱动模块", ("水泵",), 52, "闭合候选_隔离CAN/水泵闭合施工合同.json"),
}
FIELDS = ("LCSC_Code", "MPN", "Manufacturer", "Footprint", "Vehicle_Qty", "Boards_Refs",
          "Category", "LCSC_Order_Qty", "Procurement_Status")
C_CODE = re.compile(r"C[1-9]\d*")
WEACT_MODULE = re.compile(r"weact.{0,30}(?:stm32|cbu\d|core)|coreboard_weact|weact", re.I)


def read(path):
    return json.loads(path.read_bytes())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def merge_line_items():
    entries = {}
    identities = set()
    contracts = {}

    def add(board, ref, part, category, qty=1, status=""):
        identity = board, ref
        require(identity not in identities, f"采购重复位号 {board}/{ref}")
        identities.add(identity)
        code = part.get("lcsc") or ""
        require(not code or C_CODE.fullmatch(code), f"非法 C 码 {board}/{ref}: {code}")
        candidate = part.get("mpn") or ""
        mpn = candidate or part.get("value") or ref
        maker = part.get("manufacturer") or ""
        footprint = part.get("footprint") or ""
        if WEACT_MODULE.search(" ".join((candidate, maker, ref))):
            return
        key = ("lcsc", code) if code else ("mpn", candidate, maker, footprint) if candidate else ("ref", board, ref)
        row = entries.setdefault(key, {"mpn": mpn, "makers": set(), "fp": set(), "category": set(),
                                       "refs": defaultdict(list), "qty": 0, "order_qty": 0, "status": set()})
        require(not code or row["mpn"] == mpn, f"{code} 对应多个料号: {row['mpn']} / {mpn}")
        if maker:
            row["makers"].add(maker)
        if footprint:
            row["fp"].add(footprint)
        row["category"].add(category)
        row["refs"][board].append((ref, qty))
        row["qty"] += qty
        if status:
            row["status"].add(f"{board} {status}")
        if code and not status:
            row["order_qty"] += qty
        if not code:
            row["status"].add("无已冻结立创 C 码")

    for board, (folder, physical_boards, count, contract_rel) in BOARDS.items():
        directory = POWER / folder / "检查"
        contract_path = directory / contract_rel
        contracts[board] = read(contract_path)
        project = POWER / folder / "eda工程文件"
        exports = project / "exports"
        bom_path = exports / "bom.csv"
        reference_path = project / f"reference_{count}_netlist.json"
        report = read(exports / "netlist_verify_report.json")
        report_hashes = report["artifacts"]
        pins = report["pin_net_comparison"]
        require(report["status"] == "PASS" and report["components"]["observed"] == count
                and pins["equal"] == pins["total"], f"{board} 网表门禁未通过")
        require(report["inputs"]["contract"]["sha256"] == sha(contract_path)
                and report_hashes["bom"]["sha256"] == sha(bom_path)
                and report_hashes["reference"]["sha256"] == sha(reference_path)
                and report_hashes["circuit"]["sha256"] == sha(exports / "circuit.json"),
                f"{board} 合同或 BOM 证据已变化")
        ref_parts = {part["ref"]: part for part in read(reference_path)["components"]}
        with bom_path.open(encoding="utf-8-sig", newline="") as stream:
            bom = list(csv.DictReader(stream))
        require(len(bom) == count and {row["Reference"] for row in bom} == ref_parts.keys(),
                f"{board} 原理图 BOM 数量不符")
        for row in bom:
            ref = row["Reference"]
            # The BMS design explicitly retains this panel switch as a vehicle part.
            retained_switch = board == "BMS" and ref == "SW_RUN_STORE"
            if (row["DNP"] == "yes" or row.get("Placement") == "schematic-only"
                    or (row.get("In BOM") == "no" and not retained_switch)):
                continue
            placement = row.get("Placement", "board")
            part = {
                "mpn": row["MPN"], "lcsc": row["LCSC"],
                "manufacturer": row.get("Manufacturer", ref_parts[ref].get("manufacturer")),
                "footprint": row["Footprint"], "value": row.get("Value"),
            }
            for physical in physical_boards:
                status = ""
                if board == "BMS" and ref.startswith("NTC_CELL"):
                    status = "HOLD 探头装配与标定"
                if board == "FOC" and ref == "F_MAIN":
                    status = "板外件，按 70 VDC 型号另行采购"
                if board == "FOC" and ref == "F_HALL_5V":
                    status = "HOLD 首板短地动作与 5V_BUCK 存活实测，料号已冻结"
                category = "板外附件" if placement == "off-board" else "板内器件"
                if retained_switch:
                    category = "板外面板开关"
                if ref.startswith("C_BUS_BULK") and board == "稳压":
                    category = "板内母线电解"
                if ref.startswith("CBUS") and board == "FOC":
                    category = "板外母线电解"
                add(physical, ref, part, category, status=status)

    procurement = read(HERE / "降本采购与SMT建议.json")
    offboard = procurement["offboard"]
    add("BMS", "R_PRE", {**offboard["r_pre"], "mpn": "RX24 300W 8.2Ω ±5%"},
        "板外功率电阻", status="HOLD 完整订货料号、报价与安装热资格")
    add("BMS", "F1", offboard["f1"], "板外保险", status="HOLD 58 VDC 订货后缀与分断资格")
    taps = offboard["tap_fuse"]
    require(len(taps["refs_and_nets"]) == taps["quantity"] == 9, "BMS 抽头保险数量")
    for ref in taps["refs_and_nets"]:
        add("BMS", ref, taps, "板外保险", status="HOLD 抽头开断与线束协调")
    carrier = offboard["tap_fuse_holder"]
    add("BMS", "F_CELL_HOLDER", carrier["selected_assembly"], "板外保险附件",
        carrier["quantity"], "HOLD 装配/端子与报价")
    for side in ("female", "male"):
        maintenance = offboard["maintenance"]
        add("BMS", f"AS150U_{side.upper()}", {
            "mpn": maintenance[side], "lcsc": maintenance[f"{side}_lcsc"],
            "manufacturer": maintenance["manufacturer"],
        }, "板外防打火接插件")

    sampling = contracts["BMS"]["cell_service_cost_down_item_3a"]["candidate"]
    for key, ref in [("harness_housing", "J_CELL_BMS"), ("harness_contacts", "J_CELL_BMS_CONTACT")]:
        part = {**sampling[key], "manufacturer": sampling["manufacturer"]}
        add("BMS", ref, part, "板外采样线束端", part["quantity"], "HOLD 抽头开断与线束协调")

    mating = procurement["connector_standard"]["medium_power_5p08"]
    mate = next(value for value in mating.values() if isinstance(value, dict) and value.get("lcsc") == "C581323")
    for board, ref in (("水泵", "J_BUS_MATE"), ("水泵", "J_MOTOR_MATE"), ("BMS", "J_R_PRE_MATE")):
        add(board, ref, mate, "板外中电流插头")

    inductor = offboard["l_main_a_b"]
    for ref in inductor["assembly"]["phases"]:
        add("稳压", ref, inductor, "板外功率电感")
    for part in contracts["稳压"]["temperature_sensors"]:
        add("稳压", part["ref"], part, "板外温度探头", part["quantity"])
    for ref in ("F_BUS", "F_SC"):
        add("稳压", ref, {"mpn": "AMG-80 70VDC", "manufacturer": "Eaton"}, "板外保险",
            status="HOLD 精确后缀、70 VDC/2.5 kA 和采购证据")
    add("稳压", "R_DUMP_EXT", {"value": "0.82Ω ±5%"}, "板外功率电阻",
        status="HOLD 完整料号、功率与安装热资格")

    supercap = contracts["稳压"]["supercap_tap_fuses"]
    for fuse in supercap["fuses"]:
        add("稳压", fuse["ref"], supercap["part"], "板外保险",
            status="HOLD 预算、实际开断时间与装配")
    return entries


def main():
    entries = merge_line_items()
    output = HERE / "整车立创一键采购BOM.csv"
    with output.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        for key, row in sorted(entries.items()):
            writer.writerow({
                "LCSC_Code": key[1] if key[0] == "lcsc" else "",
                "MPN": row["mpn"], "Manufacturer": " | ".join(sorted(row["makers"])),
                "Footprint": " | ".join(sorted(row["fp"])), "Vehicle_Qty": row["qty"],
                "Boards_Refs": " | ".join(
                    f"{board}:" + ",".join(ref if qty == 1 else f"{ref}×{qty}" for ref, qty in sorted(refs))
                    for board, refs in sorted(row["refs"].items())),
                "Category": " | ".join(sorted(row["category"])), "LCSC_Order_Qty": row["order_qty"],
                "Procurement_Status": " | ".join(sorted(row["status"])) or "目录 C 码已冻结",
            })
    print(f"{len(entries)} vehicle BOM rows; {sum(row['qty'] for row in entries.values())} items; "
          f"{sum(row['order_qty'] > 0 for row in entries.values())} LCSC order rows.")


if __name__ == "__main__":
    main()
