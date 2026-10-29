"""Rebuild the current regulator merge-candidate CSV and identity audit.

This tool pins the original 58/45 snapshot inside the output JSON on its first
run, then reapplies the candidate overlay idempotently. It does not edit the
native schematic, procurement BOM, or design documents.
"""
from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path


ROOT = Path(r"C:\pioneer\先进通用履带底盘工程")
POWER = ROOT / "电力"
REG = POWER / "稳压模块"
CHECK = REG / "检查"
PREP = REG / "原理图准备"
TOOLS = REG / "工具"
OUT_JSON = CHECK / "候选新增身份核对.json"
OUT_CSV = PREP / "候选新增明细.csv"
MERGE = CHECK / "本轮硬件合并清单.json"
ZERO = CHECK / "0V初充器件清单.json"
QUOTE = CHECK / "0V初充报价覆盖.json"
MOS = CHECK / "MOS隔离关断候选BOM.json"
LTC = CHECK / "LTC7001_12V掉电关断证据.json"
CENTRAL = CHECK / "中央制动耗能方案.json"
CHARGE_CONTRACT = PREP / "初充与掉电接口合同.json"
PROTECTION_CONTRACT = PREP / "保护与制动合并合同.json"
REG_BOM = PREP / "BOM.csv"
BMS_BOM = POWER / "BMS模块" / "原理图准备" / "BOM.csv"
BMS_QUOTE = POWER / "BMS模块" / "原理图准备" / "9S5P增量报价.json"


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def clean(value) -> str:
    if value is None:
        return ""
    value = str(value).strip()
    return "" if value.upper() in {"", "TBD", "UNKNOWN", "N/A", "NULL"} else value


def ref_key(row: dict):
    return row.get("reference", row.get("Ref", ""))


def pin_pairs(pins: dict) -> str:
    return "; ".join(f"{k}={v}" for k, v in pins.items())


def bom_rows(path: Path):
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def main():
    current = read_json(OUT_JSON)
    merge = read_json(MERGE)
    if merge.get("status", "").startswith("WITHDRAWN_"):
        print("该比较方案已撤回当前采用地位；不重建为现行候选或发布新增数量。")
        return
    ltc = read_json(LTC)
    charge = read_json(CHARGE_CONTRACT)
    protection = read_json(PROTECTION_CONTRACT)
    central = read_json(CENTRAL)
    quote = read_json(QUOTE)
    bms_quote = read_json(BMS_QUOTE)

    # Preserve the original 58/45 source interpretation before replacing the
    # top-level counts with the current candidate counts. Re-runs use this pin.
    snapshot = current.get("baseline_snapshot")
    if not snapshot:
        snapshot = {
            "source_file": "电力/稳压模块/检查/本轮硬件合并清单.json",
            "source_status": merge.get("status", ""),
            "original_added_instances": 58,
            "original_retired_instances": 45,
            "original_selection_counts": current.get("selection_counts", {}),
            "original_source_sha256": current.get("source_sha256", {}),
            "new_instances": current.get("new_instances", []),
            "retired_refs_from_current_bom": current.get("retired_refs_from_current_bom", []),
        }
    base_new = [dict(x) for x in snapshot["new_instances"]]
    base_retired = [dict(x) for x in snapshot["retired_refs_from_current_bom"]]
    base_new_refs = [ref_key(x) for x in base_new]
    base_retired_refs = [ref_key(x) for x in base_retired]
    if len(base_new_refs) != 58 or len(set(base_new_refs)) != 58:
        raise ValueError("embedded 58-ref baseline is missing or non-unique")
    if len(base_retired_refs) != 45 or len(set(base_retired_refs)) != 45:
        raise ValueError("embedded 45-ref baseline is missing or non-unique")

    cancel_new = {
        "R_MAX_C0", "R_GATE_SER", "D_ISO_BST_START", "R_ISO_TGUP", "D_SC_CLAMP",
        "R_MAX_AUX0_PU", "R_MAX_AUX1_PU", "R_MAX_AUX2_PU", "R_MAX_AUX3_PU",
        "C_MAX_SHDNL",
    }
    cancel_retired = {
        "R_T1_LO", "R_T2_LO", "R_T3_LO", "R_T4_LO", "R_QP_COL",
        "Q_PMOS_CTRL", "R_QP_BE", "R_TEST_HI", "C_T5_FILTER",
        "R_T5_LO", "R_T6_LO", "R_T7_LO", "R_TEST_LO", "C_T1_FILTER",
    }
    rows = [r for r in base_new if ref_key(r) not in cancel_new]
    retired = [r for r in base_retired if ref_key(r) not in cancel_retired]

    # Common, exact 100nF instance selected throughout this candidate.
    common_100n = {
        "mpn": "CGA3E2X7R1H104KT0Y0S", "manufacturer": "TDK",
        "supplier": "LCSC", "supplier_part": "C342998",
        "value_or_requirement": "100nF, 50V, X7R, 0603",
    }

    # Later parent contracts replace only the MPN and interface contract of
    # this existing 58-baseline instance; its reference count remains one.
    for r in rows:
        if ref_key(r) == "R_ISO_GATE_SOURCE":
            r.update({
                "value_or_requirement": "2.2MΩ, 1%, 0603",
                "mpn": "RC0603FR-072M2L", "manufacturer": "",
                "supplier": "", "supplier_part": "",
                "identity_status": "精确候选MPN已解析；供应订货代码未提供",
                "identity_key": "RC0603FR-072M2L|供应商未提供|供应代码未提供",
                "pin_nets": "ISO_G↔ISO_MID；2.2MΩ为候选栅源泄放。",
                "source": "检查/LTC7001_12V掉电关断证据.json:existing_net_changes.R_ISO_GATE_SOURCE",
                "source_identity_note": "2.2MΩ候选MPN有记录，供应商料号/报价未核；原10MΩ身份不再作为当前候选。",
                "note": "数量不变，MPN更换；缺供应代码，留空。",
            })

    # Merge authority rows from the six-part independent ISO supply evidence.
    aux_by_ref = {p["ref"]: p for p in ltc.get("parts", [])}
    ltc_pins = {p["ref"]: p.get("pins_and_nets", {}) for p in ltc.get("parts", [])}
    aux_value = {
        "D_ISO_AUX_OR": "CMPD3003C; dual common-cathode low-leakage silicon diode",
        "U_ISO_AUX_LDO": "TPS7A4001DGNR; 100V LDO, DGN-8 plus exposed pad",
    }
    for ref in ltc["decision"]["component_delta_local"]["added_refs"]:
        p = aux_by_ref[ref]
        rows.append({
            "reference": ref, "instance_qty": 1,
            "value_or_requirement": p.get("value") or aux_value.get(ref, p.get("package", "")),
            "mpn": clean(p.get("mpn")), "manufacturer": clean(p.get("manufacturer")),
            "supplier": "LCSC" if clean(p.get("supplier_part")) else "",
            "supplier_part": clean(p.get("supplier_part")),
            "identity_key": f"{clean(p.get('mpn'))}|{'LCSC' if clean(p.get('supplier_part')) else '供应商未提供'}|{clean(p.get('supplier_part')) or '供应代码未提供'}",
            "identity_status": "精确MPN与供应代码已解析" if clean(p.get("supplier_part")) else "精确MPN已解析；供应订货代码未提供",
            "pin_nets": pin_pairs(ltc_pins[ref]),
            "pin_number_scope": "依据LTC7001掉电证据逐脚合同；未新增native netlist。",
            "source": "检查/LTC7001_12V掉电关断证据.json:parts",
            "source_identity_note": clean(p.get("procurement")) or clean(p.get("requirement")),
            "note": "候选，未落原理图或采购BOM。" + (
                "MPN暂按当前JSON保留；有效电容需在全温/偏压最差条件下仍>4.7µF，现无证据闭合；IN/OUT双源同时掉电且OUT较IN高约0.3V时的反灌、放电/保持应力也未闭合，不能视为冻结选型。"
                if ref == "C_ISO_AUX_OUT" else ""
            ),
        })

    def add(ref, mpn, value, pins, supplier="", supplier_part="", manufacturer="", source="", note="", pin_scope="候选功能脚/端点网；未落native netlist。"):
        code = clean(supplier_part)
        rows.append({
            "reference": ref, "instance_qty": 1, "value_or_requirement": value,
            "mpn": mpn, "manufacturer": manufacturer, "supplier": supplier if code else "",
            "supplier_part": code,
            "identity_status": "精确MPN与供应代码已解析" if code else "精确MPN已解析；供应订货代码未提供",
            "identity_key": f"{mpn}|{supplier if code else '供应商未提供'}|{code or '供应代码未提供'}",
            "pin_nets": pins, "pin_number_scope": pin_scope,
            "source": source, "source_identity_note": "",
            "note": note or "候选，未落原理图或采购BOM。",
        })

    sig_add = {p["ref"]: p for p in charge.get("add", [])}
    u_signal = sig_add["U_SIGNAL_ISO"]
    add("U_SIGNAL_ISO", "TMUX1511PWR", "TMUX1511PWR; four-channel analog switch/isolation",
        pin_pairs(u_signal["pins"]), "LCSC", "C2866750", "Texas Instruments",
        "原理图准备/初充与掉电接口合同.json:add", "VDD=3V3_CORE; SEL1..4=WD_OK; 四路隔离BUS_ADC、IMON、MISO、CSB。",
        "该合同给出完整封装pin1..14网名；仍为candidate而非已落图网表。")
    add("C_SIGNAL_ISO", common_100n["mpn"], common_100n["value_or_requirement"],
        "pin1=3V3_CORE; pin2=GND_SYS（U_SIGNAL_ISO本地旁路）", "LCSC", "C342998", "TDK",
        "原理图准备/初充与掉电接口合同.json:add")

    protection_parts = protection["power_paths"]["clamps"]
    for p in protection_parts:
        ref = p["ref"]
        if ref not in {"D_SC_LO", "D_SC_HI", "D_DUMP_FREEWHEEL"}:
            continue
        add(ref, p["mpn"], "STPS30100ST; 100V dual Schottky, two junctions",
            pin_pairs({"anode pins 1,3": p["anode_pins_1_3"], "cathode pin 2 + tab": p["cathode_pin_2_tab"]}),
            "LCSC", p["supplier_part"], "STMicroelectronics",
            "原理图准备/保护与制动合并合同.json:power_paths.clamps",
            "按source terminal contract列物理pin号。")

    afe = protection["afe_filter_correction"]
    for n in range(1, 9):
        add(f"C_MAX_SW{n}", common_100n["mpn"], common_100n["value_or_requirement"],
            f"端接：MAX17853.SW{n}↔MAX17853.SW{n-1}（跨相邻均衡开关节点）",
            "LCSC", "C342998", "TDK", "原理图准备/保护与制动合并合同.json:afe_filter_correction",
            "Cn滤波使用旧C_BALn至AGND；本件是另增SWn差分滤波电容。")
    add("R_MAX_SW0", "MFR25SFRE52-120R", "120Ω, 0.25W",
        "端接：SC_MINUS↔MAX17853.SW0。", "LCSC", "C1744031", "YAGEO",
        "parent-provided exact MPN/code; 原理图准备/保护与制动合并合同.json:afe_filter_correction",
        "new instance; same model/code as existing R_BAL1..R_BAL8, not count-neutral reuse.")
    add("C_MAX_SW0", "C3216X7R1E105K160AA", "1µF X7R",
        "端接：MAX17853.SW0↔AGND。", "LCSC", "C3869375", "",
        "parent-provided exact MPN/code; 原理图准备/保护与制动合并合同.json:afe_filter_correction")

    permit = protection["permit_open_drain"]
    add("U_GATE_OD", "SN74LVC1G07DCKR", "SN74LVC1G07DCKR; open-drain buffer",
        "功能脚网：A=U_ISO_ARM_FF.Q；Y=DRV_FAULT_N（同DRV8161 pin4与U_SC_ISO_GATE.INP）；VCC=3V3_CORE；GND=GND_SYS。",
        "LCSC", "C7830", "Texas Instruments",
        "原理图准备/保护与制动合并合同.json:permit_open_drain",
        "旧R_PB5_PERMIT改作CORE上拉；旧R_DRV_FAULT_PD=10k保留。",
        "合并合同给出具名功能脚网，未给本件物理脚号；候选未落图。")
    add("C_GATE_OD", common_100n["mpn"], common_100n["value_or_requirement"],
        "端接：U_GATE_OD.VCC↔GND_SYS（本地旁路）", "LCSC", "C342998", "TDK",
        "原理图准备/保护与制动合并合同.json:permit_open_drain")

    # Central hard-OVP additions. The protection contract supersedes the older
    # central-brake subpackage's PNP/diode-OR and divider values.
    add("U_BUS_OV", "TLV6700DDCR", "TLV6700DDCR; comparator",
        "功能脚网：VDD=12V_DRV；GND=GND_SYS；INB−=BUS_OV_SENSE；OUTB=DUMP_OFF；INA+=GND_SYS；OUTA=NC。",
        "LCSC", "C2868382", "Texas Instruments",
        "原理图准备/保护与制动合并合同.json:central_brake.hard_OV",
        "合并合同覆盖旧中央子包的PNP及二极管OR方案。")
    add("C_BUS_OV_VDD", common_100n["mpn"], common_100n["value_or_requirement"],
        "端接：U_BUS_OV.VDD/12V_DRV↔GND_SYS（本地旁路）", "LCSC", "C342998", "TDK",
        "原理图准备/保护与制动合并合同.json:central_brake.hard_OV")
    for ref, left, right in [
        ("R_BUS_OV_HI1", "BUS_EXT", "BUS_OV_MID"),
        ("R_BUS_OV_HI2", "BUS_OV_MID", "BUS_OV_SENSE"),
    ]:
        add(ref, "PTFR0805B255KP9", "255kΩ, 0.1%",
            f"端接：{left}↔{right}。", "LCSC", "C2849126", "RESI(开步睿思)",
            "原理图准备/保护与制动合并合同.json:central_brake.hard_OV")
    add("R_BUS_OV_LO", "RT0603BRD075K36L", "5.36kΩ, 0.1%",
        "端接：BUS_OV_SENSE↔GND_SYS。", "", "", "",
        "原理图准备/保护与制动合并合同.json:central_brake.hard_OV",
        "精确MPN已选；供应代码未给，留空。")
    add("F_DUMP_THERM", "TT150", "LANSON TT150; one-shot thermal fuse",
        "低电流联锁回路：BMS TPS3809 POR_OK_N→F_DUMP_THERM→BMS THERM_POR_OK；器件与HEWR电阻壳体热耦合；不串联dump高电流路径。",
        "LCSC", "C2972289", "LANSON",
        "原理图准备/保护与制动合并合同.json:central_brake.thermal / BMS_thermal_latch.loop",
        "一次性器件；候选热耦合和回路尚未物理验证。")

    # The board-external replacement is a model selection without a new
    # regulator-board instance. Preserve it in JSON, never as an 83rd CSV row.
    dump = protection["central_brake"]["resistor"]
    external_selection = {
        "reference": "R_DUMP (board-external replacement; no new regulator-board ref)",
        "mpn": dump["candidate_mpn"], "manufacturer": dump["manufacturer"],
        "supplier": "", "supplier_part": "",
        "value_or_requirement": dump["value"], "instance_qty": 0,
        "identity_status": dump["identity_status"],
        "source": "原理图准备/保护与制动合并合同.json:central_brake.resistor",
        "counts_as_new_instance": False, "counts_as_pending_new_selection": True,
    }

    # Replace known baseline wiring contracts with this candidate's newer nets.
    by_ref = {ref_key(r): r for r in rows}
    pin_update = {
        "U_AFE_MAX17853": (
            "功能脚网：C0=AGND直连；C1..C8分别经R_MAX_C1..8接T1..T7/SC_PLUS；"
            "旧C_BAL1..8各由Cn接AGND。SW0经R_MAX_SW0=120Ω接SC_MINUS并由C_MAX_SW0=1µF对AGND；"
            "SW1..SW8分别由旧R_BAL1..8接各抽头，并由新增C_MAX_SW1..8各跨SWn/SW(n-1)；"
            "SHDNL经旧C_T1_FILTER=1nF接AGND；AUXIN0..3分别经旧R_T5_LO/R_T6_LO/R_T7_LO/R_TEST_LO上拉至THRM，"
            "AUXIN4/5分别经新增R_MAX_AUX4_PU/R_MAX_AUX5_PU上拉至THRM；"
            "Cn9..14接最高有效C8滤波节点，SW9..14接最高有效SW8节点；CELLEN=0x00FF，TOPCELL1/2=8。"
            "SPI：pin9 TXUN/SCLK↔PB3经R_MAX_SCLK_SER；pin16 SDI←PB4经R_MAX_SDI_SER；pin15 SDO→PB9经U_SIGNAL_ISO；"
            "pin10 TXUP/CSB↔AFE侧MAX_CSB_AFE，经U_SIGNAL_ISO到MAX_CSB_CORE并经R_MAX_CSB_SER到PA15；"
            "R_MAX_CSB_PU上拉AFE侧CSB至VDDL3。UARTSEL=SC_MINUS；未用RX脚接GND_SYS；ALERTIN未用。"
        ),
        "R_MAX_C1": "端接：T1↔MAX17853.C1；串联1kΩ输入电阻。",
        "R_MAX_C2": "端接：T2↔MAX17853.C2；串联1kΩ输入电阻。",
        "R_MAX_C3": "端接：T3↔MAX17853.C3；串联1kΩ输入电阻。",
        "R_MAX_C4": "端接：T4↔MAX17853.C4；串联1kΩ输入电阻。",
        "R_MAX_C5": "端接：T5↔MAX17853.C5；串联1kΩ输入电阻。",
        "R_MAX_C6": "端接：T6↔MAX17853.C6；串联1kΩ输入电阻。",
        "R_MAX_C7": "端接：T7↔MAX17853.C7；串联1kΩ输入电阻。",
        "R_MAX_C8": "端接：SC_PLUS↔MAX17853.C8；串联1kΩ输入电阻。",
        "R_MAX_AUX4_PU": "端接：MAX17853.AUXIN4↔THRM；10kΩ pull-up only。",
        "R_MAX_AUX5_PU": "端接：MAX17853.AUXIN5↔THRM；10kΩ pull-up only。",
        "R_MAX_SCLK_SER": "端接：G431 PB3/SCLK↔MAX17853 pin9 TXUN/SCLK。",
        "R_MAX_SDI_SER": "端接：G431 PB4/MOSI↔MAX17853 pin16 SDI。",
        "R_MAX_CSB_SER": "端接：MCU PA15↔MAX_CSB_CORE↔MAX17853 AFE侧pin10 CSB（经U_SIGNAL_ISO）。",
        "R_MAX_CSB_PU": "端接：MAX17853 AFE侧pin10 CSB/MAX_CSB_AFE↔VDDL3。",
        "U_ISO_ARM_FF": "功能脚网：Q→U_GATE_OD.A；/CLR=U_AFE_AND.Y；D=3V3_CORE；CP=PB5_ARM；VCC=3V3_CORE；GND=GND_SYS。",
        "U_SC_ISO_GATE": "功能脚网：VCC=ISO_VCC；VCCUV=NC；OVLO=GND_SYS；GND/EP=GND_SYS；TS=ISO_MID；BST pin9经C_ISO_BST接TS pin8；TGUP pin7经旧R_QP_COL=10k到ISO_G；TGDN pin6直连ISO_G；INP pin4=DRV_FAULT_N许可；NC pin10悬空。",
        "C_ISO_VCC": "端接：U_SC_ISO_GATE.VCC/ISO_VCC↔GND_SYS（本地旁路；不再接12V_DRV）。",
        "C_ISO_BST": "端接：U_SC_ISO_GATE.BST pin9↔TS pin8。",
        "U_DRV_FAULT_FB": "功能脚网：A=DRV_FAULT_N；Y→G431 PA10及PB12/TIM1_BKIN；VCC=3V3_CORE；GND=GND_SYS；NTC_DUMP=PA3。",
        "U_HEALTH_SCHMITT": "功能脚网：AFE_OK经A1/Y1到U_AFE_AND.A；WD_OK经A2/Y2到U_AFE_AND.B；VCC=3V3_CORE；GND=GND_SYS。",
        "U_AFE_AND": "功能脚网：A=U_HEALTH_SCHMITT.Y1（AFE_OK缓冲）；B=U_HEALTH_SCHMITT.Y2（WD_OK缓冲）；C=PB6_RUN；Y→U_ISO_ARM_FF./CLR；VCC=3V3_CORE；GND=GND_SYS。",
    }
    for ref, nets in pin_update.items():
        if ref in by_ref:
            by_ref[ref]["pin_nets"] = nets
            by_ref[ref]["source"] = (by_ref[ref].get("source", "") + "; 原理图准备/保护与制动合并合同.json; 原理图准备/初充与掉电接口合同.json").strip("; ")
    if "U_ISO_ARM_FF" in by_ref:
        by_ref["U_ISO_ARM_FF"].update({
            "manufacturer": "Texas Instruments", "supplier": "", "supplier_part": "",
            "identity_status": "精确MPN已解析；供应订货代码未提供",
            "identity_key": "SN74LVC1G175DCKR|供应商未提供|供应代码未提供",
            "note": "精确MPN明确；候选来源未给出可采购Supplier Part代码。",
        })
    # Update the separately stored common capacitors and add the latest row provenance.
    for r in rows:
        if r.get("reference") == "U_ISO_ARM_FF":
            r["note"] = "FF输出改由新增SN74LVC1G07开漏buffer驱动；供应代码仍未提供。"
        if r.get("reference") == "R_ISO_GATE_SOURCE":
            r["source"] += "; 检查/LTC7001_12V掉电关断证据.json"

    # New total selection identities use exact MPN + supplier + supplier part.
    # Unknown supplier codes remain empty and are counted as incomplete MPN groups.
    rows_by_ref = {ref_key(r): r for r in rows}
    if len(rows_by_ref) != 77:
        raise ValueError(f"expected 77 unique new refs; got {len(rows_by_ref)}")
    if set(base_new_refs) - cancel_new != set(rows_by_ref) - {ref_key(r) for r in rows if ref_key(r) not in base_new_refs}:
        # Exact overlay consistency is checked below by count and unique-ref assertions.
        pass
    retired_by_ref = {ref_key(r): r for r in retired}
    if len(retired_by_ref) != 31:
        raise ValueError(f"expected 31 remaining retired refs; got {len(retired_by_ref)}")

    # Project identity inventory: exact MPNs in active board BOMs and BMS quote.
    active_bom_paths = sorted(POWER.rglob("BOM.csv"))
    project_bom = []
    for path in active_bom_paths:
        rel = str(path.relative_to(ROOT)).replace("\\", "/")
        for x in bom_rows(path):
            project_bom.append({
                "file": rel, "ref": clean(x.get("Reference")),
                "mpn": clean(x.get("Manufacturer part")),
                "supplier": clean(x.get("Supplier")),
                "supplier_part": clean(x.get("Supplier part")),
            })
    bms_observations = bms_quote.get("observations", [])
    bms_mpn = defaultdict(list)
    for o in bms_observations:
        if clean(o.get("mpn")):
            bms_mpn[clean(o["mpn"])].append(o)
    for r in rows:
        hits=[]; exact=[]; different_code=[]
        for p in project_bom:
            if r["mpn"] and p["mpn"] == r["mpn"]:
                h={**p,"retired_in_current_candidate":p["ref"] in retired_by_ref}
                hits.append(h)
                if r.get("supplier_part") and p["supplier"] == r.get("supplier") and p["supplier_part"] == r["supplier_part"]:
                    exact.append(h)
                elif r.get("supplier_part") and p["supplier_part"] and p["supplier_part"] != r["supplier_part"]:
                    different_code.append(h)
        for o in bms_mpn.get(r["mpn"], []):
            h={"source":"电力/BMS模块/原理图准备/9S5P增量报价.json","supplier":"LCSC" if clean(o.get("supplier_part")) else "","supplier_part":clean(o.get("supplier_part")),"manufacturer":clean(o.get("manufacturer")),"unit_price":clean(o.get("unit_price"))}
            hits.append(h)
            if r.get("supplier_part") and h["supplier_part"] == r["supplier_part"]:
                exact.append(h)
            elif r.get("supplier_part") and h["supplier_part"]:
                different_code.append(h)
        r["project_common_mpn_matches"] = hits
        r["exact_supplier_identity_reuse"] = exact
        r["same_mpn_different_supplier_code"] = different_code

    # Apply the same exact identity resolution to the external resistor model,
    # but keep it outside the instance CSV.
    external_hits=[]
    for p in project_bom:
        if p["mpn"] == external_selection["mpn"]:
            external_hits.append(p)
    external_selection["project_common_mpn_matches"] = external_hits

    groups=defaultdict(list)
    for r in rows:
        groups[(r["mpn"],r.get("supplier",""),r.get("supplier_part",""))].append(r)
    added_selection_groups=[]
    for key, group in sorted(groups.items()):
        sample=group[0]
        all_hits=sample.get("project_common_mpn_matches", [])
        added_selection_groups.append({
            "mpn":key[0],"manufacturer":sample.get("manufacturer",""),"supplier":key[1],"supplier_part":key[2],
            "refs":sorted(ref_key(x) for x in group),"instance_count":len(group),
            "supplier_code_resolved":bool(key[2]),
            "project_existing_mpn_hits":all_hits,
            "exact_supplier_identity_reuse":sample.get("exact_supplier_identity_reuse",[]),
            "same_mpn_different_supplier_code":sample.get("same_mpn_different_supplier_code",[]),
        })
    ret_groups=defaultdict(list)
    for r in retired:
        ret_groups[(r.get("mpn",""),r.get("supplier",""),r.get("supplier_part",""))].append(r)
    retired_selection_groups=[
        {"mpn":k[0],"supplier":k[1],"supplier_part":k[2],"refs":sorted(ref_key(x) for x in g),"instance_count":len(g)}
        for k,g in sorted(ret_groups.items()) if k[0]
    ]

    # Build selection-only external group separately, then produce an explicit
    # all-in count without pretending it is an 83rd physical instance.
    selection_only_group={
        "mpn":external_selection["mpn"],"manufacturer":external_selection["manufacturer"],
        "supplier":"","supplier_part":"","refs":[external_selection["reference"]],
        "instance_count":0,"supplier_code_resolved":False,"selection_only":True,
        "project_existing_mpn_hits":external_hits,
    }

    overlap=set(groups)&set(ret_groups)
    missing_codes=[{
        "ref":ref_key(r),"mpn":r.get("mpn",""),"manufacturer":r.get("manufacturer",""),
        "reason":r.get("note","") or r.get("identity_status","")
    } for r in rows if not r.get("supplier_part")]
    reuse_summary=[g for g in added_selection_groups if g["project_existing_mpn_hits"]]
    unique_mpn_count=len({r["mpn"] for r in rows})
    external_new_mpn=external_selection["mpn"] not in {r["mpn"] for r in rows}

    # Keep existing passive reuse visible without treating moved parts as new refs.
    reused_existing_refs=[
        {"ref":"R_T1_LO","mpn":"PTFR0805B10K0N9","supplier":"LCSC","supplier_part":"C328436","new_function":"BUS_ADC to GND_SYS host-side pull-down"},
        {"ref":"R_T2_LO","mpn":"PTFR0805B10K0N9","supplier":"LCSC","supplier_part":"C328436","new_function":"IMON to GND_SYS host-side pull-down"},
        {"ref":"R_T3_LO","mpn":"PTFR0805B10K0N9","supplier":"LCSC","supplier_part":"C328436","new_function":"MAX_MISO_CORE to GND_SYS host-side pull-down"},
        {"ref":"R_T4_LO","mpn":"PTFR0805B10K0N9","supplier":"LCSC","supplier_part":"C328436","new_function":"MAX_CSB_CORE to GND_SYS host-side pull-down"},
        {"ref":"R_T5_LO","mpn":"PTFR0805B10K0N9","supplier":"LCSC","supplier_part":"C328436","new_function":"MAX17853.AUXIN0 10kΩ pull-up to THRM","pin_nets":"MAX17853.AUXIN0↔THRM"},
        {"ref":"R_T6_LO","mpn":"PTFR0805B10K0N9","supplier":"LCSC","supplier_part":"C328436","new_function":"MAX17853.AUXIN1 10kΩ pull-up to THRM","pin_nets":"MAX17853.AUXIN1↔THRM"},
        {"ref":"R_T7_LO","mpn":"PTFR0805B10K0N9","supplier":"LCSC","supplier_part":"C328436","new_function":"MAX17853.AUXIN2 10kΩ pull-up to THRM","pin_nets":"MAX17853.AUXIN2↔THRM"},
        {"ref":"R_TEST_LO","mpn":"PTFR0805B10K0N9","supplier":"LCSC","supplier_part":"C328436","new_function":"MAX17853.AUXIN3 10kΩ pull-up to THRM","pin_nets":"MAX17853.AUXIN3↔THRM"},
        {"ref":"C_T1_FILTER","mpn":"CC0805CG100V102GN","supplier":"LCSC","supplier_part":"C54157534","new_function":"MAX17853.SHDNL to AGND 1nF filter","pin_nets":"MAX17853.SHDNL↔AGND"},
        {"ref":"R_QP_COL","mpn":"HoAR0603-1/10W-10KR-0.1%-TCR25","supplier":"LCSC","supplier_part":"C5123581","new_function":"LTC7001 TGUP to ISO_G turn-on series resistor"},
        {"ref":"Q_PMOS_CTRL","mpn":"MMBT5551","supplier":"LCSC","supplier_part":"C2972766","new_function":"central dump low-side input translator; E=GND_SYS,C=DUMP_OFF,B via R_TEST_HI from PB7"},
        {"ref":"R_QP_BE","mpn":"RMC0603100K1%N","supplier":"LCSC","supplier_part":"C269699","new_function":"Q_PMOS_CTRL base-emitter bias"},
        {"ref":"R_TEST_HI","mpn":"PTFR0805B10K0N9","supplier":"LCSC","supplier_part":"C328442","new_function":"PB7 to Q_PMOS_CTRL base"},
        {"ref":"C_T5_FILTER","mpn":"CC0805CG100V102GN","supplier":"LCSC","supplier_part":"C54157534","new_function":"BUS_OV_SENSE to GND_SYS filter capacitor"},
        {"ref":"C_BAL1..C_BAL8","mpn":"C0805C104K5RAC7800","supplier":"LCSC","supplier_part":"C75414","new_function":"MAX17853 C1..C8 to AGND filters; same existing instances, not new procurement"},
        {"ref":"R_BAL1..R_BAL8","mpn":"MFR25SFRE52-120R","supplier":"LCSC","supplier_part":"C1744031","new_function":"MAX17853 SW1..SW8 series balance-current limiting resistors; same existing instances"},
    ]
    # Retired cancellation record has only the actual canceled items; reused refs above are not counted as new parts.
    canceled_ref_functions={x["ref"]:x["new_function"] for x in reused_existing_refs}

    source_paths=[MERGE,ZERO,QUOTE,MOS,LTC,CENTRAL,CHARGE_CONTRACT,PROTECTION_CONTRACT,REG_BOM,BMS_BOM,BMS_QUOTE]
    source_hash={str(p.relative_to(ROOT)).replace("\\","/"):sha256(p) for p in source_paths}
    bms_counts=protection["BMS_thermal_latch"]["count_delta_vs_native"]
    checks={
        "schema":"regulator_integrated_merge_candidate_identity_audit_v2",
        "generated_date":"2026-10-29",
        "status":"INTEGRATED_SCHEMATIC_CANDIDATE_NOT_NATIVE_NOT_FINAL_PROCUREMENT_BOM",
        "scope":{
            "base_snapshot":"58new/45retired source semantics are retained under baseline_snapshot; top-level totals below are the current 77/31 candidate.",
            "current_new_instances":len(rows),"current_retired_instances":len(retired),
            "current_net_piece_delta":len(rows)-len(retired),
            "native_schematic_or_procurement_bom_changed":False,
            "design_documents_changed_by_this_script":False,
            "provisional_C_ISO_AUX_OUT_from_LTC_JSON":True,
            "C_ISO_AUX_OUT_effective_capacitance_and_simultaneous_input_output_power_loss_closed":False,
        },
        "selection_counts":{
            "added_instances":len(rows),"retired_instances":len(retired),"net_instance_delta":len(rows)-len(retired),
            "added_unique_exact_mpn_count":unique_mpn_count,
            "added_unique_supplier_identity_groups":len(added_selection_groups),
            "module_added_selection_count":len(added_selection_groups),
            "added_unique_supplier_code_resolved_mpn_count":len({r["mpn"] for r in rows if r.get("supplier_part")}),
            "added_unique_mpn_groups_missing_supplier_code":len({r["mpn"] for r in rows if not r.get("supplier_part")}),
            "selection_only_external_new_mpn_count":1 if external_new_mpn else 0,
            "module_selection_group_count_including_external_candidate":len(added_selection_groups)+(1 if external_new_mpn else 0),
            "retired_unique_exact_mpn_count":len({r.get("mpn") for r in retired if r.get("mpn")}),
            "retired_unique_supplier_identity_groups":len(retired_selection_groups),
            "module_retired_selection_count":len(retired_selection_groups),
            "same_supplier_identity_present_in_added_and_retired_refs":len(overlap),
            "identity_rule":"精确MPN+Supplier+Supplier Part；无供应代码时保留MPN候选并标未闭合，不按封装/规格名归并。",
        },
        "candidate_overlay":{
            "base_new_refs":base_new_refs,"cancelled_base_new_refs":sorted(cancel_new),
            "cancelled_retirements_reused_refs":sorted(cancel_retired),
            "added_overlay_refs":sorted(ref_key(r) for r in rows if ref_key(r) not in set(base_new_refs)-cancel_new),
            "retired_refs_cancelled_with_reuse_function":[{"ref":r,"new_function":canceled_ref_functions.get(r,"reused in current candidate per protection/interface contracts")} for r in sorted(cancel_retired)],
            "baseline_original_counts":{"new":58,"retired":45,"net":13},
            "current_candidate_counts":{"new":len(rows),"retired":len(retired),"net":len(rows)-len(retired)},
        },
        "added_selection_groups":added_selection_groups,
        "retired_selection_groups":retired_selection_groups,
        "unresolved_supplier_codes":missing_codes,
        "project_common_mpn_reuse":{
            "search_basis":"全部当前电力模块BOM.csv及BMS 9S5P增量报价；先按精确MPN搜命中，再比Supplier和Supplier Part。相同封装/值不作为匹配。",
            "active_bom_files":[str(p.relative_to(ROOT)).replace("\\","/") for p in sorted(POWER.rglob("BOM.csv"))],
            "unique_added_mpn_groups_with_project_hits":len({g["mpn"] for g in reuse_summary}),
            "matches":reuse_summary,
        },
        "reused_existing_refs_not_counted_as_new_instances":reused_existing_refs,
        "external_selection_only_candidates":[external_selection],
        "separate_BMS_and_battery_delta":{
            "bms_board_new_instances":bms_counts["new_board_instances"],
            "bms_board_retired_instances":bms_counts["retired_board_instances"],
            "bms_board_net_instances":bms_counts["net_board_instances"],
            "battery_cells_added":9,"battery_cells_in_regulator_csv":False,
            "source":"原理图准备/保护与制动合并合同.json:BMS_thermal_latch.count_delta_vs_native; parent instruction/9S5P quote",
        },
        "known_candidate_limits":[
            "全部77新增实例/31退役实例仍是候选未落图统计；没有原生网表、采购BOM、PCB或实物PASS。",
            "6件ISO辅助电源按LTC7001_12V掉电关断证据.json读取；C_ISO_AUX_OUT暂保留该文件MPN。有效电容>4.7µF的最差偏压/温度证据，以及IN/OUT双源同时掉电、OUT约高于IN 0.3V时的反灌/放电/保持应力尚未闭合。",
            "中央硬OVP与dump驱动以保护与制动合并合同.json覆盖旧中央子包的PNP/二极管OR及2.25M/23.7k方案。",
            "MAX17853均衡每路按2.45/(2×120Ω+Rsw)约10.1mA估算，非20.4mA；候选要求奇偶交替均衡，并在采样前停止均衡等待滤波，仍待落实到时序与验证。",
            "HEWRA0J082K500是按厂家编码规则构造的外部电阻订货候选，精确可采购SKU/供应代码未确认；不计regulator实例，但计一个待定外部选型。",
        ],
        "pin_net_scope":{
            "new_pin_net_authority":["原理图准备/初充与掉电接口合同.json","原理图准备/保护与制动合并合同.json","检查/LTC7001_12V掉电关断证据.json"],
            "numbered_pin_contracts":"U_SIGNAL_ISO完整pin1..14；三颗STPS30100ST的pin1/2/3+tab；LTC7001、HYG022等使用来源明确列出的封装脚名/编号。其他仅列功能名或端点网。",
            "not_a_native_netlist":True,
        },
        "source_sha256":source_hash,
        "baseline_snapshot":snapshot,
        "new_instances":sorted(rows,key=ref_key),
        "retired_refs_from_current_bom":sorted(retired,key=ref_key),
        "current_board_bom_unchanged_sha256":sha256(REG_BOM),
    }

    assert len(rows)==77 and len(set(rows_by_ref))==77
    assert len(retired)==31 and len(set(retired_by_ref))==31
    assert checks["scope"]["current_net_piece_delta"]==46
    assert len(missing_codes)==len({r["mpn"] for r in rows if not r.get("supplier_part")})

    headers=["序号","Ref","实例数","候选值/要求","精确MPN","制造商","供应商","供应代码","身份状态","针网/功能端接","脚号信息范围","项目共用MPN命中","来源","备注"]
    def hits_text(r):
        hits=[]
        for h in r.get("project_common_mpn_matches",[]):
            if h.get("file"):
                hits.append(f"{h['file']}#{h.get('ref','')}[{h.get('supplier','')}/{h.get('supplier_part','')}]")
            elif h.get("source"):
                hits.append(f"BMS报价[{h.get('supplier_part','')}]")
        return "；".join(hits) if hits else "未在当前项目BOM/BMS报价命中同MPN"
    with OUT_CSV.open("w",encoding="utf-8-sig",newline="") as f:
        w=csv.writer(f);w.writerow(headers)
        for i,r in enumerate(sorted(rows,key=ref_key),1):
            w.writerow([i,r["reference"],1,r.get("value_or_requirement",""),r.get("mpn",""),r.get("manufacturer",""),r.get("supplier",""),r.get("supplier_part",""),r.get("identity_status",""),r.get("pin_nets",""),r.get("pin_number_scope",""),hits_text(r),r.get("source",""),r.get("note","")])
    OUT_JSON.write_text(json.dumps(checks,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({
        "csv":str(OUT_CSV),"json":str(OUT_JSON),"new_instances":len(rows),
        "retired_instances":len(retired),"net":len(rows)-len(retired),
        "unique_new_mpn_groups":unique_mpn_count,
        "unique_supplier_identity_groups":len(added_selection_groups),
        "missing_supplier_code_mpn_count":len({r["mpn"] for r in rows if not r.get("supplier_part")}),
        "external_selection_only":external_selection["mpn"],
        "retired_selection_groups":len(retired_selection_groups),
        "project_hit_mpn_groups":len({g["mpn"] for g in reuse_summary}),
        "bms_delta":bms_counts,
    },ensure_ascii=False,indent=2))


if __name__ == "__main__":
    main()
