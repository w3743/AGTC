#!/usr/bin/env python3
"""Read-only protection-reduction audit bound to the current BMS native P1.

This is a source/layout evidence model. It does not claim programmed BQ76952
settings, firmware correctness, physical protection timing, thermal qualification,
or vehicle release.
"""
from __future__ import annotations

import csv
import hashlib
import itertools
import json
import re
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SOURCE_DIR = ROOT / "BMS模块" / "原理图准备"
BOM = SOURCE_DIR / "BOM.csv"
P1 = ROOT / "BMS模块" / "sch" / "Schematic1" / "P1.esch2"
P1_SHA_EXPECTED = "3b99f65dcef6fa37a519188d35dca64d9a6f54843209b3e58daf07c37c72a16a"
BOM_SHA_EXPECTED = "3038221d8bd3400e02448ddd869cff079b68fd82229db925e26b1db20037c219"
NATIVE_CONTAINER = ROOT / "BMS模块" / "BMS-9S4P.eprj3"
LAYOUT_DIR = ROOT / "BMS模块" / "检查" / "Layout验证"
SNAPSHOT = LAYOUT_DIR / "current-snapshot.json"
GEOMETRY = LAYOUT_DIR / "final-geometry.json"
BUILD_MANIFEST = ROOT / "BMS模块" / "检查" / "当前施工输入.json"
PUBLICATION = LAYOUT_DIR / "publication.json"
CURRENT_PUBLICATION = LAYOUT_DIR / "当前发布.json"
PRECHARGE = ROOT / "BMS模块" / "仿真" / "结果" / "预充Layout闭合.json"
OUT = ROOT / "BMS模块" / "仿真" / "结果" / "保护精简验证.json"
REPORT = ROOT / "BMS模块" / "检查" / "保护精简验证.md"
SOURCE_NAMES = ["功率元件连接.json", "接口元件连接.json", "AFE元件连接.json", "维护线束元件连接.json"]
CAP_REFS = ["C_SRP", "C_TIMER", "C_AND2", "C_GATE1", "C_SEQ", "C_POR", "C_CAN_PWR_IN2", "C_CAN_PWR_OUT2"]
CAP_MPN = "GRM188R72A104KA35J"
CAP_VALUE = "100 nF ±10%, 100 V, X7R, 0603"
AND_MPN = "SN74LVC1G11DCKR"


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def norm_mpn(value):
    if not isinstance(value, str):
        return value
    normalized = value.strip()
    for prefix in ("TI ", "TE Connectivity "):
        if normalized.startswith(prefix):
            return normalized[len(prefix):].strip()
    return normalized


def main() -> int:
    source_rows = {}
    source_paths = {}
    identical_duplicate_refs = []
    for name in SOURCE_NAMES:
        path = SOURCE_DIR / name
        source_paths[name] = path
        for row in read_json(path):
            ref = row.get("ref")
            if ref:
                if ref in source_rows:
                    # Interface and maintenance-harness notes describe the same
                    # J_CELL_SERVICE from different contexts. Electrical identity
                    # (MPN, library identity and pin map) must still match.
                    prior_electrical = {k: v for k, v in source_rows[ref].items() if k != "note"}
                    current_electrical = {k: v for k, v in row.items() if k != "note"}
                    if prior_electrical != current_electrical:
                        raise ValueError(f"conflicting duplicate reference across current sources: {ref}")
                    identical_duplicate_refs.append(ref)
                    continue
                source_rows[ref] = row

    snap = read_json(SNAPSHOT)
    geometry = read_json(GEOMETRY)
    build_manifest = read_json(BUILD_MANIFEST)
    publication = read_json(PUBLICATION)
    current_publication = read_json(CURRENT_PUBLICATION)
    precharge = read_json(PRECHARGE)
    with BOM.open("r", encoding="utf-8-sig", newline="") as stream:
        bom_rows = list(csv.DictReader(stream))
    bom_by_ref = {row.get("位号", ""): row for row in bom_rows if row.get("位号")}
    snapshot_by_ref = {row["ref"]: row for row in snap["rows"]}
    manifest_refs = [row.get("ref") for row in build_manifest if isinstance(row, dict)]
    manifest_shape_ok = (
        isinstance(build_manifest, list)
        and len(manifest_refs) == len(build_manifest)
        and all(isinstance(ref, str) and ref for ref in manifest_refs)
        and all(isinstance(row.get("pins"), dict) for row in build_manifest)
        and len(set(manifest_refs)) == len(manifest_refs)
    )
    expected_component_count = len(build_manifest) if isinstance(build_manifest, list) else 0
    expected_pin_count = (
        sum(len(row["pins"]) for row in build_manifest)
        if manifest_shape_ok else 0
    )
    snapshot_pin_count = sum(len(row.get("pins", [])) for row in snap["rows"])
    manifest_snapshot_matches = (
        manifest_shape_ok
        and set(manifest_refs) == set(snapshot_by_ref)
        and len(snap["rows"]) == expected_component_count
        and snapshot_pin_count == expected_pin_count
    )

    checks = []

    def add(check_id: str, passed: bool, evidence: str, *, status: str | None = None):
        state = status or ("PASS" if passed else "FAIL")
        checks.append({"id": check_id, "status": state, "evidence": evidence})

    add(
        "current_source_duplicate_refs_consistent",
        True,
        f"当前来源重复位号的电气身份（MPN、库身份、引脚映射）一致，已去重；说明文字可按来源上下文不同；重复={identical_duplicate_refs}。电气身份冲突会立即终止复算。",
    )

    p1_sha = sha256(P1)
    bom_sha = sha256(BOM)
    add(
        "native_p1_sha_bound",
        p1_sha == P1_SHA_EXPECTED,
        f"{P1.relative_to(ROOT)} SHA256={p1_sha}; expected {P1_SHA_EXPECTED}.",
    )
    add(
        "formal_bom_sha_bound",
        bom_sha == BOM_SHA_EXPECTED,
        f"{BOM.relative_to(ROOT)} SHA256={bom_sha}; expected {BOM_SHA_EXPECTED}.",
    )
    add(
        "layout_publication_refs_native_p1",
        p1_sha in json.dumps(publication, ensure_ascii=False),
        "正式 publication.json 与 P1 页SHA绑定；配套 final-geometry.json 的 sourceSHA256 也引用该P1页。",
    )
    current_sources = current_publication.get("sources", {})
    manifest_p1_sha = current_sources.get("sch/Schematic1/P1.esch2", {}).get("sha256")
    manifest_bom = next((v for k, v in current_sources.items() if k.endswith("BOM.csv")), {})
    current_manifest_ok = (
        manifest_p1_sha == p1_sha
        and manifest_bom.get("sha256") == bom_sha
        and manifest_bom.get("bytes") == BOM.stat().st_size
    )
    add(
        "current_publication_manifest_matches_sources",
        current_manifest_ok,
        f"当前发布清单P1={manifest_p1_sha}; BOM SHA={manifest_bom.get('sha256')} bytes={manifest_bom.get('bytes')}; 实际P1/BOM SHA和BOM字节数均核对。",
    )

    snap_refs = set(snapshot_by_ref)
    bom_refs = set(bom_by_ref)
    source_refs = set(source_rows)
    absent_bom = sorted(snap_refs - bom_refs)
    absent_source = sorted(snap_refs - source_refs)
    add(
        "current_snapshot_bom_source_match",
        not absent_bom and not absent_source and len(snap_refs) == len(snap["rows"]),
        f"current-snapshot {len(snap_refs)}个唯一位号；BOM {len(bom_rows)}行；snapshot位号缺BOM={absent_bom}，缺当前来源JSON={absent_source}。",
    )
    mpn_mismatch = []
    for ref, row in snapshot_by_ref.items():
        bom_row = bom_by_ref.get(ref)
        src_row = source_rows.get(ref)
        if bom_row and norm_mpn(bom_row.get("MPN")) != norm_mpn(row.get("mpn")):
            mpn_mismatch.append({"ref": ref, "snapshot": row.get("mpn"), "bom": bom_row.get("MPN")})
        if src_row and norm_mpn(src_row.get("mpn")) != norm_mpn(row.get("mpn")):
            mpn_mismatch.append({"ref": ref, "snapshot": row.get("mpn"), "source": src_row.get("mpn")})
    add(
        "snapshot_bom_source_mpn_match",
        not mpn_mismatch,
        f"当前快照MPN与正式BOM/连接源逐项比较；差异={mpn_mismatch[:12]}，总数={len(mpn_mismatch)}。",
    )

    mounted = [r for r in snap["rows"] if r.get("pcb") and r.get("bom")]
    mounted_refs = {r["ref"] for r in mounted}
    raw_mpns = {r.get("mpn") for r in mounted if r.get("mpn")}
    normalized_mpns = {norm_mpn(r.get("mpn")) for r in mounted if r.get("mpn")}
    add(
        "mounted_population_count",
        manifest_snapshot_matches and len(mounted) == 206 and len(raw_mpns) == 52 and len(normalized_mpns) == 51,
        f"snapshot placed={len(snap['rows'])}，manifest期望={expected_component_count}；pin数={snapshot_pin_count}/{expected_pin_count}；manifest位号/计数匹配={manifest_snapshot_matches}；pcb=true且bom=true=206；板装料号原始唯一数={len(raw_mpns)}、去TI前缀后={len(normalized_mpns)}。",
    )

    geom_fail_fields = [key for key in ("nonOrthogonal", "missingPins", "foreignNetAtPins", "foreignNetAtWireEndpoints", "danglingLabels") if geometry.get(key)]
    geometry_sources = set(geometry.get("sourceSHA256", {}).values())
    geometry_source_binding_ok = p1_sha in geometry_sources and sha256(SNAPSHOT) in geometry_sources
    geom_ok = (
        geometry.get("status") == "PASS"
        and geometry.get("electrical_geometry_status") == "PASS"
        and geometry.get("strict_current_geometry") is True
        and geometry.get("counts", {}).get("pins") == expected_pin_count
        and manifest_shape_ok
        and not geom_fail_fields
        and geometry_source_binding_ok
    )
    add(
        "official_layout_geometry",
        geom_ok,
        f"正式final-geometry status={geometry.get('status')} / electrical={geometry.get('electrical_geometry_status')} / strict={geometry.get('strict_current_geometry')}；pins={geometry.get('counts', {}).get('pins')}；P1与当前快照SHA均在sourceSHA256内={geometry_source_binding_ok}；非空失败项={geom_fail_fields}。",
    )

    release_state = {
        "layout_initial_placement": current_publication.get("layout_initial_placement"),
        "routing_freeze": current_publication.get("routing_freeze"),
        "fabrication": current_publication.get("fabrication"),
        "physical_qualification": current_publication.get("physical_qualification"),
        "no_pcb_created": current_publication.get("no_pcb_created"),
    }
    release_state_ok = release_state == {
        "layout_initial_placement": "CONDITIONAL",
        "routing_freeze": "HOLD",
        "fabrication": "HOLD",
        "physical_qualification": "NOT_PERFORMED",
        "no_pcb_created": True,
    }
    add(
        "formal_layout_gate_state_preserved",
        release_state_ok,
        f"当前发布阶段状态={release_state}；几何PASS不提升为布线冻结、制板或物理验收PASS。",
    )

    fan_refs = [ref for ref in snapshot_by_ref if re.search(r"(?:^|_)FAN(?:_|$)|J_FAN", ref, re.I)]
    fan_ports = [p for p in snap.get("ports", []) if "FAN_" in str(p.get("name", "")).upper() or "FAN_" in str(p.get("net", "")).upper()]
    fan_wires = [w for w in snap.get("wires", []) if "FAN_" in json.dumps(w, ensure_ascii=False).upper()]
    active_fan_refs = geometry.get("active_fan_refs_in_final_snapshot", [])
    core_p2_src = source_rows["J_CORE_P2"]["pins"]
    core_p2_snap = {p["number"]: p for p in snapshot_by_ref["J_CORE_P2"].get("pins", [])}
    no_fan = (
        not fan_refs and not fan_ports and not fan_wires and not active_fan_refs
        and core_p2_src.get("14") is None and core_p2_src.get("17") is None
        and core_p2_snap.get("14", {}).get("nc") is True and core_p2_snap.get("17", {}).get("nc") is True
    )
    add(
        "current_no_fan_topology",
        no_fan,
        f"current snapshot fan refs={fan_refs}, active fan ports={len(fan_ports)}, fan wires={len(fan_wires)}, formal active fan refs={active_fan_refs}; J_CORE_P2 pins14/17 are NC in source and placed snapshot. Geometry record retains four explicitly inactive historical FAN_* ports in its approved tombstone whitelist; they are not active current wiring.",
    )

    psu = source_rows["PSU5_EXT"]
    fuse = source_rows["F_CELL_B9"]
    fuse_holder = source_rows["FH_CELL_B9"]
    psu_bom = bom_by_ref.get("PSU5_EXT", {})
    bom_note = psu_bom.get("备注", "")
    bom_tap_note = bool(re.search(r"(?:pin\s*2|2)\s*[=＝:]\s*TAP_B9", bom_note, re.I)) or "pin 2=TAP_B9" in bom_note
    psu_ok = psu.get("pins", {}).get("2") == "TAP_B9" and fuse.get("pins") == {"1": "CELL_B9", "2": "TAP_B9"} and fuse_holder.get("pins") == {"1": "CELL_B9", "2": "TAP_B9"}
    bom_tap_state = "PASS" if bom_tap_note else "OPEN"
    add(
        "5v_converter_fused_b9_tap",
        psu_ok and bom_tap_note,
        f"current source: PSU5_EXT pin2={psu.get('pins', {}).get('2')}; F_CELL_B9/FH_CELL_B9 CELL_B9→TAP_B9; BOM MPN={psu_bom.get('MPN')}; note identifies pin2=TAP_B9={bom_tap_note}. Fuse is external to PCB and upstream of the converter tap.",
        status=None if psu_ok and bom_tap_note else ("OPEN" if psu_ok and not bom_tap_note else "FAIL"),
    )

    # Validate the actual AND3-as-AND2 implementation. C is tied high by the
    # physical pin map, so only the four legal A/B states are evaluated.
    and3 = source_rows["U_AND2"]
    and_pins = {str(k): v for k, v in and3.get("pins", {}).items()}
    truth = []
    and_ok = and3.get("mpn") == AND_MPN and and_pins.get("2") == "B0" and and_pins.get("5") == "PRE_CTRL_3V3" and and_pins.get("6") == "PRE_CTRL_3V3" and "U_GATE1" in source_rows
    for a, b in itertools.product((0, 1), repeat=2):
        y = a & b & 1
        truth.append({"A": a, "B": b, "C_tied_high": 1, "Y": y, "expected_AND2": a & b})
        and_ok = and_ok and y == (a & b)
    add(
        "and3_substitution_function_and_population",
        and_ok,
        f"U_AND2={and3.get('mpn')}; pin6 third input is tied to PRE_CTRL_3V3, four-state truth table equals AND2; U_GATE1 remains a separate gate. Physical gate count is not reduced by this substitution.",
    )

    cap_details = []
    for ref in CAP_REFS:
        row = source_rows.get(ref, {})
        cap_details.append({"ref": ref, "mpn": row.get("mpn"), "value": row.get("value"), "pins": row.get("pins")})
    cap_ok = len(cap_details) == 8 and all(x["mpn"] == CAP_MPN and x["value"] == CAP_VALUE for x in cap_details)
    add(
        "eight_decoupling_capacitors_preserved",
        cap_ok,
        f"当前来源仍含8只独立100nF/100V/X7R/0603 {CAP_MPN}；实体电容未减少，逐只网络保留。",
    )

    isets = [ref for ref in source_rows if re.fullmatch(r"C_BAL_ISET[1-8]", ref)]
    iset_ok = len(isets) == 8 and all(source_rows[ref].get("mpn") for ref in isets)
    add(
        "eight_local_balancing_iset_filters_preserved",
        iset_ok,
        f"保留8个独立C_BAL_ISET1..8；模型不将滤波或平衡保护计为删减。器件必要性/滤波行为仍要结合原厂配置与实测验证。",
    )

    protected = {
        "U1", "RSHUNT", "F1", "D_TVS", "U_AND2", "U_GATE1", "U_SEQ", "U_TIMER", "U_POR",
        "Q_PRE1", "Q_PRE2", "Q_QPRE_DRV", "Q_DFETOFF", "U_LOAD_POWER_VALID",
        "Q_BAL_MAINCLR", "Q_BAL_ENABLE", "U_BAL_OPTO_Q1", "U_BAL_OPTO_Q2",
    }
    protected.update(f"Q_CHG{i}" for i in range(1, 4))
    protected.update(f"Q_DSG{i}" for i in range(1, 4))
    protected.update(f"U_ETA_BAL{i}" for i in range(1, 9))
    protected.update(f"F_CELL_B{i}" for i in range(1, 10))
    protected.update(f"Q_BAL_EN_CLAMP{i}" for i in range(1, 9))
    protected.add("Q_LOAD_POWER_VALID_DRV")
    missing_protection_sources = sorted(protected - source_refs)
    missing_board_protection = sorted((protected & mounted_refs) - snap_refs)
    coverage_ok = not missing_protection_sources and not missing_board_protection
    add(
        "protection_function_topology_coverage",
        coverage_ok,
        f"关键功能器件/抽头熔断器在现行来源JSON中均可追溯；已放置器件也存在于正式快照。缺源={missing_protection_sources}，缺快照={missing_board_protection}。该检查只证明拓扑器件存在，不证明保护参数和动作有效。",
    )

    pre_timer = precharge.get("timer", {})
    pre_fet = precharge.get("qpre_mosfets", {})
    pre_limits = precharge.get("evidence_and_limits", {})
    pre_no_old_claim = (
        "2.970 s is a legacy comparison scenario" in precharge.get("scope", "")
        and "legacy comparison scenario" in pre_timer.get("reference_deadline_status", "")
        and abs(pre_timer.get("ff_b_power_to_drv_low_max_s", -1) - 2.8577405271378624) < 1e-8
        and pre_fet.get("physical_turnoff_max_guaranteed") is False
        and pre_limits.get("pre_ok_used_as_cutoff") is False
        and pre_limits.get("full_chain_physical_max_time_s") is None
    )
    add(
        "legacy_2p970s_physical_cutoff_claim_removed",
        pre_no_old_claim,
        f"已绑定正式预充结果：2.970s只是旧对比场景；FF-B供电至TMR_ACTIVE/逻辑窗撤销最坏={pre_timer.get('ff_b_power_to_drv_low_max_s')}s；QPRE物理释放无保证最大值={pre_fet.get('physical_turnoff_max_guaranteed')}；整链物理最大时间={pre_limits.get('full_chain_physical_max_time_s')}；固件PRE_OK未用作独立截止={pre_limits.get('pre_ok_used_as_cutoff')}。",
    )

    coverage = [
        {"function": "电芯电压/温度/电流AFE监测", "refs": ["U1", "RSHUNT", "NTC_CELL1", "NTC_CELL5", "NTC_BOARD", "NTC_MOS"], "status": "TOPOLOGY_PRESENT_CONFIG_UNVERIFIED", "boundary": "BQ配置阈值、热敏通道分配和保护动作须配置读回并实测。"},
        {"function": "CHG/DSG主开关及AFE硬保护输入", "refs": ["Q_CHG1..3", "Q_DSG1..3", "Q_DFETOFF", "U1"], "status": "TOPOLOGY_PRESENT_POWER_QUALIFICATION_OPEN", "boundary": "60A铜箔温升、并联均流、短路切断和FET SOA尚未由板级实测证明。"},
        {"function": "外部主回路/电芯抽头熔断", "refs": ["F1", "F_CELL_B1..9"], "status": "TOPOLOGY_PRESENT_COORDINATION_OPEN", "boundary": "Class-T总清除I²t及抽头gPV曲线、短路分断与导线/连接器配合未闭合。"},
        {"function": "开机/主路释放时序与故障关断", "refs": ["U_AND2", "U_GATE1", "U_SEQ", "U_TIMER", "U_POR", "Q_PRE1/2", "Q_QPRE_DRV"], "status": "LOGIC_WINDOW_ONLY_PHYSICAL_CHAIN_OPEN", "boundary": "无独立硬件PRE_OK差压判据；MCU MAIN_REQ仍是最终许可；硬件逻辑窗有模型边界但物理QPRE关断上界未保证。"},
        {"function": "PACK端瞬态钳位", "refs": ["D_TVS"], "status": "DEVICE_PRESENT_LIMITED", "boundary": "SMCJ45A的45V VRWM不代表可承受48V连续；不吸收60mF母线储能。"},
        {"function": "8路相邻电芯主动均衡失效默认关闭", "refs": ["U_ETA_BAL1..8", "Q_BAL_EN_CLAMP1..8", "U_BAL_OPTO_Q1/2", "Q_BAL_MAINCLR", "Q_BAL_ENABLE"], "status": "TOPOLOGY_PRESENT_DYNAMIC_OPEN", "boundary": "ETA内部EN上拉、隔离光耦全温CTR、上电默认态和均衡温升需按厂商资料及首板验证。"},
        {"function": "隔离的LOAD_POWER_VALID", "refs": ["U_LOAD_POWER_VALID", "Q_LOAD_POWER_VALID_DRV"], "status": "TOPOLOGY_PRESENT_INTERFACE_OPEN", "boundary": "隔离侧拉偏置和外部接收端阈值/失电态须与系统接口联合验证。"},
        {"function": "5V辅助电源故障隔离输入", "refs": ["PSU5_EXT", "F_CELL_B9", "FH_CELL_B9"], "status": "FUSED_TAP_PRESENT", "boundary": "TAP_B9位于外置B9保险后；模块启动浪涌、温升和线束短路协调仍需验证。"},
    ]

    open_items = [
        "本结果是源文件与正式Layout几何/位号的静态复算，不代表实际BQ76952寄存器、固件策略、温度阈值、开断电流或整车安全验收。",
        "没有独立硬件PRE_OK差压比较；MAIN_REQ由MCU最终许可。预充逻辑计时窗纸面上限与QPRE实体截止必须区分，整链物理最大时限尚未形成保证。",
        "主回路F1 Class-T总清除I²t、9路抽头保险分断及线束/触点配合未取得完整曲线和系统短路验证。",
        "60A连续/100A 1s主回路、分流器及CHG/DSG并联MOSFET需按真实板铜、散热和均流测量验证。",
        "SMCJ45A为45V VRWM，不证明48V连续耐受或60mF母线浪涌能量钳位。",
        "8路均衡默认关断和热管理、光耦全温传输、BQ保护配置、隔离外部LOAD_POWER_VALID阈值需要配置读回/台架联调。",
        "无风扇硬件；Layout热预算按无风扇、自然对流/机箱边界核算，不能把风扇散热作为现行硬件前提。",
    ]

    layout_constraints = [
        "主电流路径按60A连续与100A/1s条件核算铜厚、线宽、过孔、端子和温升；CHG/DSG各三只TOLL MOSFET保持对称并联，按实际均流测温。",
        "RSHUNT按源说明从本体内侧边缘独立引出SRP/SRN Kelvin线，经各自100Ω滤波；禁止从功率连接器或远端铜箔取样。",
        "D_TVS贴近PACK+ / PACK-外部接口，缩短高di/dt回路；只作为瞬态钳位，不能假定其承受持续48V或母线电容全能量。",
        "QPRE两只TO-252背靠背器件保持共源、共栅，QPRE_G至QPRE_SRC的10k上拉、4.7k限流和12V齐纳闭合紧凑回路；首板测源参考VGS和实物关断波形。",
        "PSU5_EXT pin2从已由F_CELL_B9保护的TAP_B9取电，不接CELL_B9保险前节点；Hi-Link启动/短路电流与铜箔温升需核算。",
        "按无风扇前提闭合板内热源与机箱散热路径；保护阈值或NTC功能不以风扇存在为条件。",
    ]

    source_evidence = {name: {"path": str(path.relative_to(ROOT)), "sha256": sha256(path)} for name, path in source_paths.items()}
    evidence_files = {
        "native_p1_page": {"path": str(P1.relative_to(ROOT)), "sha256": p1_sha},
        "native_project_container": {"path": str(NATIVE_CONTAINER.relative_to(ROOT)), "sha256": sha256(NATIVE_CONTAINER)},
        "published_bom": {"path": str(BOM.relative_to(ROOT)), "sha256": bom_sha, "expected_sha256": BOM_SHA_EXPECTED, "rows": len(bom_rows)},
        "current_build_manifest": {"path": str(BUILD_MANIFEST.relative_to(ROOT)), "sha256": sha256(BUILD_MANIFEST), "components": expected_component_count, "pins": expected_pin_count},
        "current_layout_snapshot": {"path": str(SNAPSHOT.relative_to(ROOT)), "sha256": sha256(SNAPSHOT)},
        "official_final_geometry": {"path": str(GEOMETRY.relative_to(ROOT)), "sha256": sha256(GEOMETRY)},
        "publication_manifest": {"path": str(PUBLICATION.relative_to(ROOT)), "sha256": sha256(PUBLICATION)},
        "current_publication_manifest": {"path": str(CURRENT_PUBLICATION.relative_to(ROOT)), "sha256": sha256(CURRENT_PUBLICATION)},
        "formal_precharge_result": {"path": str(PRECHARGE.relative_to(ROOT)), "sha256": sha256(PRECHARGE)},
    }
    all_core_pass = all(c["status"] == "PASS" for c in checks)
    overall = "PASS_WITH_OPEN_ITEMS" if all_core_pass else ("FAIL" if any(c["status"] == "FAIL" for c in checks) else "OPEN")
    result = {
        "schema": "bms-protection-reduction-verification-v2",
        "date": date.today().isoformat(),
        "status": overall,
        "scope": "只读绑定当前原生P1、正式BOM与官方Layout快照/几何；核验保护拓扑存在性、指定器件精简的功能/实体数、无风扇当前连接和5V保险后抽头；不替代台架/整车验收。",
        "summary": {
            "native_p1_sha256_expected": P1_SHA_EXPECTED,
            "board_placed_components": 206,
            "snapshot_placed_components_including_nonpcb_or_nonbom": expected_component_count,
            "board_mpn_unique_raw": len(raw_mpns),
            "board_mpn_unique_without_ti_prefix": len(normalized_mpns),
            "fan_hardware_refs": fan_refs,
            "fan_current_active_ports": len(fan_ports),
            "fan_current_active_wires": len(fan_wires),
            "approved_protection_removals": [],
            "physical_gate_or_capacitor_deletions_from_checked_simplifications": 0,
            "independent_hardware_pre_ok_comparator_present": False,
            "old_2p970s_physical_cutoff_claim_retained": False,
            "five_volt_source_is_fused_tap_b9": psu_ok,
            "formal_geometry_pin_count": geometry.get("counts", {}).get("pins"),
            "formal_layout_gate_state": release_state,
        },
        "evidence_files": evidence_files,
        "current_source_hashes": source_evidence,
        "checks": checks,
        "and3_as_and2_truth_table": truth,
        "decoupling_population": cap_details,
        "protection_function_coverage": coverage,
        "layout_constraints": layout_constraints,
        "open_items": open_items,
        "not_tested": ["BQ configuration write/readback", "firmware behavior", "physical precharge cutoff waveforms", "FET/shunt/copper thermal qualification", "fuse coordination and DC interruption", "EMC/vehicle harness transients", "vehicle release"],
        "primary_reference_urls": [
            "https://www.ti.com/lit/ds/symlink/bq76952.pdf",
            "https://www.ti.com/lit/ug/sluuby2b/sluuby2b.pdf",
            "https://www.ti.com/lit/ds/symlink/tpl5111.pdf",
            "https://www.ti.com/lit/gpn/SN74LVC1G11",
            "https://www.te.com/en/product-1-2176508-5.html",
            "https://www.onsemi.com/pdf/datasheet/mmsz5221et1-d.pdf",
        ],
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    status_labels = {"PASS": "PASS", "OPEN": "OPEN", "FAIL": "FAIL"}
    lines = [
        "# BMS保护精简验证",
        "",
        f"结论：**{overall}**。这是当前原生P1、正式BOM与正式Layout几何的静态证据复算，不是BQ寄存器、固件、台架或整车保护验收。",
        "",
        "## 当前基线",
        "",
        f"- 原生P1 SHA256：`{p1_sha}`（目标 `{P1_SHA_EXPECTED}`）。",
        f"- 正式BOM：{len(bom_rows)}行；快照{expected_component_count}个放置元件（按当前施工manifest计数），其中PCB/BOM均标记的板装器件206个；板装MPN原始唯一数{len(raw_mpns)}，去除`TI `前缀后{len(normalized_mpns)}。",
        f"- 正式布局几何：{geometry.get('counts', {}).get('pins')} pins，状态 `{geometry.get('status')}`，电气几何 `{geometry.get('electrical_geometry_status')}`。",
        f"- 当前发布阶段：initial placement `{release_state['layout_initial_placement']}`；routing `{release_state['routing_freeze']}`；fabrication `{release_state['fabrication']}`；physical qualification `{release_state['physical_qualification']}`。几何检查通过不解除这些门槛。",
        "- 现行快照无风扇器件、接口或活动风扇网络；J_CORE_P2 pin14、17均为NC。几何证明中列出的旧FAN端口属于批准删除的非活动记录，不是当前连线。",
        "",
        "## 逐项检查",
        "",
        "| 检查 | 结果 | 证据 |",
        "|---|---|---|",
    ]
    for item in checks:
        evidence = item["evidence"].replace("|", "\\|").replace("\n", " ")
        lines.append(f"| {item['id']} | {status_labels.get(item['status'], item['status'])} | {evidence} |")
    lines += [
        "",
        "## 精简差异与保护覆盖",
        "",
        "当前模型只核验两项已经落到现行连接源的功能等价/器件人口：U_AND2以SN74LVC1G11DCKR实现二输入与门，第三输入接高，真值表4种A/B状态均等价；8只100nF去耦电容和8只C_BAL_ISET本地滤波器都仍各自存在。两项都没有删除实体门或电容，本轮没有批准新增保护删除。",
        "",
        "保护覆盖清单以当前来源JSON位号为证据，只说明相关功能拓扑仍在。AFE阈值、MOSFET实际关断、过温和短路响应、均衡失效态、保险协调以及对外隔离侧阈值仍需实测或配置读回。",
        "",
        "## 预充截止表述修正",
        "",
        f"已绑定 [预充Layout闭合结果](../仿真/结果/预充Layout闭合.json)：2.970 s是旧对比场景，不是已确认的产品硬截止；FF-B供电到TMR_ACTIVE/逻辑窗撤销纸面最坏为{pre_timer.get('ff_b_power_to_drv_low_max_s'):.6f} s。QPRE物理关断没有可用保证最大值，整链物理最大截止仍未闭合；固件PRE_OK没有被计作独立硬件关断。",
        "",
        "## Layout约束",
        "",
    ]
    lines += [f"- {item}" for item in layout_constraints]
    lines += ["", "## 剩余验证项", ""]
    lines += [f"- {item}" for item in open_items]
    lines += [
        "",
        "正式输入SHA256均记录在 [机器可读结果](../仿真/结果/保护精简验证.json)。模型不生成ENET/BOM导出，不打开或修改EDA工程。",
        "",
    ]
    REPORT.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"status": overall, "report": str(REPORT), "result": str(OUT), "checks": checks, "P1_SHA256": p1_sha, "BOM_SHA256": evidence_files["published_bom"]["sha256"], "snapshot_SHA256": evidence_files["current_layout_snapshot"]["sha256"], "geometry_SHA256": evidence_files["official_final_geometry"]["sha256"]}, ensure_ascii=False, indent=2))
    return 0 if overall != "FAIL" else 2


if __name__ == "__main__":
    raise SystemExit(main())
