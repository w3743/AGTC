"""Generate a read-only candidate wiring plan for nearby BMS control circuits.

The planner reads the official API snapshot and the existing pin-net manifest.
It never starts or connects to EasyEDA and writes only candidate-local-controls.json
beside this script. Every proposed route is checked against the strict native
geometry folded by verify_semantic_geometry.py.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
MODULE = HERE.parents[1]
# afe-snapshot.json is the latest saved-state capture after the parent agent's
# VC/SRP work.  Keep baseline-snapshot.json as a provenance/reference input,
# but do not route from its stale coordinates or netports.
BASELINE = HERE / "afe-snapshot.json"
REFERENCE_BASELINE = HERE / "baseline-snapshot.json"
MANIFEST = MODULE / "检查" / "当前施工输入.json"
NATIVE = MODULE / "sch" / "Schematic1" / "P1.esch2"
VERIFIER = MODULE / "检查" / "DRC修复" / "verify_semantic_geometry.py"
DEFAULT_OUTPUT = HERE / "candidate-local-controls.json"


# Routes use [start pin, orthogonal bend points..., end pin/junction].
# Pins are identified by reference and pin number so stale coordinates fail closed.
ROUTES: list[dict[str, Any]] = [
    {
        "id": "can_input_vout_parallel",
        "net": "VOUT_5V",
        "start": ["C_CAN_PWR_IN1", "1"],
        "via": [[920, 680], [1000, 680]],
        "end": ["C_CAN_PWR_IN2", "1"],
        "purpose": "并联输入去耦电容的VOUT_5V端",
    },
    {
        "id": "can_input_b0_parallel",
        "net": "B0",
        "start": ["C_CAN_PWR_IN1", "2"],
        "via": [[960, 720], [1040, 720]],
        "end": ["C_CAN_PWR_IN2", "2"],
        "purpose": "并联输入去耦电容的B0端",
    },
    {
        "id": "can_output_iso5v_parallel",
        "net": "CAN_ISO_5V",
        "start": ["C_CAN_PWR_OUT1", "1"],
        "via": [[1160, 680], [1240, 680]],
        "end": ["C_CAN_PWR_OUT2", "1"],
        "purpose": "并联隔离侧去耦电容的CAN_ISO_5V端",
    },
    {
        "id": "can_output_pack_return_parallel",
        "net": "PACK-",
        "start": ["C_CAN_PWR_OUT1", "2"],
        "via": [[1200, 720], [1280, 720]],
        "end": ["C_CAN_PWR_OUT2", "2"],
        "purpose": "并联隔离侧去耦电容的PACK-端",
    },
    {
        "id": "por_vdd_filter",
        "net": "PRE_CTRL_3V3",
        "start": ["C_POR", "1"],
        "via": [[4020, 2590], [4120, 2590], [4120, 2400], [3960, 2400], [3960, 2420]],
        "end": ["U_POR", "3"],
        "purpose": "C_POR绕过POR_OK_N原有线连接U_POR的VDD引脚",
    },
    {
        "id": "por_ground_filter",
        "net": "B0",
        "start": ["C_POR", "2"],
        "via": [[4080, 2560], [4080, 2470], [3975, 2470], [3975, 2440]],
        "end": ["U_POR", "1"],
        "purpose": "C_POR绕过POR_OK_N原有线连接U_POR的GND引脚",
    },
    {
        "id": "timer_vdd_filter",
        "net": "TMR_VDD",
        "start": ["C_TIMER", "1"],
        "via": [[3180, 2560], [3180, 2450]],
        "end": ["U_TIMER", "1"],
        "purpose": "C_TIMER连接U_TIMER的TMR_VDD引脚",
    },
    {
        "id": "timer_ground_filter",
        "net": "B0",
        "start": ["C_TIMER", "2"],
        "via": [[3420, 2560], [3420, 2580], [3100, 2580], [3100, 2440]],
        "end": ["U_TIMER", "2"],
        "purpose": "C_TIMER绕开TMR_SET原有线连接U_TIMER的GND引脚",
    },
    {
        "id": "ntc_board_filter_return",
        "net": "B0",
        "start": ["C_NTC_BOARD_FILT", "2"],
        "via": [[2795, 1720], [2850, 1720], [2850, 1680], [2870, 1680]],
        "end": ["NTC_BOARD", "2"],
        "purpose": "将板载NTC滤波电容地端接到NTC_BOARD回流",
    },
    {
        "id": "ntc_sensor_returns_local_bus",
        "net": "B0",
        "start": ["NTC_BOARD", "2"],
        "via": [[2850, 1660], [2850, 1720], [2990, 1720], [2990, 1680], [2970, 1680]],
        "end": ["NTC_MOS", "2"],
        "purpose": "将NTC_BOARD与NTC_MOS的本地B0回流合并；支路供滤波电容使用",
    },
    {
        "id": "ntc_mos_filter_signal",
        "net": "BQ_TS3",
        "start": ["NTC_MOS", "1"],
        "via": [[2910, 1640], [3045, 1640]],
        "end": ["C_BQ_TS3_FILT", "1"],
        "purpose": "将MOS温度NTC与BQ_TS3滤波电容信号端相连",
    },
    {
        "id": "ntc_mos_filter_return",
        "net": "B0",
        "start": ["C_BQ_TS3_FILT", "2"],
        "via": [[3075, 1680], [2970, 1680]],
        "end": ["NTC_MOS", "2"],
        "purpose": "将BQ_TS3滤波电容地端接入本地NTC回流总线",
    },
    {
        "id": "ntc_ts1_filter_return",
        "net": "B0",
        "start": ["C_BQ_TS1_FILT", "2"],
        "via": [],
        "end": ["C_BQ_TS3_FILT", "2"],
        "purpose": "将两路AFE温度滤波电容的B0端合并",
    },
    {
        "id": "alert_pullup_signal",
        "net": "BQ_ALERT_N",
        "start": ["R_ALERT_PU", "2"],
        "via": [[2910, 2020], [2910, 2040], [3020, 2040], [3020, 2020]],
        "end": ["R_ALERT_SER", "2"],
        "purpose": "绕过R_ALERT_SER信号侧与现有BQ_ALERT_IC_N线，连接两只电阻的BQ_ALERT_N端",
    },
]


# Remove only the duplicate local labels made redundant by the routes above.
REMOVE_PORT_PINS = [
    ("C_CAN_PWR_IN2", "1"),
    ("C_CAN_PWR_IN2", "2"),
    ("C_CAN_PWR_OUT2", "1"),
    ("C_CAN_PWR_OUT2", "2"),
    ("C_POR", "1"),
    ("C_POR", "2"),
    ("C_TIMER", "1"),
    ("C_TIMER", "2"),
    ("C_NTC_BOARD_FILT", "2"),
    ("NTC_BOARD", "2"),
    ("C_BQ_TS1_FILT", "2"),
    ("C_BQ_TS3_FILT", "1"),
    ("C_BQ_TS3_FILT", "2"),
    ("R_ALERT_SER", "2"),
]


ANCHOR_PORT_PINS = [
    ("C_CAN_PWR_IN1", "1"), ("C_CAN_PWR_IN1", "2"),
    ("C_CAN_PWR_OUT1", "1"), ("C_CAN_PWR_OUT1", "2"),
    ("U_AND2", "5"), ("U_AND2", "2"),
    ("U_GATE1", "5"), ("U_GATE1", "2"),
    ("U_POR", "3"), ("U_POR", "1"),
    ("U_TIMER", "1"), ("U_TIMER", "2"),
    ("NTC_MOS", "1"), ("NTC_MOS", "2"),
    ("R_ALERT_PU", "2"),
]


MOVE_REVIEW = [
    {
        "refs": ["C_CAN_PWR_IN1", "C_CAN_PWR_IN2", "U_CAN"],
        "reason": "输入电容对可本地并联，但其VDDP/GNDP1接到U_CAN.6/.5会与现有CAN_TX走线相交或贴近芯片符号边界；若要求直接短线，建议把电容对移到U_CAN左侧电源引脚附近。",
        "affectedPins": ["C_CAN_PWR_IN1.1", "C_CAN_PWR_IN1.2", "C_CAN_PWR_IN2.1", "C_CAN_PWR_IN2.2", "U_CAN.5", "U_CAN.6"],
    },
    {
        "refs": ["C_CAN_PWR_OUT1", "C_CAN_PWR_OUT2", "U_CAN"],
        "reason": "输出电容对可本地并联；连接U_CAN.10/.11/.12需绕过CANH走线树，当前坐标会形成较长绕线，建议电容移到隔离侧供电脚附近后再定路径。",
        "affectedPins": ["C_CAN_PWR_OUT1.1", "C_CAN_PWR_OUT1.2", "C_CAN_PWR_OUT2.1", "C_CAN_PWR_OUT2.2", "U_CAN.10", "U_CAN.11", "U_CAN.12"],
    },
    {
        "refs": ["C_CAN_VCC2_2", "U_CAN"],
        "reason": "单颗VDDL去耦电容离U_CAN.7/.8较远，连接B0侧会碰到CANH树；建议移到U_CAN的VDDL/GND1旁边，再画本地回路。",
        "affectedPins": ["C_CAN_VCC2_2.1", "C_CAN_VCC2_2.2", "U_CAN.7", "U_CAN.8"],
    },
    {
        "refs": ["C_SEQ", "U_SEQ", "U_GATE1", "U_TIMER"],
        "reason": "C_SEQ目前在TMR_ACTIVE总线下方；到U_SEQ供电脚的两条回路需绕过TMR_ACTIVE，并受MAIN_CLR_N/POR_OK_N路径约束。候选不强拉跨线，建议先把C_SEQ移到U_SEQ供电脚侧的空白处。",
        "affectedPins": ["C_SEQ.1", "C_SEQ.2", "U_SEQ.7", "U_SEQ.14"],
    },
    {
        "refs": ["C_GATE1", "U_GATE1", "U_SEQ", "U_POR"],
        "reason": "U_GATE1.5/6之间已有TMR_ACTIVE连接，且U_GATE1上方为POR_OK_N横线；C_GATE1在芯片下方。现位置无法用不跨异网的短回路接入VCC/GND，候选保留原网标，建议把电容移到芯片供电脚右侧空白区后再连。",
        "affectedPins": ["C_GATE1.1", "C_GATE1.2", "U_GATE1.5", "U_GATE1.2"],
    },
    {
        "refs": ["C_AND2", "U_AND2"],
        "reason": "C_AND2位于U_AND2上方；MAIN_CLR_N现有水平和垂直导线形成围挡，直接靠近供电脚的候选走线需穿越该异网或绕到芯片符号下侧，暂不新增长绕线，建议将电容移到U_AND2右下方空白处。",
        "affectedPins": ["C_AND2.1", "C_AND2.2", "U_AND2.5", "U_AND2.2"],
    },
    {
        "refs": ["C_BQ_TS1_FILT", "U1"],
        "reason": "C_BQ_TS1_FILT.1到U1.21的曼哈顿距离为900原理图单位；当前没有同区域传感器引脚，保留同名网标并建议把滤波电容移到U1.21附近。",
        "affectedPins": ["C_BQ_TS1_FILT.1", "U1.21"],
    },
    {
        "refs": ["NTC_MOS", "U1"],
        "reason": "NTC_MOS与C_BQ_TS3_FILT将在本地实连；U1.23仍以BQ_TS3网名接入，跨越约900单位的远线不在本候选内。",
        "affectedPins": ["NTC_MOS.1", "C_BQ_TS3_FILT.1", "U1.23"],
    },
    {
        "refs": ["R_ALERT_PU", "R_SCL_PU", "R_SDA_PU"],
        "reason": "R_ALERT_PU.1到相邻CORE_3V3上拉轨的短路径会穿过现有BQ_ALERT_IC_N水平线；保留CORE_3V3网标，建议后续移动R_ALERT_PU或重排该线后再接上拉轨。",
        "affectedPins": ["R_ALERT_PU.1", "R_SCL_PU.1", "R_SDA_PU.1"],
    },
]


ALREADY_PHYSICAL = [
    {
        "net": "NTC_BOARD_NODE",
        "pins": ["U1.22", "NTC_BOARD.1", "C_NTC_BOARD_FILT.1"],
        "reason": "strict verifier已确认这些针脚在当前保存态处于同一原生物理wire cluster；不重复加线。",
    }
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_verifier():
    spec = importlib.util.spec_from_file_location("strict_semantic_geometry", VERIFIER)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load verifier: {VERIFIER}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def key_of(pair: list[str]) -> tuple[str, str]:
    return str(pair[0]), str(pair[1])


def main() -> int:
    for path in (BASELINE, REFERENCE_BASELINE, MANIFEST, NATIVE, VERIFIER):
        if not path.is_file():
            raise FileNotFoundError(path)

    snapshot = json.loads(BASELINE.read_text(encoding="utf-8"))
    manifest_rows = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if not isinstance(manifest_rows, list):
        raise ValueError("current construction manifest must be a list")
    manifest = {str(row["ref"]): row for row in manifest_rows}
    snapshot_rows = {str(row["ref"]): row for row in snapshot.get("rows", [])}
    port_rows = list(snapshot.get("ports", []))
    ports_by_id = {str(port.get("id")): port for port in port_rows}
    pin_rows: dict[tuple[str, str], dict[str, Any]] = {}
    for ref, row in snapshot_rows.items():
        for pin in row.get("pins", []):
            pin_rows[(ref, str(pin.get("number")))] = pin

    verifier = load_verifier()
    current_check = verifier.collect_semantics(snapshot, snapshot, manifest_rows, NATIVE, [BASELINE, MANIFEST])
    # The parent has already changed the VC/SRP region, so the prior full-file
    # baseline is intentionally stale.  Continue only if every manifest pin
    # still has a passing expected-vs-physical result in the current capture.
    failed_pins = [row for row in current_check["pinResults"] if row.get("status") != "PASS"]
    if failed_pins:
        raise RuntimeError(
            "current capture has pin-level semantic defects; refusing candidate generation:\n"
            + json.dumps(failed_pins[:30], ensure_ascii=False, indent=2)
        )

    def expected_net(ref: str, number: str) -> str | None:
        return verifier.normalize_expected_net(manifest[ref]["pins"][number])

    def pin_point(pair: list[str]) -> tuple[float, float]:
        ref, number = key_of(pair)
        pin = pin_rows.get((ref, number))
        if pin is None:
            raise ValueError(f"missing snapshot pin {ref}.{number}")
        return verifier.point(pin["x"], pin["y"])

    def port_for_pin(ref: str, number: str) -> dict[str, Any]:
        net = expected_net(ref, number)
        if net is None:
            raise ValueError(f"cannot select a netport for NC pin {ref}.{number}")
        xy = pin_point([ref, number])
        matches = [
            port for port in port_rows
            if port.get("net") == net
            and any(verifier.point(pin["x"], pin["y"]) == xy for pin in port.get("pins", []))
        ]
        if len(matches) != 1:
            raise ValueError(f"expected exactly one active netport at {ref}.{number} ({net}), found {len(matches)}")
        return matches[0]

    removals: list[dict[str, Any]] = []
    removed_ids: set[str] = set()
    for ref, number in REMOVE_PORT_PINS:
        port = port_for_pin(ref, number)
        port_id = str(port["id"])
        if port_id in removed_ids:
            raise ValueError(f"duplicate port-removal request for {port_id}")
        removed_ids.add(port_id)
        removals.append({
            "id": port_id,
            "ref": ref,
            "pin": number,
            "net": expected_net(ref, number),
            "at": list(pin_point([ref, number])),
            "reason": "该引脚已由候选实线接入同名本地物理cluster；保留同net anchor，删除此重复localport。",
        })

    anchors: list[dict[str, Any]] = []
    for ref, number in ANCHOR_PORT_PINS:
        port = port_for_pin(ref, number)
        if str(port["id"]) in removed_ids:
            raise ValueError(f"retained anchor {ref}.{number} is also scheduled for removal")
        anchors.append({
            "id": str(port["id"]),
            "ref": ref,
            "pin": number,
            "net": expected_net(ref, number),
            "at": list(pin_point([ref, number])),
        })

    routes: list[dict[str, Any]] = []
    all_candidate_segments: list[dict[str, Any]] = []
    for recipe in ROUTES:
        start = recipe["start"]
        end = recipe["end"]
        points = [pin_point(start), *(verifier.point(*p) for p in recipe["via"])]
        end_is_pin = isinstance(end, list)
        if end_is_pin:
            points.append(pin_point(end))
            end_label = f"{end[0]}.{end[1]}"
        else:
            junction = end.get("junction")
            if not isinstance(junction, list) or len(junction) != 2:
                raise ValueError(f"invalid endpoint for {recipe['id']}")
            points.append(verifier.point(*junction))
            end_label = f"junction@{junction[0]},{junction[1]}"

        for index, (a, b) in enumerate(zip(points, points[1:]), 1):
            if abs(a[0] - b[0]) > verifier.EPS and abs(a[1] - b[1]) > verifier.EPS:
                raise ValueError(f"non-orthogonal segment in {recipe['id']}: {a} -> {b}")
            if a == b:
                raise ValueError(f"zero-length segment in {recipe['id']} at {a}")
            all_candidate_segments.append({
                "route": recipe["id"], "net": recipe["net"], "segmentIndex": index,
                "start": a, "end": b,
            })
        routes.append({
            "id": recipe["id"],
            "net": recipe["net"],
            "start": f"{start[0]}.{start[1]}",
            "end": end_label,
            "purpose": recipe["purpose"],
            "points": [list(p) for p in points],
            "segments": [[a[0], a[1], b[0], b[1]] for a, b in zip(points, points[1:])],
            "manhattanLength": int(sum(abs(a[0] - b[0]) + abs(a[1] - b[1]) for a, b in zip(points, points[1:]))),
        })

    # Check each proposed segment against all manifest pins and netports.
    features: list[dict[str, Any]] = []
    for ref, row in snapshot_rows.items():
        for pin in row.get("pins", []):
            number = str(pin["number"])
            features.append({
                "kind": "componentPin", "ref": ref, "pin": number,
                "point": verifier.point(pin["x"], pin["y"]),
                "net": expected_net(ref, number),
            })
    for port in port_rows:
        for pin in port.get("pins", []):
            features.append({
                "kind": "netport", "ref": str(port.get("id")), "pin": str(pin.get("number")),
                "point": verifier.point(pin["x"], pin["y"]),
                "net": str(port.get("net") or "").strip() or None,
            })

    native_lines, _, line_group_names, fold_info, _ = verifier.fold_native(NATIVE)
    if fold_info["malformed"]:
        raise RuntimeError("native fold has malformed records; refusing candidate generation")
    existing_lines = []
    for line in native_lines:
        if line["start"] == line["end"]:
            continue
        names = set(line_group_names.get(line["lineGroup"], set()))
        existing_lines.append({**line, "names": names})

    geometry_errors: list[dict[str, Any]] = []
    for segment in all_candidate_segments:
        a, b, net = segment["start"], segment["end"], segment["net"]
        for feature in features:
            if verifier.point_on_segment(feature["point"], a, b) and feature["net"] != net:
                geometry_errors.append({
                    "route": segment["route"], "segment": segment["segmentIndex"],
                    "type": "foreign_pin_or_port", "feature": f"{feature['kind']}:{feature['ref']}.{feature['pin']}",
                    "featureNet": feature["net"], "routeNet": net, "point": list(feature["point"]),
                })
        for line in existing_lines:
            contacts = verifier.endpoint_contacts(a, b, line["start"], line["end"])
            crossing = None if contacts else verifier.strict_interior_intersection(a, b, line["start"], line["end"])
            if contacts or crossing is not None:
                existing_net = next(iter(line["names"])) if len(line["names"]) == 1 else None
                contact_points = contacts or {crossing}
                # Even same-net crossings are surfaced; only same-net endpoint contacts are accepted.
                endpoint_contact = bool(contacts) and all(p in {a, b} for p in contacts)
                if not (existing_net == net and endpoint_contact):
                    geometry_errors.append({
                        "route": segment["route"], "segment": segment["segmentIndex"],
                        "type": "existing_wire_contact_or_crossing", "existingLine": line["id"],
                        "existingNet": sorted(line["names"]), "routeNet": net,
                        "points": [list(p) for p in contact_points],
                    })

    # Candidate-to-candidate intersections across unlike nets would short if connected at a vertex.
    for i, left in enumerate(all_candidate_segments):
        for right in all_candidate_segments[i + 1:]:
            if left["route"] == right["route"] or left["net"] == right["net"]:
                continue
            contacts = verifier.endpoint_contacts(left["start"], left["end"], right["start"], right["end"])
            crossing = None if contacts else verifier.strict_interior_intersection(
                left["start"], left["end"], right["start"], right["end"]
            )
            if contacts or crossing is not None:
                geometry_errors.append({
                    "route": left["route"], "otherRoute": right["route"],
                    "type": "candidate_foreign_wire_contact_or_crossing",
                    "points": [list(p) for p in (contacts or {crossing})],
                    "net": left["net"], "otherNet": right["net"],
                })

    # Every route endpoint and its named pins must agree with the manifest net.
    for recipe in ROUTES:
        for endpoint in (recipe["start"], recipe["end"]):
            if isinstance(endpoint, list):
                found = expected_net(str(endpoint[0]), str(endpoint[1]))
                if found != recipe["net"]:
                    geometry_errors.append({
                        "route": recipe["id"], "type": "manifest_net_mismatch",
                        "endpoint": f"{endpoint[0]}.{endpoint[1]}",
                        "expected": found, "routeNet": recipe["net"],
                    })

    if geometry_errors:
        raise RuntimeError("candidate geometry did not pass safety checks:\n" + json.dumps(geometry_errors, ensure_ascii=False, indent=2))

    pin_results = {(row["ref"], str(row["pin"])): row for row in current_check["pinResults"]}
    existing_connections = []
    for item in ALREADY_PHYSICAL:
        rows = [pin_results[(ref.rsplit(".", 1)[0], ref.rsplit(".", 1)[1])] for ref in item["pins"]]
        clusters = {row["cluster"] for row in rows}
        names = {tuple(row["actualNames"]) for row in rows}
        if len(clusters) != 1 or names != {(item["net"],)}:
            raise RuntimeError(f"claimed existing connection is not confirmed: {item}")
        existing_connections.append({
            **item,
            "cluster": next(iter(clusters)),
            "actualNames": list(next(iter(names))),
        })

    output = {
        "title": "BMS局部控制与去耦连线候选",
        "status": "candidate-not-applied",
        "source": {
        "currentSavedSnapshot": str(BASELINE),
        "priorBaselineSnapshot": str(REFERENCE_BASELINE),
            "constructionManifest": str(MANIFEST),
            "nativeForStrictGeometryOnly": str(NATIVE),
            "sha256": {
                "currentSavedSnapshot": sha256(BASELINE),
                "priorBaselineSnapshot": sha256(REFERENCE_BASELINE),
                "constructionManifest": sha256(MANIFEST),
                "native": sha256(NATIVE),
            },
        },
        "currentPinSemanticCheck": {
            "status": "PASS",
            "pinsChecked": len(current_check["pinResults"]),
            "method": "verify_semantic_geometry.collect_semantics against latest saved-state capture and current native geometry",
            "wholeDocumentStatus": current_check["status"],
            "nonPinDefectCounts": {
                key: value for key, value in current_check["defectCounts"].items()
                if key not in {"floatingExpectedPins", "pinNetMismatches", "ncContact"} and value
            },
        },
        "alreadyPhysicalConnections": existing_connections,
        "addWires": routes,
        "removeLocalports": removals,
        "retainedAnchorPorts": anchors,
        "needsMoveOrReview": MOVE_REVIEW,
        "geometryCheck": {
            "status": "PASS",
            "candidateSegments": len(all_candidate_segments),
            "foreignPinOrPortContacts": 0,
            "foreignExistingWireContactsOrCrossings": 0,
            "foreignCandidateWireContactsOrCrossings": 0,
            "orthogonal": True,
            "note": "检查以native fold后的非零LINE为基准；端点pin/net与manifest一致。原生符号外形未作碰撞建模，正式套用前仍应使用官方API局部截图复核。",
        },
        "notApplied": True,
    }

    output_path = DEFAULT_OUTPUT
    if output_path.resolve().parent != HERE.resolve():
        raise RuntimeError("output path must stay beside this read-only planner")
    output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(output_path),
        "status": output["status"],
        "currentPinSemanticCheck": "PASS",
        "addWires": len(routes),
        "candidateSegments": len(all_candidate_segments),
        "removeLocalports": len(removals),
        "needsMoveOrReview": len(MOVE_REVIEW),
        "geometryCheck": output["geometryCheck"]["status"],
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
