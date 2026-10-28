"""Read-only strict geometry check against the current published BMS schematic.

The native .esch2 is append-record storage. For document-global objects used by
this check (LINE/WIRE/TEXT/placed COMPONENT and NET ATTR), the latest record by
(type, id) wins; an empty payload is a deletion tombstone. Component-library
primitive IDs such as POLY/e0 are scoped to partId and deliberately are not
folded into document-global objects.
"""
from pathlib import Path
import hashlib
import importlib.util
import json
import sys

P = Path(__file__).resolve().parent
E = P / 'Layout验证'
native = Path(sys.argv[1]) if len(sys.argv) > 1 else P.parent / 'sch/Schematic1/P1.esch2'
snapshot = Path(sys.argv[2]) if len(sys.argv) > 2 else E / 'current-snapshot.json'
portsfile = Path(sys.argv[3]) if len(sys.argv) > 3 else E / 'ports-with-cleanup-whitelist.json'
cleanupfile = Path(sys.argv[4]) if len(sys.argv) > 4 else E / 'cleanup-manifest.json'
output = Path(sys.argv[5]) if len(sys.argv) > 5 else E / 'final-geometry.json'

GLOBAL_TYPES = {'LINE', 'WIRE', 'TEXT', 'COMPONENT'}


def parse_native(path):
    latest = {}
    malformed = []
    raw_count = 0
    tombstones = 0
    updates = 0
    for line_number, line in enumerate(path.read_text(encoding='utf8').splitlines(), 1):
        raw_count += 1
        try:
            header_text, payload_text = line.split('||', 1)
            header = json.loads(header_text)
            payload_text = payload_text.rstrip('|')
            payload = json.loads(payload_text) if payload_text.strip() else None
        except (ValueError, json.JSONDecodeError) as exc:
            malformed.append({'line': line_number, 'error': str(exc)})
            continue

        kind, record_id = header.get('type'), header.get('id')
        if kind not in GLOBAL_TYPES and kind != 'ATTR':
            continue
        if not record_id:
            malformed.append({'line': line_number, 'error': 'relevant record missing id', 'type': kind})
            continue
        if kind == 'ATTR' and payload is not None and payload.get('key') != 'NET':
            continue
        # Native document entities use globally unique IDs. ATTR/NET IDs are
        # likewise unique among NET records; unrelated symbol-local ATTR IDs
        # are intentionally excluded above.
        key = (kind, record_id)
        if payload is None:
            tombstones += 1
            latest.pop(key, None)
        else:
            if key in latest:
                updates += 1
            latest[key] = {'header': header, 'payload': payload, 'line': line_number}
    return latest, {'raw_records': raw_count, 'folded_records': len(latest), 'relevant_tombstones': tombstones,
                    'relevant_replacements': updates, 'malformed': malformed}


latest, fold_summary = parse_native(native)
active_lines = [r['payload'] for (t, _), r in latest.items() if t == 'LINE']
active_wires = [r['payload'] for (t, _), r in latest.items() if t == 'WIRE']
active_text = {record_id: r['payload'] for (t, record_id), r in latest.items() if t == 'TEXT'}
active_components = {record_id: r['payload'] for (t, record_id), r in latest.items() if t == 'COMPONENT'}
net_attrs = [r['payload'] for (t, _), r in latest.items() if t == 'ATTR']

net_names_by_group = {}
conflicting_group_nets = []
for attr in net_attrs:
    group = attr.get('parentId')
    if not group:
        continue
    name = attr.get('value', '')
    prior = net_names_by_group.setdefault(group, name)
    if prior != name:
        conflicting_group_nets.append({'lineGroup': group, 'values': [prior, name]})

wires = [
    {'net': net_names_by_group[a['lineGroup']], 'line': [a['startX'], a['startY'], a['endX'], a['endY']]}
    for a in active_lines if a.get('lineGroup') in net_names_by_group
]

snapshot_doc = json.loads(snapshot.read_text(encoding='utf8'))
snapshot_rows = snapshot_doc['rows']
snapshot_ids = {row['id'] for row in snapshot_rows}
missing_snapshot_ids = sorted(snapshot_ids - active_components.keys())
extra_snapshot_component_ids = sorted(active_components.keys() - snapshot_ids)

removed_refs = {'J_FAN', 'J_FAN_CTRL', 'Q_FAN_PWM', 'R_FAN_PWM_GATE', 'R_FAN_PWM_PD', 'R_FAN_TACH_PU'}
EXPECTED_DELETED_ISOLATED_PORT_IDS = {
    # Four old fan ports and five isolated non-fan ports; CAD explicitly
    # confirmed all nine port deletions are in the approved cleanup whitelist.
    '0d2c81e4ee537e9f', '19eda31157b13543', '3900d12f0143e431', '318750167a99ce89',
    'e7f034638a72c9a8', 'f520f8a7c42f4632', '40b950ded73e037a',
    'e48735dd2c089039', 'ea84fc4fc45d5f1f',
}
manifest = json.loads((P / '当前施工输入.json').read_text(encoding='utf8'))

# The port source predates final cleanup. Netport symbols are not part of the
# 207-row API component snapshot, so verify them against active native IDs.
ports_doc = json.loads(portsfile.read_text(encoding='utf8'))
active_port_rows = [row for row in ports_doc['ports'] if row['id'] in active_components]
inactive_port_rows = [row for row in ports_doc['ports'] if row['id'] not in active_components]
unexpected_inactive_port_rows = [
    row for row in inactive_port_rows
    if row['id'] not in EXPECTED_DELETED_ISOLATED_PORT_IDS
]
expected_inactive_port_rows = [row for row in inactive_port_rows if row['id'] in EXPECTED_DELETED_ISOLATED_PORT_IDS]
labels = [
    {'net': row['net'], 'x': row['pins'][0]['x'], 'y': row['pins'][0]['y']}
    for row in active_port_rows
]

spec = importlib.util.spec_from_file_location('checkgeom', P / '校验逻辑绘图.py')
checker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(checker)
result = checker.check({'wires': wires, 'labels': labels}, snapshot_doc, manifest)

cleanup = json.loads(cleanupfile.read_text(encoding='utf8'))
removed_text_ids = {row['id'] for row in cleanup.get('removed', [])}
active_removed_text_ids = sorted(removed_text_ids & active_text.keys())
active_removed_text_details = [
    {'id': record_id, 'x': active_text[record_id].get('x'), 'y': active_text[record_id].get('y'),
     'value': active_text[record_id].get('value')}
    for record_id in active_removed_text_ids
]
snapshot_refs = {row['ref'] for row in snapshot_rows}
active_fan_refs = sorted(ref for ref in snapshot_refs if 'FAN' in ref.upper())
manifest_refs = {row['ref'] for row in manifest}
manifest_snapshot_mismatch = {
    'missing_from_manifest': sorted(snapshot_refs - manifest_refs),
    'missing_from_snapshot': sorted(manifest_refs - snapshot_refs),
}

source_files = (native, snapshot, portsfile, cleanupfile)
result.update({
    'status': 'PASS' if (
        not any(result[k] for k in ('nonOrthogonal', 'missingPins', 'foreignNetAtPins', 'foreignNetAtWireEndpoints', 'danglingLabels'))
        and not missing_snapshot_ids
        and not active_removed_text_ids
        and not active_fan_refs
        and not unexpected_inactive_port_rows
        and len(inactive_port_rows) == 9
        and len(expected_inactive_port_rows) == 9
        and not conflicting_group_nets
        and not fold_summary['malformed']
        and not any(manifest_snapshot_mismatch.values())
    ) else 'FAIL',
    'electrical_geometry_status': 'PASS' if not any(
        result[k] for k in ('nonOrthogonal', 'missingPins', 'foreignNetAtPins', 'foreignNetAtWireEndpoints', 'danglingLabels')
    ) else 'FAIL',
    'strict_current_geometry': True,
    'active_record_summary': {
        'fold': fold_summary,
        'active_LINE': len(active_lines),
        'active_WIRE': len(active_wires),
        'active_TEXT': len(active_text),
        'active_COMPONENT_records': len(active_components),
        'snapshot_placed_components': len(snapshot_ids),
        'snapshot_ids_missing_from_native_COMPONENT': missing_snapshot_ids,
        'native_COMPONENT_ids_outside_snapshot': len(extra_snapshot_component_ids),
        'net_ATTR': len(net_attrs),
        'line_groups_with_net_ATTR': len(net_names_by_group),
        'active_lines_without_net_ATTR': sum(1 for a in active_lines if a.get('lineGroup') not in net_names_by_group),
        'conflicting_line_group_nets': conflicting_group_nets,
    },
    'cleanup_text_check': {
        'expected_removed_count': cleanup.get('expected_count'),
        'reported_removed_count': cleanup.get('removed_count'),
        'removed_id_count': len(removed_text_ids),
        'still_active_removed_ids': active_removed_text_ids,
        'still_active_removed_details': active_removed_text_details,
        'still_active_removed_count': len(active_removed_text_ids),
    },
    'precleanup_port_source_check': {
        'source_port_count': len(ports_doc['ports']),
        'active_native_port_count': len(active_port_rows),
        'inactive_source_port_count': len(inactive_port_rows),
        'approved_deletion_whitelist': {
            'confirmed_by': 'Documented isolated-port cleanup whitelist; no removed active circuit connections',
            'expected_count': 9,
            'observed_expected_count': len(expected_inactive_port_rows),
            'all_inactive_ids_whitelisted': len(expected_inactive_port_rows) == len(inactive_port_rows),
            'explanation': 'All nine are isolated netport symbols removed under the CAD-approved whitelist: four obsolete FAN ports and five non-FAN ports (CORE_3V3, VOUT_5V, and three B0 ports). They do not add a geometry FAIL when all nine IDs match this explicit whitelist.',
        },
        'inactive_expected_isolated_ports': [
            {'id': row['id'], 'name': row.get('name'), 'net': row.get('net'), 'x': row.get('x'), 'y': row.get('y')}
            for row in expected_inactive_port_rows
        ],
        'inactive_expected_nonfan_ports': [
            {'id': row['id'], 'name': row.get('name'), 'net': row.get('net'), 'x': row.get('x'), 'y': row.get('y')}
            for row in expected_inactive_port_rows
            if 'FAN' not in str(row.get('name', '')).upper() and 'FAN' not in str(row.get('net', '')).upper()
        ],
        'inactive_unexpected_ports': [
            {'id': row['id'], 'name': row.get('name'), 'net': row.get('net'), 'x': row.get('x'), 'y': row.get('y')}
            for row in unexpected_inactive_port_rows
        ],
    },
    'active_fan_refs_in_final_snapshot': active_fan_refs,
    'manifest_vs_final_snapshot_refs': manifest_snapshot_mismatch,
    'sourceSHA256': {str(f): hashlib.sha256(f.read_bytes()).hexdigest() for f in source_files},
    'method': 'Fold latest global document records by (type,id); empty payload tombstones remove the preceding entity. Symbol-library IDs are local to partId and are excluded from document geometry. Native LINE primitives are individual segments; Netport symbols are verified against active native COMPONENT IDs because the reopened component snapshot contains BOM components only.',
})
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf8')
print(json.dumps({
    'status': result['status'],
    'counts': result['counts'],
    'active_record_summary': result['active_record_summary'],
    'cleanup_text_check': result['cleanup_text_check'],
    'precleanup_port_source_check': result['precleanup_port_source_check'],
    'active_fan_refs_in_final_snapshot': active_fan_refs,
    'manifest_vs_final_snapshot_refs': manifest_snapshot_mismatch,
    'defect_counts': {k: len(result[k]) for k in ('nonOrthogonal', 'missingPins', 'foreignNetAtPins', 'foreignNetAtWireEndpoints', 'danglingLabels')},
    'output': str(output),
}, ensure_ascii=False, indent=2))
if result['status'] != 'PASS':
    sys.exit(1)
