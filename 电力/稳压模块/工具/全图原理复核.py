"""按当前ENET逐脚核对设计合同，并记录完整功能审查；不修改EDA源/PCB。"""
from pathlib import Path
import csv
import hashlib
import json
import math
import re
import zipfile

M = Path(__file__).resolve().parents[1]
O = M / '检查'
POWER = M.parent
def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()
def load(p):
    return json.loads(p.read_text(encoding='utf-8-sig'))
def sha_bytes(payload):
    return hashlib.sha256(payload).hexdigest()
def power_path(relative):
    p = Path(relative)
    assert not p.is_absolute() and '..' not in p.parts, relative
    parts = list(p.parts)
    if parts and parts[0] == POWER.name:
        parts = parts[1:]
    return POWER.joinpath(*parts)
def archive_payload(zf, relative, expected_sha256=None):
    wanted = Path(relative).as_posix().replace('\\', '/')
    if wanted.startswith(POWER.name + '/'):
        wanted = wanted[len(POWER.name) + 1:]
    exact = [n for n in zf.namelist() if n.replace('\\', '/') == wanted]
    matches = exact or [n for n in zf.namelist() if n.replace('\\', '/').endswith('/' + wanted)]
    if expected_sha256 is not None and len(matches) > 1:
        matches = [n for n in matches if sha_bytes(zf.read(n)) == expected_sha256]
    assert len(matches) == 1, (relative, matches)
    return matches[0], zf.read(matches[0])
def verify_file_bindings(zf, records):
    verified = {}
    for record in records:
        path = record['path']
        member, payload = archive_payload(zf, path, record['sha256'])
        digest = sha_bytes(payload)
        assert digest == record['sha256'], path
        assert len(payload) == record['bytes'], path
        verified[path] = {'archiveEntry':member, 'sha256':digest, 'bytes':len(payload)}
    return verified
def parse_native_records(payload):
    records = {}
    in_schematic_page = False
    found_schematic_page = False
    for line in payload.splitlines():
        if b'||' not in line:
            continue
        head, body = line.split(b'||', 1)
        meta = json.loads(head.decode('utf-8'))
        kind, record_id = meta['type'], meta.get('id', '')
        if kind == 'DOCHEAD':
            doc_head = json.loads(body.rstrip(b'|').decode('utf-8'))
            if in_schematic_page:
                break
            if doc_head.get('docType') == 'SCH_PAGE':
                assert not found_schematic_page, 'multiple schematic-page sections in one source file'
                in_schematic_page = True
                found_schematic_page = True
            continue
        if not in_schematic_page:
            continue
        body = body.rstrip(b'|')
        if not body:
            # EasyEDA retains empty placeholder records for deleted/unused objects.
            continue
        key = (kind, record_id, meta.get('ticket'))
        parsed = json.loads(body.decode('utf-8'))
        assert key not in records, key
        records[key] = parsed
    assert found_schematic_page, 'native source has no SCH_PAGE document section'
    return records
def native_components(payload):
    records = parse_native_records(payload)
    ref_by_id = {}
    for (kind, _, _), body in records.items():
        if kind == 'ATTR' and body.get('key') == 'Designator':
            parent = body.get('parentId')
            value = body.get('value')
            if parent and value:
                assert parent not in ref_by_id or ref_by_id[parent] == value
                ref_by_id[parent] = value
    components = {}
    for (kind, component_id, _), body in records.items():
        if kind == 'COMPONENT' and component_id in ref_by_id:
            ref = ref_by_id[component_id]
            assert ref not in components
            components[ref] = body
    return records, components
def live_pin_rows(live_rows):
    refs = [row.get('ref') for row in live_rows]
    assert all(refs) and len(refs) == len(set(refs))
    return {row['ref']: row for row in live_rows}
def procurement_ntc_rows(path, expected_by_ref):
    with path.open('r', encoding='utf-8-sig', newline='') as f:
        records = list(csv.DictReader(f))
    assert records, path
    required = ('Reference','Quantity','Manufacturer part','Supplier part','Convert to PCB')
    assert all(name in records[0] for name in required), records[0].keys()
    found = {ref:[] for ref in expected_by_ref}
    for record in records:
        refs = [ref.strip() for ref in str(record.get('Reference') or '').replace(';', ',').split(',') if ref.strip()]
        for ref in found:
            if ref in refs:
                found[ref].append((record, refs))
    proof = {}
    for ref, matches in found.items():
        assert len(matches) == 1, (ref, matches)
        record, refs = matches[0]
        expected = expected_by_ref[ref]
        quantity = int(str(record['Quantity']).strip())
        assert quantity == len(refs), (ref, quantity, refs)
        assert record['Manufacturer part'].strip() == str(expected['mpn']), (ref, record['Manufacturer part'], expected['mpn'])
        assert record['Supplier part'].strip() == str(expected['supplierId']), (ref, record['Supplier part'], expected['supplierId'])
        assert record['Convert to PCB'].strip().lower() in ('false','no','0'), (ref,record['Convert to PCB'])
        proof[ref] = {'quantity':1,'groupQuantity':quantity,'references':refs,
                      'manufacturerPart':record['Manufacturer part'].strip(),
                      'supplierPart':record['Supplier part'].strip(),
                      'convertToPcb':record['Convert to PCB'].strip()}
    return proof
def drc_groups(drc_records):
    assert isinstance(drc_records, list), 'current official verification must preserve raw DRC array'
    errors, warnings = [], []
    for record in drc_records:
        assert isinstance(record, dict), record
        severity = str(record.get('type') or record.get('severity') or record.get('level') or '').lower()
        if severity in ('error', 'err', 'fatal'):
            errors.append(record)
        elif severity in ('warn', 'warning'):
            warnings.append(record)
        else:
            raise AssertionError(f'unknown official DRC severity: {record!r}')
    return errors, warnings
def check_geometry_bridges(native_before, native_after, current_rows, current_enet,
                           bridge_response, refs, expected_count, expected_wire_persistence=None):
    before_records, _ = native_components(native_before)
    after_records, _ = native_components(native_after)
    before_lines = {record_id: body for (kind, record_id, _), body in before_records.items() if kind == 'LINE'}
    after_lines = {record_id: body for (kind, record_id, _), body in after_records.items() if kind == 'LINE'}
    added = set(after_lines) - set(before_lines)
    deleted = set(before_lines) - set(after_lines)
    changed = {record_id for record_id in set(before_lines) & set(after_lines)
               if before_lines[record_id] != after_lines[record_id]}
    persistence = {'baselineLinePrimitives':len(before_lines),
                   'currentLinePrimitives':len(after_lines),
                   'addedLinePrimitives':len(added), 'deletedLinePrimitives':len(deleted),
                   'changedLinePrimitives':len(changed),
                   'existingConnectionSegmentsVerified':0}

    assert bridge_response.get('ok') is True, bridge_response
    bridges = bridge_response.get('value', {}).get('bridges')
    assert isinstance(bridges, list) and len(bridges) == expected_count, bridges
    expected_pairs = {(ref, str(pin.get('no') or pin.get('number')))
                      for ref in refs for pin in current_rows[ref].get('pins', [])}
    assert {(b.get('ref'), str(b.get('no'))) for b in bridges} == expected_pairs
    verified = []
    for bridge in bridges:
        ref, number = bridge['ref'], str(bridge['no'])
        pin = next(p for p in current_rows[ref].get('pins', [])
                   if str(p.get('no') or p.get('number')) == number)
        start, end = bridge['from'], bridge['to']
        assert len(start) == len(end) == 2
        x, y = pin.get('x'), pin.get('y')
        assert math.isclose(float(x), float(end[0]), rel_tol=0, abs_tol=1e-6)
        assert math.isclose(float(y), float(end[1]), rel_tol=0, abs_tol=1e-6)
        length_units = math.hypot(float(start[0]) - float(end[0]),
                                  float(start[1]) - float(end[1]))
        assert math.isclose(length_units, 10.0, rel_tol=0, abs_tol=1e-6), bridge
        enet_pin = current_enet[ref]['pinInfoMap'][number]
        assert str(enet_pin.get('net') or '') == str(bridge['net'] or ''), (bridge, enet_pin)
        group = bridge['id']

        def same_segment(line):
            a = (line.get('startX'), line.get('startY'))
            b = (line.get('endX'), line.get('endY'))
            return ((math.isclose(float(a[0]),float(start[0]),abs_tol=1e-6) and
                     math.isclose(float(a[1]),float(start[1]),abs_tol=1e-6) and
                     math.isclose(float(b[0]),float(end[0]),abs_tol=1e-6) and
                     math.isclose(float(b[1]),float(end[1]),abs_tol=1e-6)) or
                    (math.isclose(float(b[0]),float(start[0]),abs_tol=1e-6) and
                     math.isclose(float(b[1]),float(start[1]),abs_tol=1e-6) and
                     math.isclose(float(a[0]),float(end[0]),abs_tol=1e-6) and
                     math.isclose(float(a[1]),float(end[1]),abs_tol=1e-6)))

        current_matches = [(line_id,line) for line_id,line in after_lines.items()
                           if line.get('lineGroup') == group and same_segment(line)]
        baseline_matches = [(line_id,line) for line_id,line in before_lines.items()
                            if line.get('lineGroup') == group and same_segment(line)]
        assert len(current_matches) == len(baseline_matches) == 1, (bridge,current_matches,baseline_matches)
        line_id, line = current_matches[0]
        to_point = (float(end[0]), float(end[1]))
        network_contacts = [other_id for other_id, other in after_lines.items()
                            if other_id != line_id and other.get('lineGroup') == group and
                            any(math.isclose(float(point[0]),to_point[0],rel_tol=0,abs_tol=1e-6) and
                                math.isclose(float(point[1]),to_point[1],rel_tol=0,abs_tol=1e-6)
                                for point in ((other.get('startX'),other.get('startY')),
                                              (other.get('endX'),other.get('endY')))
                                if point[0] is not None and point[1] is not None)]
        assert network_contacts, (bridge, 'no continuation in the same native wire group at the current pin')
        verified.append({'ref':ref,'pin':number,'net':bridge['net'],'wireGroup':group,
                         'lineId':line_id,'from':start,'to':end,
                         'lengthSchematicUnits':length_units,'lengthMm':length_units*0.254,
                         'currentPinNetMatchesBridge':True,'networkContactLineIds':network_contacts,
                         'segmentAlreadyPresentInBaseline':True})
    persistence['existingConnectionSegmentsVerified'] = len(verified)
    persistence['schematicUnitMm'] = 0.254
    assert len(verified) == expected_count
    if expected_wire_persistence is not None:
        for field, expected in expected_wire_persistence.items():
            if field in persistence:
                assert persistence[field] == expected, (field, persistence[field], expected)
    return verified, persistence

plan = load(M/'原理图准备/design-plan.json')
material_proof_path = O/'原理整改/PEH226NL80V_变更证明.json'
material_proof = load(material_proof_path)
ad01_proof_path = O/'原理整改/AD01_分压修正.json'
ad01_proof = load(ad01_proof_path)
native_page = next((M/'eda工程文件/sc-regulator').rglob('*.esch2'))
previous_review = load(O/'全功能与逐脚复核.json')
previous_io02 = next(f for f in previous_review['findings'] if f['id'] == 'IO02')
busfix_model_update = previous_io02['modelUpdate']
busfix_input_path = M/'仿真/输入/V1.6-BUSFIX/model_input.json'
busfix_input = load(busfix_input_path)
assert (busfix_input['series_cells'], busfix_input['parallel_cells_per_group'],
        busfix_input['cell_count'], busfix_input['cell_capacitance_min_EOL_F']) == (8,1,8,280.0)
assert all(value is False for value in busfix_input['qualification_gate'].values())
assert busfix_model_update['testCount'] == '27/27'
assert busfix_model_update['persistedQualificationGatesAllFalse'] is True
# Preserve the earlier PEH226 proof as an intermediate baseline; it is not the current native hash.
peh_archive = O/'原理整改/PEH226NL80V_修改前源.zip'
assert sha(peh_archive) == material_proof['backup']['sha256']
with zipfile.ZipFile(peh_archive) as z:
    peh_members = [n for n in z.namelist() if n.endswith(native_page.name)]
    assert len(peh_members) == 1
    peh_before = z.read(peh_members[0])
assert hashlib.sha256(peh_before).hexdigest() == material_proof['nativeProof']['beforeSha256']

baseline_binding = ad01_proof['baseline']
after_binding = ad01_proof['after']
ad01_source_zip = M / baseline_binding['sourceZip']['path']
assert sha(ad01_source_zip) == baseline_binding['sourceZip']['sha256']
entry = baseline_binding['sourceZip']['nativeEntry'].replace('\\', '/')
with zipfile.ZipFile(ad01_source_zip) as z:
    baseline_members = [n for n in z.namelist()
                        if n.replace('\\', '/') == entry or n.replace('\\', '/').endswith('/' + entry)]
    assert len(baseline_members) == 1, baseline_members
    ad01_baseline = z.read(baseline_members[0])
    baseline_aux = {}
    for rel, expected_sha in [('原理图准备/BOM.csv', baseline_binding['bomSha256']),
                              ('原理图准备/design-plan.json', baseline_binding['designPlanSha256'])]:
        matches = [n for n in z.namelist() if n.replace('\\', '/') == rel or n.replace('\\', '/').endswith('/' + rel)]
        assert len(matches) == 1, (rel, matches)
        payload = z.read(matches[0])
        assert hashlib.sha256(payload).hexdigest() == expected_sha, rel
        baseline_aux[rel] = {'archiveEntry':matches[0], 'sha256':hashlib.sha256(payload).hexdigest(), 'bytes':len(payload)}
assert hashlib.sha256(ad01_baseline).hexdigest() == baseline_binding['nativeSha256']
assert baseline_binding['nativeSha256'] == material_proof['nativeProof']['afterSha256']
assert baseline_binding['bomSha256'] == material_proof['bomProof']['afterSha256']
assert baseline_binding['designPlanSha256'] == material_proof['planProof']['afterSha256']

# Reproduce the 48-record PEH attribute proof against the AD01 before-source ZIP,
# not against the later current schematic.
old_lines, peh_lines = peh_before.splitlines(keepends=True), ad01_baseline.splitlines(keepends=True)
assert len(old_lines) == len(peh_lines) == material_proof['nativeProof']['lineCountAfter']
allowed = {(r['parentId'], r['key']) for r in material_proof['changedNativeAttributes']}
changed_count = 0
for old, intermediate in zip(old_lines, peh_lines):
    if old == intermediate: continue
    old_header, old_body = old.decode('utf-8').split('||', 1)
    new_header, new_body = intermediate.decode('utf-8').split('||', 1)
    assert old_header == new_header and json.loads(new_header)['type'] == 'ATTR'
    a, b = json.loads(old_body.rstrip('\r\n|')), json.loads(new_body.rstrip('\r\n|'))
    assert (a['parentId'], a['key']) in allowed
    assert {k:v for k,v in a.items() if k != 'value'} == {k:v for k,v in b.items() if k != 'value'}
    changed_count += 1
assert changed_count == material_proof['nativeProof']['changedAttributeRecords'] == 48
ad01_official = ad01_proof['officialVerification']
assert ad01_official['sourceNativeSha256'] == after_binding['nativeSha256']
assert ad01_official['savedReopened'] is True and ad01_official['sessionClosed'] is True
assert ad01_official['liveApiComponentCount'] == ad01_official['componentCount'] == 167
assert ad01_official['liveApiPinCount'] == ad01_official['pinCount'] == 452
assert ad01_official['liveApiMismatches'] == [] and ad01_official['contractMismatches'] == []
assert ad01_official['drc']['strict'] is True and ad01_official['drc']['errors'] == []
ad01_enetpath = M / ad01_official['enetPath']
ad01_enetpath_current_sha = sha(ad01_enetpath) if ad01_enetpath.exists() else None
ad01_raw_bundle = ad01_official['rawResponseZip']
ad01_raw_bundle_path = M / ad01_raw_bundle['path']
assert sha(ad01_raw_bundle_path) == ad01_raw_bundle['sha256']
ad01_raw_reference = ad01_official['drc']['rawResponsePath']
assert ad01_raw_reference.startswith(ad01_raw_bundle['path'] + '!/')
ad01_raw_member = ad01_raw_reference.split('!/', 1)[1]
with zipfile.ZipFile(ad01_raw_bundle_path) as z:
    assert ad01_raw_member in z.namelist(), ad01_raw_member
    ad01_raw_api_bytes = z.read(ad01_raw_member)
    ad01_raw_api = json.loads(ad01_raw_api_bytes)
    ad01_save_raw = json.loads(z.read('ad01-formal-save.json'))
    ad01_close_raw = json.loads(z.read('ad01-formal-close-before-reopen.json'))
    ad01_reopen_raw = json.loads(z.read('ad01-formal-reopen.json'))
    ad01_final_close_raw = json.loads(z.read('ad01-formal-final-close.json'))
ad01_raw_value = ad01_raw_api['value']
assert ad01_raw_api.get('ok') is True
assert ad01_save_raw.get('ok') is True and ad01_save_raw['value']['saved'] is True
assert ad01_close_raw.get('ok') is True and ad01_close_raw['value']['status'] == 'closed'
assert ad01_reopen_raw.get('ok') is True and Path(ad01_reopen_raw['value']['path']).resolve() == (M/'eda工程文件/sc-regulator/sc-regulator.eprj3').resolve()
assert ad01_final_close_raw.get('ok') is True and ad01_final_close_raw['value']['status'] == 'closed'
assert ad01_raw_value['componentCount'] == ad01_official['componentCount'] == 167
assert ad01_raw_value['pinCount'] == ad01_official['pinCount'] == 452
assert ad01_raw_value['drc'].get('strict') is True and ad01_raw_value['drc'].get('errors') == []
assert ad01_raw_value.get('liveMismatches') == [] and ad01_raw_value.get('contractMismatches') == []
assert ad01_raw_value['project']['uuid'] == ad01_official['projectUuid']
assert ad01_raw_value['document']['uuid'] == ad01_official['pageUuid']
ad01_enet_bytes = ad01_raw_value['enetText'].encode('utf-8')
assert sha_bytes(ad01_enet_bytes) == ad01_official['enetSha256']
ad01_enet_path_matches_proof = (ad01_enetpath_current_sha == ad01_official['enetSha256'])
if ad01_enet_path_matches_proof:
    assert ad01_enetpath.read_bytes() == ad01_enet_bytes
assert ad01_raw_value['drc'].get('errors') == ad01_official['drc']['errors']
assert ad01_raw_value['drc'].get('warnings', []) == ad01_official['drc'].get('warnings', [])
ad01_api_evidence = [{
    'archivePath':ad01_raw_bundle_path.relative_to(M).as_posix(),
    'archiveSha256':sha(ad01_raw_bundle_path), 'archiveBytes':ad01_raw_bundle_path.stat().st_size,
    'rawResponseMember':ad01_raw_member,
    'rawResponseSha256':sha_bytes(ad01_raw_api_bytes),
    'historicalEnetReconstructedFromRawVerificationText':True,
    'historicalEnetSha256':sha_bytes(ad01_enet_bytes),
    'historicalEnetExportPath':ad01_enetpath.relative_to(M).as_posix(),
    'historicalEnetPathCurrentSha256':ad01_enetpath_current_sha,
    'historicalEnetPathStillMatchesProof':ad01_enet_path_matches_proof,
    'rawDrc':ad01_raw_value['drc'],
    'warningFieldPresentInRawDrc':'warnings' in ad01_raw_value['drc'],
    'saveCloseReopenClose':{
        'saveResult':ad01_save_raw['value']['saved'],
        'preReopenStatus':ad01_close_raw['value']['status'],
        'reopenPath':ad01_reopen_raw['value']['path'],
        'finalSessionStatus':ad01_final_close_raw['value']['status'],
        'rawMemberSha256':sha_bytes(ad01_raw_api_bytes)
    }
}]
ad01_enet = json.loads(ad01_enet_bytes)
ad01_actual = {c['props']['Designator']:c for c in ad01_enet['components'].values()}
change_proof = ad01_proof['changeProof']
assert change_proof['addedComponents'] == ['R_BUS_HI2']
assert change_proof['deletedComponents'] == []
assert change_proof['pinNetChanges'] == [
    {'ref':'R_BUS_HI','pin':'1','before':'BUS_EXT','after':'BUS_HI_MID'}]
assert change_proof['otherExistingPinNetChanges'] == 0

# The global cross-board proof starts at the fully verified AD01 source and binds
# the newer REG native/BOM/plan plus a separate official save-close-reopen readback.
crossboard_proof_path = POWER/'检查/统一审查/跨板并料复核.json'
crossboard_proof = load(crossboard_proof_path)
crossboard_reg = crossboard_proof['modules']['REG']
reg_baseline = crossboard_reg['baseline']
reg_after = crossboard_reg['after']
reg_change_proof = crossboard_reg['changeProof']
assert reg_baseline['nativeSha256'] == after_binding['nativeSha256']
assert reg_baseline['bomSha256'] == after_binding['bomSha256']
assert reg_baseline['designPlanSha256'] == after_binding['designPlanSha256']
assert reg_change_proof['identityRefs'] == ['TH_SC','TH_MAIN']
assert reg_change_proof['pinNetChanges'] == []
source_zip_binding = crossboard_proof['sourceZip']
source_zip = power_path(source_zip_binding['path'])
assert sha(source_zip) == source_zip_binding['sha256']
with zipfile.ZipFile(source_zip) as z:
    reg_baseline_archive_files = verify_file_bindings(z, reg_baseline['files'])
    native_rel = native_page.relative_to(POWER).as_posix()
    baseline_native_entry, reg_baseline_native = archive_payload(z, native_rel, reg_baseline['nativeSha256'])
assert sha_bytes(reg_baseline_native) == reg_baseline['nativeSha256']
assert reg_baseline['nativeSha256'] == after_binding['nativeSha256']
assert reg_baseline['bomSha256'] == after_binding['bomSha256']
assert reg_baseline['designPlanSha256'] == after_binding['designPlanSha256']
current_native = native_page.read_bytes()
bom_path = M/'原理图准备/BOM.csv'
plan_path = M/'原理图准备/design-plan.json'
assert sha(native_page) == reg_after['nativeSha256']
assert sha(bom_path) == reg_after['bomSha256']
assert sha(plan_path) == reg_after['designPlanSha256']
for record in reg_after['files']:
    path = power_path(record['path'])
    assert sha(path) == record['sha256'] and path.stat().st_size == record['bytes'], record['path']
reg_official = crossboard_reg['officialVerification']
official = reg_official
if official.get('sourceNativeSha256') is not None:
    assert official['sourceNativeSha256'] == reg_after['nativeSha256']
assert official['savedReopened'] is True and official['sessionClosed'] is True
assert official['componentCount'] == 167 and official['pinCount'] == 452
assert official['netMismatches'] == [] and official['liveMismatches'] == []
assert official['drc']['strict'] is True and official['drc']['errors'] == []
enetpath = power_path(official['enetPath'])
assert sha(enetpath) == official['enetSha256']
raw_bundle = crossboard_proof['rawResponseZip']
raw_bundle_path = power_path(raw_bundle['path'])
assert sha(raw_bundle_path) == raw_bundle['sha256']
with zipfile.ZipFile(raw_bundle_path) as z:
    raw_api_bytes = z.read(official['verificationEntry'])
    save_raw = json.loads(z.read(official['saveEntry']))
    close_raw = json.loads(z.read(official['closeEntry']))
    reopen_raw = json.loads(z.read(official['reopenEntry']))
    final_close_raw = json.loads(z.read(official['finalCloseEntry']))
    bridge_raw_bytes = z.read(reg_change_proof['bridgeRawEntry'])
bridge_raw_response = json.loads(bridge_raw_bytes)
raw_api = json.loads(raw_api_bytes)
raw_value = raw_api['value']
assert raw_api.get('ok') is True
assert save_raw.get('ok') is True and save_raw['value']['saved'] is True
assert close_raw.get('ok') is True and close_raw['value']['status'] == 'closed'
assert reopen_raw.get('ok') is True and Path(reopen_raw['value']['path']).resolve() == (M/'eda工程文件/sc-regulator/sc-regulator.eprj3').resolve()
assert final_close_raw.get('ok') is True and final_close_raw['value']['status'] == 'closed'
if official.get('pageUuid') is not None:
    assert raw_value['doc']['uuid'] == official['pageUuid']
if official.get('projectUuid') is not None:
    assert raw_value['project']['uuid'] == official['projectUuid']
assert raw_value['componentCount'] == official['componentCount']
assert raw_value['pinCount'] == official['pinCount']
enet_bytes = enetpath.read_bytes()
raw_enet_bytes = bytes(raw_value['enetUtf8Bytes'])
raw_bom_file_bytes = bytes(raw_value['bomFileBytes'])
assert raw_enet_bytes == enet_bytes
cad_bom_path = power_path(official['cadBomPath'])
assert sha(cad_bom_path) == official['cadBomSha256']
assert raw_bom_file_bytes == cad_bom_path.read_bytes()
assert raw_bom_file_bytes.startswith(b'\xff\xfe')
assert raw_value['bomFileEncoding'] == 'UTF-16LE BOM'
raw_enet_text = raw_value.get('enetText')
raw_enet_text_matches_bytes = (raw_enet_text is not None and
                               raw_enet_text.encode('utf-8') == raw_enet_bytes)
raw_enet_text_replacement_count = (raw_enet_text.count('\ufffd')
                                   if raw_enet_text is not None else None)
enet = json.loads(enet_bytes)
actual = {c['props']['Designator']:c for c in enet['components'].values()}
expected = {c['ref']:c for c in plan['components']+plan.get('systemBoundaries', [])}
procurement_ntc = procurement_ntc_rows(bom_path, {
    ref:expected[ref] for ref in ('TH_SC','TH_MAIN')
})
live_rows = raw_value['rows']
live_by_ref = live_pin_rows(live_rows)
assert len(live_rows) == raw_value['componentCount'] == len(actual) == len(expected) == 167
assert set(live_by_ref) == set(actual) == set(expected), '官方live元件、ENET与plan位号集合不一致'
assert sum(len(r.get('pins', [])) for r in live_rows) == raw_value['pinCount'] == 452
rows = []
computed_live_mismatches = []
computed_net_mismatches = []
for ref, component in actual.items():
    target_component = expected[ref]
    live = live_by_ref[ref]
    for field in ('x','y','rotation'):
        if field not in target_component or not math.isclose(float(live[field]), float(target_component[field]), rel_tol=0, abs_tol=1e-6):
            computed_live_mismatches.append({'ref':ref,'field':field,'expected':target_component.get(field),'actual':live.get(field)})
    bom_expectation = target_component.get('addIntoBom', target_component.get('bom'))
    pcb_expectation = target_component.get('addIntoPcb', target_component.get('pcb'))
    if bom_expectation is not None and bool(live.get('bom')) != bool(bom_expectation):
        computed_live_mismatches.append({'ref':ref,'field':'bom','expected':bool(bom_expectation),'actual':bool(live.get('bom'))})
    if pcb_expectation is not None and bool(live.get('pcb')) != bool(pcb_expectation):
        computed_live_mismatches.append({'ref':ref,'field':'pcb','expected':bool(pcb_expectation),'actual':bool(live.get('pcb'))})
    pins = target_component['pins']
    net_pins = component['pinInfoMap']
    live_pins = {str(pin.get('no') or pin.get('number')):pin for pin in live.get('pins', [])}
    assert set(pins) == set(net_pins) == set(live_pins), (ref,'live/ENET/plan pin set mismatch')
    for number, pin in net_pins.items():
        target = pins[number] or ''
        live_pin = live_pins[number]
        if live_pin.get('name') is not None and str(live_pin['name']) != str(pin['name']):
            computed_live_mismatches.append({'ref':ref,'pin':number,'field':'pinName',
                                             'expected':pin['name'],'actual':live_pin['name']})
        no_connected = live_pin.get('nc', live_pin.get('noConnected'))
        if no_connected is not None and bool(no_connected) != (not bool(pin['net'])):
            computed_live_mismatches.append({'ref':ref,'pin':number,'field':'connectivity',
                                             'expectedNoConnected':not bool(pin['net']),
                                             'actualNoConnected':bool(no_connected)})
        live_net = live_pin.get('net', live_pin.get('netName'))
        if live_net is not None and str(live_net or '') != str(pin['net'] or ''):
            computed_live_mismatches.append({'ref':ref,'pin':number,'field':'liveNet',
                                             'expected':pin['net'],'actual':live_net})
        rows.append({'ref':ref,'pin':number,'name':pin['name'],
                     'expectedNet':target,'actualNet':pin['net'],
                     'status':'PASS' if target == pin['net'] else 'FAIL'})
        if str(pin['net'] or '') != str(target):
            computed_net_mismatches.append({'ref':ref,'pin':number,'expected':target,'actual':pin['net']})
assert all(r['status']=='PASS' for r in rows), [r for r in rows if r['status']!='PASS']
assert computed_net_mismatches == []
assert computed_live_mismatches == official['liveMismatches'], computed_live_mismatches
assert len(actual) == official['componentCount'] == 167 and len(rows) == official['pinCount'] == 452

def enet_nets(enet_doc):
    result = {}
    for component in enet_doc['components'].values():
        ref = component['props']['Designator']
        result[ref] = {str(number):str(pin.get('net') or '')
                       for number,pin in component['pinInfoMap'].items()}
    return result
old_nets, new_nets = enet_nets(ad01_enet), enet_nets(enet)
computed_pin_net_changes = []
for ref in sorted(set(old_nets) | set(new_nets)):
    old_pins, new_pins = old_nets.get(ref,{}), new_nets.get(ref,{})
    for number in sorted(set(old_pins) | set(new_pins)):
        before, after = old_pins.get(number), new_pins.get(number)
        if before != after:
            computed_pin_net_changes.append({'ref':ref,'pin':number,'before':before,'after':after})
assert computed_pin_net_changes == reg_change_proof['pinNetChanges'] == []
old_actual = ad01_actual
identity_fields = ('Device','DeviceName','Footprint','FootprintName','Manufacturer Part','Manufacturer','Supplier Part')
identity_refs = []
for ref in sorted(set(old_actual) & set(actual)):
    old_props, new_props = old_actual[ref]['props'], actual[ref]['props']
    if any(old_props.get(field) != new_props.get(field) for field in identity_fields):
        identity_refs.append(ref)
assert identity_refs == sorted(reg_change_proof['identityRefs'])
for ref in identity_refs:
    target = expected[ref]
    live = live_by_ref[ref]
    expected_mpn = target.get('mpn') or target.get('manufacturerId')
    if expected_mpn:
        library_title = live.get('device', {}).get('name')
        if library_title != expected_mpn:
            spelling_alias = reg_change_proof.get('mpnSpellingNormalization', {})
            assert (ref in spelling_alias.get('refs', []) and
                    expected_mpn == spelling_alias.get('canonicalMpn') and
                    library_title == spelling_alias.get('sourceLibraryTitle') and
                    str(library_title).replace(' ', '') == str(expected_mpn).replace(' ', '')), (
                        ref, expected_mpn, library_title, spelling_alias)
    expected_supplier = target.get('supplierId')
    if expected_supplier:
        assert str(live.get('other', {}).get('Supplier Part ID') or '') == str(expected_supplier), ref
old_native_records, old_native_components = native_components(reg_baseline_native)
new_native_records, new_native_components = native_components(current_native)
assert set(new_native_components) == set(actual) == set(old_native_components)
native_geometry_mismatches = []
for ref, target in expected.items():
    before_component, after_component = old_native_components[ref], new_native_components[ref]
    for field in ('x','y','rotation'):
        if not math.isclose(float(after_component[field]),float(target[field]),rel_tol=0,abs_tol=1e-6):
            native_geometry_mismatches.append({'ref':ref,'field':field,'expected':target[field],'actual':after_component[field]})
        if not math.isclose(float(before_component[field]),float(after_component[field]),rel_tol=0,abs_tol=1e-6):
            native_geometry_mismatches.append({'ref':ref,'field':'baseline_'+field,'before':before_component[field],'after':after_component[field]})
assert native_geometry_mismatches == [], native_geometry_mismatches
geometry_bridges, wire_persistence = check_geometry_bridges(
    reg_baseline_native,current_native,live_by_ref,actual,bridge_raw_response,
    reg_change_proof['identityRefs'],reg_change_proof['geometryBridgeCount'],
    reg_change_proof.get('wirePersistence'))
raw_errors, raw_warnings = drc_groups(raw_value['drc'])
assert raw_errors == official['drc']['errors'] == []
assert raw_warnings == official['drc']['warnings']
api_evidence = [{
    'archivePath':raw_bundle_path.relative_to(POWER).as_posix(),
    'archiveSha256':sha(raw_bundle_path),'archiveBytes':raw_bundle_path.stat().st_size,
    'rawResponseMember':official['verificationEntry'],
    'rawResponseSha256':sha_bytes(raw_api_bytes),
    'rawEnetUtf8ByteArraySha256':sha_bytes(raw_enet_bytes),
    'rawEnetUtf8ByteArrayMatchesOfficialEnetByteForByte':True,
    'rawBomFileBytesSha256':sha_bytes(raw_bom_file_bytes),
    'rawBomFileBytesMatchesOfficialCadBomByteForByte':True,
    'rawBomFileEncoding':raw_value['bomFileEncoding'],
    'officialCadBom':{'path':cad_bom_path.relative_to(POWER).as_posix(),
                      'sha256':sha(cad_bom_path),'bytes':cad_bom_path.stat().st_size},
    'customProcurementBomNtcRefs':procurement_ntc,
    'rawEnetTextTransport':{'fieldPresent':raw_enet_text is not None,
                            'matchesUtf8ByteArray':raw_enet_text_matches_bytes,
                            'replacementCharacterCount':raw_enet_text_replacement_count,
                            'note':'When text differs, UTF-8 byte arrays are authoritative; replacement characters in the text serialization are a transport artifact.'},
    'rawDrcArray':raw_value['drc'],'recomputedDrcErrors':raw_errors,'recomputedDrcWarnings':raw_warnings,
    'rawBridgeResponse':{'archiveMember':reg_change_proof['bridgeRawEntry'],
                         'sha256':sha_bytes(bridge_raw_bytes),
                         'value':bridge_raw_response.get('value'),
                         'wirePersistence':wire_persistence},
    'recomputedContract':{'components':len(actual),'pins':len(rows),'netMismatches':[r for r in rows if r['status']!='PASS'],
                          'liveMismatches':computed_live_mismatches,'identityRefs':identity_refs,
                          'pinNetChanges':computed_pin_net_changes,'geometryBridges':geometry_bridges},
    'saveCloseReopenClose':{'saveResult':save_raw['value']['saved'],'preReopenStatus':close_raw['value']['status'],
                            'reopenPath':reopen_raw['value']['path'],'finalSessionStatus':final_close_raw['value']['status'],
                            'rawMemberSha256':sha_bytes(raw_api_bytes)}
}]
def resistor_spec(ref):
    component = expected[ref]
    raw = str(component.get('value') or component.get('procurementSpec') or '')
    match = re.fullmatch(r'\s*(\d+(?:\.\d+)?)\s*([kKmM]?)\s*(?:Ω|ohm)?\s*±\s*(\d+(?:\.\d+)?)\s*%\s*', raw)
    assert match, (ref, raw)
    multiplier = {'':1.0,'k':1e3,'K':1e3,'m':1e6,'M':1e6}[match.group(2)]
    ohms, tolerance_pct = float(match.group(1))*multiplier, float(match.group(3))
    procurement = component.get('procurementSpec')
    if procurement:
        procurement_match = re.fullmatch(r'\s*(\d+(?:\.\d+)?)\s*([kKmM]?)\s*(?:Ω|ohm)?\s*±\s*(\d+(?:\.\d+)?)\s*%\s*', str(procurement))
        assert procurement_match and float(procurement_match.group(1))*{'':1.0,'k':1e3,'K':1e3,'m':1e6,'M':1e6}[procurement_match.group(2)] == ohms
        assert float(procurement_match.group(3)) == tolerance_pct
    return {'ref':ref,'nominalOhm':ohms,'tolerancePct':tolerance_pct,'sourceValue':raw}

upper_parts = [resistor_spec('R_BUS_HI2'), resistor_spec('R_BUS_HI')]
lower_part = resistor_spec('R_BUS_LO')
assert expected['R_BUS_HI2']['pins'] == {'1':'BUS_EXT','2':'BUS_HI_MID'}
assert expected['R_BUS_HI']['pins'] == {'1':'BUS_HI_MID','2':'BUS_ADC'}
assert expected['R_BUS_LO']['pins'] == {'1':'BUS_ADC','2':'GND_SYS'}
upper_nominal = sum(p['nominalOhm'] for p in upper_parts)
lower_nominal = lower_part['nominalOhm']
assert len({p['tolerancePct'] for p in upper_parts+[lower_part]}) == 1
tolerance = upper_parts[0]['tolerancePct']/100
ratio_nominal = 1+upper_nominal/lower_nominal
ratio_min = 1+upper_nominal*(1-tolerance)/(lower_nominal*(1+tolerance))
ratio_max = 1+upper_nominal*(1+tolerance)/(lower_nominal*(1-tolerance))
bus_voltages = [37.8,69.4,80.0]
bus_divider = {
    'status':'IMPLEMENTED_IN_CURRENT_NATIVE; OFFICIAL_PROOF_BOUND; NOT_HARDWARE_QUALIFIED',
    'resistors':upper_parts+[lower_part],
    'upperTotalNominalOhm':upper_nominal,'lowerNominalOhm':lower_nominal,
    'nominalBusToAdcRatio':ratio_nominal,
    'tolerancePctEach':upper_parts[0]['tolerancePct'],
    'busToAdcRatioRange':[ratio_min,ratio_max],
    'reviewPoints':[{
        'busV':v,'nominalAdcV':v/ratio_nominal,
        'minAdcV':v/ratio_max,'maxAdcV':v/ratio_min,
        'below3V3AtListedInitialTolerance':v/ratio_min < 3.3
    } for v in bus_voltages],
    'scope':'Static resistor-ratio tolerance only. 69.4V is the listed TVS clamp point at specified Ipp; 80V is a separate voltage-boundary screen. Neither establishes the installed transient ceiling or MCU-off safety.',
    'tmux1511Alternative':{
        'ratioVbusPerVadc':37.386,
        'status':'UNIMPLEMENTED_CANDIDATE_ONLY',
        'appliesToCurrentNetlist':False
    }
}
assert math.isclose(ratio_nominal,31.0,rel_tol=0,abs_tol=1e-12)

# 每个实际器件均归入且只归入一块功能；功能检查和合同一致性分别记录。
def domain(ref):
    g = expected[ref].get('visualGroup', '')
    if ref == 'SYS_BAT_BUS': return '系统电源边界'
    if ref.startswith('SC') and ref[2:].isdigit(): return '固定超容与串联抽头'
    if g.startswith('cell'): return '逐节采样及均衡'
    return {'power':'功率桥/隔离/电感/电容/熔断', 'driver':'DRV8161及栅极网络',
            'aux12':'12V辅助电源', 'scov':'总压硬件比较', 'dump':'泄放功率及驱动',
            'core1':'核心板排针', 'core2':'核心板排针', 'logic_supply':'5V/3.3V电源边界',
            'busadc':'BUS采样', 'testfb':'抽头测试回读', 'can':'CAN通信及终端',
            'taps':'抽头接口', 'ntc_SC':'三路温度采样', 'ntc_MAIN':'三路温度采样',
            'ntc_DUMP':'三路温度采样'}.get(g, '其他已逐件检查')
groups = {}
for ref in actual: groups.setdefault(domain(ref), []).append(ref)

def item(id, state, subject, finding, action):
    return dict(id=id,status=state,subject=subject,finding=finding,action=action)
findings = [
 item('SC01','OPEN','启动瞬态容量与保护实现',
      '现行约束固定LICAP SC0350-270-RSS共8只、8S1P，新增数量0；16只/8S2P已取消。用途是启动瞬态补能与母线支撑，不要求持续1秒。BUSFIX为8S1P模型；实际启动功率缺口、峰值电流和持续时间未量化，变换器、ESR、逐组保护、响应及故障隔离未完成实物验收。',
      '保持8只不扩容；按实车启动波形确定功率/持续时间，再核验峰值、变换器、ESR、逐节保护与故障路径。BQ76952或其他专用AFE仍是8S1P逐组保护待验证候选。'),
 item('SC02','FAIL','逐节差分测量',
      'T1..7为68k/10k并联47k TEST支路，TEST为低时比例9.24681；T8比例7.8。累积抽头相减会累加电阻比、ADC、VREF、漏电及shunt参考误差；既有预算不满足逐组保护。',
      '停止把此链作为已证明单体硬保护；优先复用BMS的BQ76952候选，仍须解决0V首次充电、输入诊断和硬关断。'),
 item('SC03','PASS_WITH_LIMITS','BW6101脚序/真实阈值',
      '8路均为VDD=组+、GND/SEL=组-、LED悬空、IOUT经120Ω回组-。原厂SEL低为2.45V(25°C范围2.42..2.48)，不是2.65V；外部电阻把泄流限制到约20mA量级，不能替代几十A充电关断。',
      '按2.45V被动均衡建模，确认软充电上限/静态损耗；不能用原厂裸芯700mA数字替代本支路电流。'),
 item('PW01','FAIL','故障隔离',
      'Q_BUS_P源=BUS_BR、漏=BUS_EXT，其体二极管允许BUS_EXT向BUS_BR供电。QH D-S短路时MCU关PWM/关P管不能阻断向超容充电；80A保险不保证在中等持续过流下及时清除。',
      '比较复用双N管的双向阻断与背靠背P管；须覆盖默认失电、器件短路、0V充电及20uH电感断流能量。'),
 item('PW02','OPEN','P管驱动/损耗',
      '10V齐纳低端不能保证VGS达到IXTH76P10T的-10V RDS测试条件；47k/10k静态栅极网络未证明快速故障隔离。R_QP_COL仅0.1W，较高BUS边界下功耗超过额定。',
      '若保留P管，重算全温VGS、上拉关断时间、串阻功率、热态损耗及SOA；优先削减高损耗P管方案的系统成本。'),
 item('PW03','OPEN','主桥/磁件/熔断',
      'QH D=BUS_BR/S=SW，QL D=SW/S=GND，2只10uH串联后接SC_FUSED，0.5mΩ置SC负端，连接方向一致。12uH热态80A最小值、快限流延迟和外部AMG80的DC故障清除未证明。',
      '以实测L-I-T和端到端关断延迟限定能力；不能把0.8V VDS OCP视为精确80A限流。'),
 item('PW04','PASS_RATING_MISMATCH_FIXED_DYNAMIC_OPEN','TVS与被保护器件耐压',
      '当前Boarden SMCJ43A保留在BUS_EXT，六只bulk已改为PEH226NL3900QE4、900µF/80V。原63V额定低于SMCJ43A表列最大钳位69.4V@21.7A的问题已修正；80V与此规定脉冲数据有10.6V名义间隔，但并非实际瞬态保证。43V只是VRWM。现有SMCJ40A钳位64.5V@23.3A、VRWM40V，对37.8V正常母线仅2.2V余量，本轮未切换。实际脉冲电流、线束寄生与重复能量尚未闭合。',
      '按实测正常最高电压和最坏断流能量复核TVS及电容的联合窗口；不能只按单个VC值定型，也不把69.4V写成每次必达。'),
 item('PW05','PASS_WITH_LIMITS','母线电容位置与纹波',
      '已实施6×900uF/80V，实际在BUS_EXT；BUS_BR只有3×2.2uF薄膜、1uF和100nF。六bulk在125°C、≥5kHz自然对流条件下额定纹波理想总和44.4A，仅为筛选；关断P管后不能继续把5.4mF计在桥侧。',
      '按真实节点进行P管RMS、电容均流、局部换流和关断能量分析，实测ESL/尖峰/温升。'),
 item('GD01','PASS_WITH_LIMITS','DRV8161全部20脚',
      '引脚与TI DGS定义相符。3.3k DT为双输入带防直通，典型370ns；ID1/2均5.6k为典型512/768mA；30k增益20；27k VDS典型0.8V。BOOT接SW、VDRAIN接BUS_BR、SP/SL=GND、SN=SC_MINUS，CSAREF接核心板3.3V且去耦。',
      '保持低态启动；CSA输入±1.94mV偏置等效±3.88A仅25°C项，快保护必须校准并覆盖温漂/增益/延迟。'),
 item('GD02','OPEN','共享nFAULT许可电平',
      'PB5经1k推高共享故障节点，10k下拉、TLV OUTB开漏拉低。名义高约3.0V，但MCU保证VOH/3.0V低供电角落与输入阈值的裕量尚未形成完整证明。MCU挂高只剩总压保护，缺逐组硬许可。',
      '为实际控制输出建立最小VOH/最大灌电流预算并落硬件逐组许可，避免以典型3.3V计算作为保证。'),
 item('OV01','PASS_WITH_LIMITS','TLV6700全部6脚',
      'VDD=12V_DRV、GND/INA+=SC_MINUS、INB=2×255k/10k总压分压、OUTB接故障线、OUTA悬空，逻辑正确。总压典型20.8V，容差边界约20.57..21.05V，不保护失衡单体；12V失电不保证有效输出。',
      '保留其为独立总压后备；不能代替逐组OV/UV、失电隔离和启动资格。'),
 item('DU01','FAIL','dump保护独立性',
      'UCC27511的IN-=GND，IN+=PB7经100Ω，输入及门极下拉齐全；12V驱动掉电/MCU停止时dump停用。当前稳压板没有独立硬件BUS过压dump，板外1.8Ω电阻也没有实际脉冲/热资格。',
      '按两块FOC现有独立dump与中央dump的整车能量分工删减重复成本，仍须证明任一要求场景有有效能量出口。'),
 item('AD01','PASS_WITH_LIMITS','BUS ADC直流分压与初始容差',
      f"现行plan和fresh ENET核对为R_BUS_HI2({upper_parts[0]['nominalOhm']/1000:g}kΩ)串R_BUS_HI({upper_parts[1]['nominalOhm']/1000:g}kΩ)，下臂R_BUS_LO({lower_nominal/1000:g}kΩ)，Vbus/Vadc={ratio_nominal:g}；独立±{upper_parts[0]['tolerancePct']:g}%阻值容差下比例{ratio_min:.5f}..{ratio_max:.5f}。列出的37.8/69.4/80V静态点最大Vadc为{max(p['maxAdcV'] for p in bus_divider['reviewPoints']):.4f}V，低于3.3V。69.4V是TVS数据表指定Ipp点，80V仅器件电压边界筛查，不是整车瞬态上界。旧37.386 V/V仅属未实施TMUX1511候选，不适用于当前直接分压。",
      '本次只关闭列定静态点下的电阻初始容差量程子项。保持低复杂度直连分压；不得将静态窗口写成TVS/浪涌、ADC动态采样或整板过压保护通过。'),
 item('AD02','OPEN','核心板失电反灌与ADC动态采样',
      'BUS_ADC仍由无源分压直接接WeAct核心板PB0/ADC1_IN15；核心板掉电而BUS仍带电时，输入钳位可能抬升3V3_CORE或造成注入电流。分压电阻/C_BUS_ADC的一阶RC不能证明STM32 ADC采样获取时间、复用通道建立/串扰、转换误差或掉电注入安全。TMUX1511隔离路由没有落在当前原生。',
      '在实际核心板和固件采样时序下测量掉电注入、3V3_CORE抬升、ADC建立与误差，并按芯片限值核对；未闭合前保持核心板掉电/带电母线场景HOLD。'),
 item('AX01','PASS_WITH_LIMITS','12V模块身份、脚位和启动边界',
      'U_AUX1的HiLink厂家与URB4812YMD-10WR3型号由本地HiLink原厂资料直接匹配（参考资料/海凌科_URAB_YMD-10WR3_V1.7.pdf，SHA256 1c55e95b4c6457f75984519c61ba2c544b15f9e65016e8ebf4cc286b545ff49d）。原厂限值：输入18..75V、输出12V/833mA、最大容性负载680uF；1=负输入、2=正输入、3=正输出、4=NC、5=输出回路、6=CTRL，CTRL悬空为开启。当前BUS_EXT、12V_DRV、GND_SYS与1/2/3/5连线相符，4 NC及6悬空配置相符。要求最小负载为额定输出的3%；390Ω/2W在12V下名义0.369W/30.77mA，即约3.69%，可作预载候选。',
      '不再把厂家身份列为歧义，也不借用MORNSUN参数。仍需实测BUS_EXT最低输入启动、实物标签与装配方向、总输出容性负载不超680uF、输出预载温升及整机热稳态；这些未测前不提升全温/量产资格。'),
 item('IO01','PASS_WITH_LIMITS','核心板全部48脚/通信/温度',
      '48脚全部核对合同；PA8/PB13 TIM1互补输出、PB1/PA7电流、PB5许可、PB6 P管、PB7 dump，CAN PA11/12与TCAN332、3路NTC、TEST回读均一致。PB2/VREF脚未占用。PB6与PA9板载跳桥、BOOT0配置、USB反灌仍要求实板读回。',
      '强制按WeAct实际修订确认SB3/SB6和外部5V/USB路径；CAN终端仅在线路两端装配。NTC开短路状态须进入保护，不用温度正常值掩盖故障。'),
 item('IO02','OPEN','0V受控初充与普通充电低压边界',
      'BUSFIX普通充电低压许可漏洞已在27/27模型用例中关闭：每节测量值须≥2.2V，2.2V仍需显式请求；全零、任一节为0、2.05V AFE接管测试和低于2.2V均拒绝，低压撤销后不自动重启，强置请求不能绕过输出门。四个持久资格门仍为false。硬件受控0V初充、限流级、逐节测量/AFE交接、驱动自举最小脉宽、MCU失效独立关断和核心板掉电反灌仍未实现/资格验证。',
      '普通充电保持每节≥2.2V且需要显式请求；2.05V仅为AFE接管边界，不是高功率许可。设计并验证受控0V启动与独立逐节保护前，不开放硬件初充。'),
]
next(f for f in findings if f['id'] == 'IO02')['modelUpdate'] = busfix_model_update

busfix_model = {
    'inputPath':busfix_input_path.relative_to(M).as_posix(),
    'inputSha256':sha(busfix_input_path),
    'seriesCells':busfix_input['series_cells'],
    'parallelCellsPerGroup':busfix_input['parallel_cells_per_group'],
    'cellCount':busfix_input['cell_count'],
    'cellMpn':busfix_input['cell_mpn'],
    'cellCapacitanceNominalF':busfix_input['cell_capacitance_nominal_F'],
    'cellCapacitanceEolF':busfix_input['cell_capacitance_min_EOL_F'],
    'seriesCapacitanceEolF':busfix_input['cell_capacitance_min_EOL_F']/busfix_input['parallel_cells_per_group']/busfix_input['series_cells'],
    'normalMeasuredCellFloorV':busfix_input['planning_lower_voltage_V'],
    'candidateAfeTakeoverBoundaryV':2.05,
    'testCount':busfix_model_update['testCount'],
    'status':'MODEL_ONLY_NOT_HARDWARE_QUALIFICATION',
    'qualificationGates':busfix_input['qualification_gate'],
    'lifetimeEsrSensitivity_mOhm':44.8,
    'stressEsrSensitivity_mOhm':63.24,
    'esrUsedInIdealEnergyModel':False
}

calc = {
 'busDivider':bus_divider,
 'tap1to7TestLowRatio':1+68000/(1/(1/10000+1/47000)),
 'tap8Ratio':7.8,
 'balanceResistor':{'nominalOhm':120,'upperCurrentAt2_48V_A':2.48/120,'upperPowerAt2_48V_W':2.48**2/120},
 'shuntAt80A':{'dropV':80*.0005,'powerW':80**2*.0005},
 'csaNominal':{'zeroV':1.65,'slopeVperA':.01,'atPositive80A':2.45,'atNegative80A':.85,'offsetErrorAt25C_A':.00194/.0005},
 'auxPreloadW':12**2/390,
 'auxPreloadCurrentA':12/390,
 'auxPreloadPercentRatedOutput':(12**2/390)/10*100,
 'dumpRminOhm':1.8*.99,
 'dumpAt42V_W':42**2/(1.8*.99),
 'dumpAt45_59V_W':45.59**2/(1.8*.99),
 'pGateCollectorResistor':{'at37_8VAssume10Vz_W':(37.8-10)**2/10000,
                           'at45_59VAssume10Vz_W':(45.59-10)**2/10000},
 'bulkBusExt_F':6*900e-6,
 'bridgeLocalNominal_F':(3*2.2+1+.1)*1e-6,
 'inductorEnergyAt80A20uH_J':.5*20e-6*80**2,
 'sc2x100uFMin160uFIdealEnergyRiseFrom19_6V':math.sqrt(19.6**2+(20e-6*80**2)/160e-6),
}

# Exact BUS capacitor/TVS reuse screen, tied to the last generated central index.
library_index_path = M.parent/'元件封装库/索引/元件总表.json'
library_index = load(library_index_path)
library_index_sha = sha(library_index_path)
source_freshness = {}
for source in library_index.get('sources', []):
    source_path = M.parent/source['path']
    current_sha = sha(source_path) if source_path.exists() else None
    source_freshness[source['module']] = {
        'path': source['path'], 'indexedSha256': source['sha256'],
        'currentSha256': current_sha, 'match': current_sha == source['sha256']
    }

def indexed_rows(mpn_fragment):
    return [row for row in library_index.get('instances', []) if mpn_fragment.casefold() in str(row.get('mpn') or '').casefold()]

def indexed_usage(mpn_fragment):
    found = indexed_rows(mpn_fragment)
    per_module = {}
    for row in found:
        per_module.setdefault(row['module'], []).append(row['ref'])
    vehicle_count = sum(len(refs) * (2 if module == 'FOC驱动模块' else 1)
                        for module, refs in per_module.items())
    stale_modules = sorted(module for module in per_module if not source_freshness.get(module, {}).get('match', False))
    return {
        'indexInstanceCount': len(found), 'perModuleDesignators': per_module,
        'vehicleCountEstimate': vehicle_count, 'staleSourceModules': stale_modules,
        'vehicleCountStatus': 'PROVISIONAL_INDEX_SOURCE_STALE' if stale_modules else 'INDEX_SOURCE_MATCH'
    }

bus_power_screen = {
    'asOf': '2026-10-21',
    'centralIndex': {
        'path': library_index_path.relative_to(M.parent).as_posix(),
        'sha256': library_index_sha, 'sourceFreshness': source_freshness,
        'vehicleQuantityBasis': 'FOC驱动模块按2块，其余已索引模块按1份；任一来源SHA不匹配时其用量标为临时。'
    },
    'bulkCapacitor': {
        'current': {
            'manufacturer': 'KEMET', 'mpn': 'PEH226NL3900QE4',
            'referenceCount': 6, 'designators': [f'C_BUS_BULK{i}' for i in range(1, 7)],
            'ratedVoltageV': 80, 'capacitanceEachUf': 900, 'totalCapacitanceUf': 5400,
            'rippleRatedNaturalConvection125CAtGe5kHz_A_each': 7.4,
            'rippleRatedNaturalConvection125CAtGe5kHz_A_total': 44.4,
            'bomFootprintStatus': 'original_ML_CAD_carrier_retained; exact_NL_footprint_qualification_pending',
            'implementationStatus': 'IMPLEMENTED_NATIVE_ATTR_AND_BOM; PRICE_NOT_QUOTED'
        },
        'sameSeriesCandidate': {
            'status': 'IMPLEMENTED; legacy schema key retained for compatibility',
            'manufacturer': 'KEMET/YAGEO', 'mpn': 'PEH226NL3900QE4',
            'referenceCountForSameNominalCapacitance': 6,
            'ratedVoltageV': 80, 'capacitanceEachUf': 900, 'totalCapacitanceUf': 5400,
            'ratedRippleNaturalConvection125CAtGe5kHz_A_each': 7.4,
            'ratedRippleNaturalConvection125CAtGe5kHz_A_total': 44.4,
            'maxRippleNaturalConvection125CAtGe5kHz_A_each_notContinuousRating': 9.4,
            'rippleAt150CWithHeatSink_A_each': 5.7,
            'rippleAt150CWithHeatSink_A_total': 34.2,
            'nominalCase': 'size L; 20.2mm x 43.7mm; 15.24mm lead spacing',
            'compatibility': '几何尺寸同级但真实封装仍待导入/核对；2026-04-28数据表注明L规格新设计采用top-weld crown，目录件可能仍为side-weld，不作drop-in声明。',
            'deltaFromCurrent': {
                'comparisonBasis': 'previous 63V ML baseline, not the present 80V selection',
                'ratedVoltageIncreaseV': 17, 'capacitanceChangeUf': 0,
                'ratedNaturalConvection125CAtGe5kHzRippleChange_A_total': -4.2,
                'ratedRippleChangePercent': -8.641975308641975,
                '40ArmsHeadroomAtThatOneRatedCondition_A': 4.4,
                'fullTemperature40ArmsAcceptance': 'NOT_ESTABLISHED'
            }
        },
        'existing100VCentralPart': {
            'manufacturer': 'KEMET', 'mpn': 'PEH200PJ5100MB2',
            'indexReferencesPerFOCBoard': 3, 'vehicleEstimateWithFOCx2': 6,
            'ratedVoltageV': 100, 'capacitanceEachUf': 10000,
            'caseAndMounting': '50mm x 95mm screw-terminal can',
            'reuseAssessment': '已分配FOC母线电容；体积、螺栓端子与当前径向15.24mm候选焊盘不同，不是稳压板drop-in。'
        },
        'electricalBoundary': '电容承压按VDC + peak VAC不得超过额定值。已选80V器件相对SMCJ43A表列69.4V钳位仅有10.6V名义差值；实际浪涌、走线尖峰和电流必须实测。'
    },
    'busTvs': {
        'normalBusUpperReferenceV': 37.8,
        'options': [
            {'manufacturer':'Boarden','mpn':'SMCJ43A','designators':['稳压模块:D_BUS_TVS'],
             'indexUsage':indexed_usage('SMCJ43A'),'vrwmV':43,'vbrMinMaxV':[47.8,52.8],
             'vcMaxV':69.4,'ippA':21.7,'headroomTo37_8V_V':5.2,
             'datasheet':'Boarden SMCJ-Series, rev.1.1, p.2; local SHA256 cd6f48ce70aa36cc1ec26b499838fcf44face4eed34dcd9e1e45ebb0bf6139dd'},
            {'manufacturer':'Littelfuse','mpn':'SMCJ40A','designators':['FOC驱动模块:DTVS1','水泵驱动模块:D_BUS'],
             'indexUsage':indexed_usage('SMCJ40A'),'vrwmV':40,'vbrMinMaxV':[44.4,49.1],
             'vcMaxV':64.5,'ippA':23.3,'headroomTo37_8V_V':2.2,
             'reuseAssessment':'跨模块已有型号可复用，但37.8V正常上限下仅2.2V至VRWM；需确认全容差/纹波最高电压严格低于40V。'},
            {'manufacturer':'Littelfuse','mpn':'SMCJ45A','designators':['BMS模块:D_TVS'],
             'indexUsage':indexed_usage('SMCJ45A'),'vrwmV':45,'vbrMinMaxV':[50.0,55.3],
             'vcMaxV':72.7,'ippA':20.6,'headroomTo37_8V_V':7.2,
             'reuseAssessment':'有更高VRWM余量，但表列钳位高于SMCJ43A，不能改善bulk电容尖峰余量。'}
        ],
        'ratingBoundary': 'Littelfuse SMCJ为1500W、10/1000µs、0.01%占空比的浪涌指标，按非重复脉冲标定；不构成动力母线重复断流能量或线束寄生的保证。',
        'screenConclusion': '本轮已实施保留SMCJ43A、六只bulk改为PEH226NL3900QE4/80V，数量与总容量不变。未切换SMCJ40A。当前只修复额定不匹配；实际波形、纹波、温升、物理封装及价格仍待验证，不授予释放资格。'
    },
    'primarySources': [
        {'title':'YAGEO/KEMET PEH226 datasheet A4017 (2026-04-28), electrical ratings and dimensions',
         'url':'https://yageogroup.com/content/datasheet/asset/file/KEM_A4017_PEH220_225_226'},
        {'title':'Littelfuse SMCJ series datasheet rev.2 (2025-02-27), electrical table p.2',
         'url':'https://www.littelfuse.com/~/media/electronics/datasheets/tvs_diodes/littelfuse_tvs_diode_smcj_datasheet.pdf.pdf'},
        {'title':'KEMET PEH200 datasheet A4034 (2023-08-08), ratings p.7 and dimensions p.3',
         'url':'https://content.kemet.com/datasheets/KEM_A4034_PEH200.pdf'},
        {'title':'Boarden SMCJ Series rev.1.1 p.2',
         'localPath':'参考资料/Boarden_SMCJ-Series.pdf',
         'sha256':'cd6f48ce70aa36cc1ec26b499838fcf44face4eed34dcd9e1e45ebb0bf6139dd',
         'url':'https://boardensh.com/wp-content/uploads/2024/11/SMCJ-Series.pdf'}
    ]
}
native = M/'eda工程文件/sc-regulator'
sources = [{'path':p.relative_to(M).as_posix(),'sha256':sha(p),'bytes':p.stat().st_size}
           for p in sorted(native.rglob('*')) if p.is_file()]
report_date = __import__('datetime').date.today().isoformat()
native_sha = sha(native_page)
bom_path = M/'原理图准备/BOM.csv'
plan_path = M/'原理图准备/design-plan.json'
result = {'schemaVersion':2,'asOf':report_date,
 'scope':f'现行8S1P原生单页全部{len(actual)}个对象及{len(rows)}脚；不含未落图AFE/TMUX候选',
 'reviewComplete':True,'designRelease':'HOLD','sourceFiles':sources,
 'downstreamWorkBoundary':{'postSchematicWorkOwner':'HUMAN',
        'pcbModifiedThisTurn':False,'physicalAssemblyAndQualification':'NOT_PERFORMED'},
 'sourceBinding':{'nativePath':native_page.relative_to(M).as_posix(),'nativeSha256':native_sha,
         'bomPath':bom_path.relative_to(M).as_posix(),'bomSha256':sha(bom_path),
         'designPlanPath':plan_path.relative_to(M).as_posix(),'designPlanSha256':sha(plan_path),
         'ad01ProofPath':ad01_proof_path.relative_to(M).as_posix(),'ad01ProofSha256':sha(ad01_proof_path),
         'crossBoardProofPath':crossboard_proof_path.relative_to(POWER).as_posix(),
         'crossBoardProofSha256':sha(crossboard_proof_path)},
 'enet':{'path':enetpath.relative_to(M).as_posix(),'sha256':sha(enetpath),
         'freshlyExportedAfterSaveReopen':True,'status':'CURRENT_OFFICIAL_ENET_BOUND_TO_AFTER_NATIVE',
         'sourceNativeSha256':reg_after['nativeSha256'],
         'basis':'正式原生保存并整会话重开后读取的官方ENET；元件/引脚/网名及几何由全局并料proof的原始API rows、ENET、当前design-plan与native重新计算核对。'},
 'contract':{'components':len(actual),'pins':len(rows),
             'mismatches':sum(r['status']!='PASS' for r in rows)+len(computed_live_mismatches)+len(native_geometry_mismatches),
             'liveGeometryMismatches':computed_live_mismatches,
             'nativeGeometryMismatches':native_geometry_mismatches,
             'verifiedNetChangesFromAD01':computed_pin_net_changes,
             'verifiedIdentityRefsFromAD01':identity_refs,
             'verifiedGeometryBridges':geometry_bridges,
             'wirePersistence':wire_persistence,
             'meaning':'合同一致只证明映射，不证明设计正确'},
 'groups':{k:{'references':v,'pins':sum(len(actual[r]['pinInfoMap']) for r in v)} for k,v in groups.items()},
 'findings':findings,'calculations':calc,'busfixModel':busfix_model,'busProtectionReuseScreen':bus_power_screen,
 'api':{'status':'FORMAL_SAVE_REOPEN_READBACK',
        'savedReopened':official['savedReopened'],
        'strictDrc':official['drc'],
        'ownedSessionClosed':True,
        'rawOfficialEvidence':api_evidence,
        'rawDrcWarningCount':len(raw_warnings)},
 'bomVerification':{'customProcurementBom':{'path':bom_path.relative_to(POWER).as_posix(),
                                             'sha256':sha(bom_path),
                                             'ntcRefs':procurement_ntc},
                    'officialCadBomFile':{'path':cad_bom_path.relative_to(POWER).as_posix(),
                                          'sha256':sha(cad_bom_path),
                                          'bytes':cad_bom_path.stat().st_size,
                                          'encoding':raw_value['bomFileEncoding'],
                                          'rawApiBytesMatchExactly':True}},
 'ad01DividerProof':{'proofPath':ad01_proof_path.relative_to(M).as_posix(),
        'proofSha256':sha(ad01_proof_path),
        'baseline':baseline_binding,'after':after_binding,
         'sourceZipSha256':sha(ad01_source_zip),'verifiedBaselineArchiveFiles':baseline_aux,
        'changeProof':change_proof,
         'officialVerification':ad01_official,'rawOfficialEvidence':ad01_api_evidence},
 'crossBoardCostProof':{'proofPath':crossboard_proof_path.relative_to(POWER).as_posix(),
         'proofSha256':sha(crossboard_proof_path),
         'sourceZip':{'path':source_zip_binding['path'],'sha256':sha(source_zip),
                      'baselineArchiveNativeEntry':baseline_native_entry,
                      'verifiedBaselineFiles':reg_baseline_archive_files},
         'rawResponseZip':{'path':raw_bundle_path.relative_to(POWER).as_posix(),
                           'sha256':sha(raw_bundle_path),'bytes':raw_bundle_path.stat().st_size},
         'baseline':reg_baseline,'after':reg_after,'changeProof':reg_change_proof,
         'computedProof':{'identityRefs':identity_refs,'pinNetChanges':computed_pin_net_changes,
                          'geometryBridges':geometry_bridges,
                          'wirePersistence':wire_persistence,
                          'netMismatches':[r for r in rows if r['status']!='PASS'],
                          'liveMismatches':computed_live_mismatches,
                          'nativeGeometryMismatches':native_geometry_mismatches},
         'officialVerification':official,'rawOfficialEvidence':api_evidence},
 'peh226IntermediateProof':{'proofPath':material_proof_path.relative_to(M).as_posix(),
        'proofSha256':sha(material_proof_path),'verifiedChangedAttributeRecords':changed_count,
        'intermediateNativeSha256':material_proof['nativeProof']['afterSha256'],
        'currentNativeSha256':native_sha,
        'note':'48项PEH属性证明绑定AD01修改前的中间基线，不代表当前native SHA。',
        'selectedPart':material_proof['selectedPart'],'nativeProof':material_proof['nativeProof']},
 'historicalPeHBackup':{'path':peh_archive.relative_to(M).as_posix(),'sha256':sha(peh_archive),
        'p1BaselineSha256':material_proof['backup']['p1BaselineSha256']},
 'busDividerContract':{'topology':'BUS_EXT -> R_BUS_HI2 -> BUS_HI_MID -> R_BUS_HI -> BUS_ADC; R_BUS_LO from BUS_ADC to GND_SYS',
        'ratioDerivedFromPlan':ratio_nominal,'ratioToleranceRange': [ratio_min,ratio_max],
        'tmuxRatio37_386Status':'UNIMPLEMENTED_CANDIDATE_ONLY'},
 'pinMatrix':rows,
 'csvPath':'检查/全部引脚合同核对.csv',
 'manufacturerDocuments':[{'manufacturer':'HiLink','mpn':'URB4812YMD-10WR3',
    'localPath':'参考资料/海凌科_URAB_YMD-10WR3_V1.7.pdf',
    'sha256':'1c55e95b4c6457f75984519c61ba2c544b15f9e65016e8ebf4cc286b545ff49d',
    'use':'U_AUX1厂家身份、针脚和电气边界的主依据；本地文件SHA读回确认。'},
   {'manufacturer':'MORNSUN','localPath':'参考资料/MORNSUN_URB_YMD-10WR3_2026.09.17-B5.pdf',
    'use':'不同厂家的同系列资料；不用于U_AUX1身份或规格判定。'}],
 'primarySources':[
 'https://www.ti.com/lit/ds/symlink/drv8161.pdf',
 'https://www.ti.com/lit/ds/symlink/tlv6700.pdf',
 'https://bitwell.cn/Upload/%E4%BF%9D%E6%8A%A4%E8%8A%AF%E7%89%87/BW6101_C%E4%BF%9D%E6%8A%A4IC-09311941072.pdf',
 'https://bitwell.cn/caseview.aspx?CateID=614&ID=122',
 'HiLink URB4812YMD-10WR3原厂资料：参考资料/海凌科_URAB_YMD-10WR3_V1.7.pdf；SHA256 1c55e95b4c6457f75984519c61ba2c544b15f9e65016e8ebf4cc286b545ff49d',
 'https://boardensh.com/wp-content/uploads/2024/11/SMCJ-Series.pdf',
 'https://yageogroup.com/content/datasheet/asset/file/KEM_A4017_PEH220_225_226',
 'https://www.littelfuse.com/~/media/electronics/datasheets/tvs_diodes/littelfuse_tvs_diode_smcj_datasheet.pdf.pdf',
 'https://content.kemet.com/datasheets/KEM_A4034_PEH200.pdf',
 'https://www.st.com/resource/en/datasheet/stm32g431cb.pdf']}
(O/'全功能与逐脚复核.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
current_csv = O/'全部引脚合同核对.csv'
with current_csv.open('w',encoding='utf-8-sig',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
legacy_csv = O/'全部450脚核对.csv'
if legacy_csv.exists():
    legacy_csv.unlink()

drc_errors = len(raw_errors)
drc_warnings = len(raw_warnings)
md=['# 稳压模块全部原理检查','',
    f"{report_date}。检查覆盖现行单页全部{len(actual)}个对象、{len(rows)}脚；原始API live rows、ENET与plan重算得到网名差异{sum(r['status']!='PASS' for r in rows)}项、几何差异{len(computed_live_mismatches)+len(native_geometry_mismatches)}项。正式工程已保存并整会话重开；从原始DRC数组重算为{drc_errors} error、{drc_warnings} warning。**分压直流容差子项局部关闭，整板原理放行仍HOLD**。固定8只LICAP、8S1P；16只/8S2P已取消。所有原理图之后的工作由人类完成；本轮未改PCB，实物装配与资格验证未执行。",
    '',
    f"当前原生 SHA256 {native_sha}、BOM {sha(bom_path)}、design-plan {sha(plan_path)} 由[跨板并料复核证明](../../检查/统一审查/跨板并料复核.json)的REG after绑定；REG baseline与AD01.after逐项同SHA。官方ENET {sha(enetpath)} ({len(actual)}元件/{len(rows)}脚)由本次保存重开后的直接API证据绑定；原始rows、net/geometry重算、DRC数组及SHA列于JSON。当前逐脚表为[全部引脚合同核对.csv](全部引脚合同核对.csv)。",
    '',
    f"TH_SC/TH_MAIN四个引脚各核对一个旧位置到现位置的10 schematic-unit连接段（2.54mm），并验证现引脚网名及同wireGroup续接。原始API规范化后LINE新增{wire_persistence['addedLinePrimitives']}条、删除{wire_persistence['deletedLinePrimitives']}条、变化{wire_persistence['changedLinePrimitives']}条；四段在AD01基线已存在，因此记录为连接验证，不作为新增导线或PCB变更。",
    '',
    f"PEH226 80V修改证明仍保留原48项ATTR差分与ZIP字节；其修改后SHA {material_proof['nativeProof']['afterSha256']} 是AD01之前的中间基线。AD01分压证明仍按原ZIP逐记录核验；其after SHA {after_binding['nativeSha256']} 是REG本轮的baseline，而非当前源。REG baseline源ZIP SHA为 {sha(source_zip)}；全图报告分别验证PEH 48项差分、AD01来源链和本轮原始API/net/geometry/DRC证据。",
    '', '| 功能区 | 对象数 | 引脚数 |','|---|---:|---:|']
for k,v in groups.items():md.append(f'| {k} | {len(v)} | {result["groups"][k]["pins"]} |')
md+=['','## 原理结论与整改边界','']
for f in findings:md += [f'### {f["id"]} · {f["subject"]} · {f["status"]}', '',f['finding'],'',f'处理：{f["action"]}','']
md += ['## BUSFIX模型边界','',
       f"BUSFIX输入固定{busfix_model['seriesCells']}S{busfix_model['parallelCellsPerGroup']}P、{busfix_model['cellCount']}只{busfix_model['cellMpn']}，280F/节寿命末期、{busfix_model['seriesCapacitanceEolF']:g}F串联等效。普通充电测量下限2.2V/节；2.05V仅为AFE接管候选边界。模型记录{busfix_model['testCount']}测试PASS，但仍是算法/理想能量模型，四个持久资格门全部为false，0V硬件初充和整板保护未通过。44.8mΩ寿命ESR敏感值和63.24mΩ应力值不纳入当前理想能量PASS判据。",
       '', '## AD01分压核验','',
       f"按现行plan电阻器件Value推导：上臂总值{upper_nominal/1000:g}kΩ、下臂{lower_nominal/1000:g}kΩ，名义Vbus/Vadc={ratio_nominal:g}，每只±{upper_parts[0]['tolerancePct']:g}%时比例范围{ratio_min:.5f}..{ratio_max:.5f}。列定37.8/69.4/80V静态点的最大ADC节点分别为{bus_divider['reviewPoints'][0]['maxAdcV']:.4f}/{bus_divider['reviewPoints'][1]['maxAdcV']:.4f}/{bus_divider['reviewPoints'][2]['maxAdcV']:.4f}V。该检查只关闭初始电阻容差静态量程子项，不等于TVS浪涌、ADC动态或core-off注入安全通过。",
       '', 'TMUX1511路径未实施；其37.386V/V仅作为TMUX候选模型，不能代替当前直接分压31V/V，也不能带入当前固件换算。','']
cap = bus_power_screen['bulkCapacitor']
tvss = bus_power_screen['busTvs']['options']
stale_index_modules = [name for name, state in source_freshness.items() if not state['match']]
md += ['## BUS bulk电容与TVS精确复用筛选','',
       f"中央索引SHA256：`{library_index_sha}`。来源SHA未匹配模块：{', '.join(stale_index_modules) if stale_index_modules else '无，四个当前来源均匹配'}。" + ('涉及未匹配来源的用量仅为临时快照。' if stale_index_modules else '当前用量与已同步源绑定；不代替电气资格。'),
       '',
       f"当前稳压原生及BOM已选{cap['current']['referenceCount']}只 KEMET {cap['current']['mpn']}（{cap['current']['capacitanceEachUf']}µF/{cap['current']['ratedVoltageV']}V，合计{cap['current']['totalCapacitanceUf']/1000:.1f}mF）。相对修改前63V型号，数量与容量不变；125°C自然对流、≥5kHz额定纹波由48.6A降到44.4A（每只8.1A降至7.4A），相比40Arms名义余量4.4A，不能据此宣称全温验收。80V器件在150°C热沉条件为5.7A/只、六只34.2A。新件待询价，未沿用旧件价格。",
       '',
       'PEH226 80V器件为同尺寸级L（20.2×43.7mm、15.24mm引脚间距），但本板PCB封装仍标待导入；YAGEO/KEMET 2026-04-28数据表指出L尺寸新设计可能采用top-weld crown，目录件可能为side-weld，必须实物核对焊冠、极性和孔距，不能视作drop-in。PEH226系列最高80V；中央库中的100V PEH200PJ5100MB2为10,000µF、50×95mm螺栓端子，已分配FOC，不是现有径向焊盘的直接替代。',
       '', '| TVS料号 | VRWM | VBR min–max | 最大VC@Ipp | 对37.8V余量 | 索引现有用量 |', '|---|---:|---:|---:|---:|---|']
for option in tvss:
    usage_text = '; '.join(f"{module}×{len(refs)} ({', '.join(refs)})" for module, refs in option['indexUsage']['perModuleDesignators'].items()) or '无'
    if option['indexUsage']['staleSourceModules']:
        usage_text += '；来源待刷新'
    md.append(f"| {option['mpn']} | {option['vrwmV']}V | {option['vbrMinMaxV'][0]}–{option['vbrMinMaxV'][1]}V | {option['vcMaxV']}V@{option['ippA']}A | {option['headroomTo37_8V_V']}V | {usage_text} |")
md += ['', 'SMCJ43A是当前稳压板器件；同型号SMCJ40A可跨模块复用，但正常最高母线到40V VRWM仅2.2V，须确认全容差、纹波和温漂都低于40V。SMCJ45A的VRWM余量更大，钳位也更高。三者的VC均是数据表规定Ipp下的值，不是任意浪涌下的母线绝对上限。Littelfuse SMCJ 1500W指标对应10/1000µs、0.01%占空比非重复脉冲，不能替代实际重复能量/寄生尖峰测量。',
       '', bus_power_screen['busTvs']['screenConclusion'], '']
md += ['## 复算中需要保留的节点边界','',
       '6×900µF位于BUS_EXT，桥侧名义局部电容只有7.7µF；不能把5.4mF当关断后桥侧储能。20µH/80A电感能量64mJ，若全部进入最小160µF的SC侧局部电容，19.6V可理想上升至约34.4V；真实断流路径须单独核对，不能拿35V名义耐压认定有充足裕量。',
       '', 'BW6101的120Ω支路在2.48V下即使忽略内部MOS压降也仅20.7mA/51.3mW。原厂网页200mA与旧PDF700mA为裸芯不同条件；均不能变成本电路支路泄流能力。',
        '', '本轮未修改PCB，也未把模型通过改写为硬件通过。详细逐脚表和计算见同目录JSON及[全部引脚合同核对.csv](全部引脚合同核对.csv)。', '', '## 原厂资料','']
md += [f'- {url}' for url in result['primarySources']]
(O/'全功能与逐脚复核.md').write_text('\n'.join(md)+'\n',encoding='utf-8')
conclusion_path = O/'原理图复核结论.md'
conclusion = conclusion_path.read_text(encoding='utf-8')
history_marker = '\n## 历史检查记录（2026-10-14）'
history_index = conclusion.find(history_marker)
assert history_index >= 0
current_summary = (
    f"# 当前原理图复核结论\n\n"
    f"> **{report_date}现行原理审查：HOLD。** 当前单页{len(actual)}个对象、{len(rows)}脚；已从官方原始live rows、ENET、design-plan和native重算网名/几何合同，映射差异0项，几何差异0项；正式保存重开读回。严格DRC从原始数组重算为{drc_errors} error、{drc_warnings} warning。"
    f" AD01分压静态初始容差子项PASS_WITH_LIMITS；核心板掉电反灌和ADC动态保持OPEN，整板不放行。固定8S1P与BUSFIX 27/27模型状态保留。"
    f" 逐项结论见[现行全图原理审查](全功能与逐脚复核.md)、[结构化问题与逐脚表](全功能与逐脚复核.json)及[引脚合同CSV](全部引脚合同核对.csv)。"
    f" PEH的48项ATTR/原ZIP只证明AD01之前的中间基线；AD01.after作为本轮REG baseline，当前native/BOM/plan由跨板并料proof after绑定。下文2026-10-14 DRC为历史记录，不代表当前原理放行。"
)
conclusion_path.write_text((current_summary + conclusion[history_index:]).rstrip('\r\n')+'\n',encoding='utf-8')
print(json.dumps({'objects':len(actual),'pins':len(rows),
                  'mismatches':sum(r['status']!='PASS' for r in rows)+len(computed_live_mismatches)+len(native_geometry_mismatches),
                  'groups':len(groups),
                  'findings':len(findings),'release':'HOLD','sourceSha':native_sha,
                  'busDividerRatioFromPlan':ratio_nominal,'drcErrors':drc_errors,
                  'drcWarnings':drc_warnings,'csvPath':current_csv.name},ensure_ascii=True))
