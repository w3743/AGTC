"""只读复核现行设计输入；只向本目录写入证据，不运行历史修改/保存脚本。"""
from pathlib import Path
import hashlib
import json
import math
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[3]
POWER = ROOT / '电力'
OUT = Path(__file__).resolve().parent

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def load(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))

def native_attrs(path):
    active = False
    attrs, comps, lines = {}, {}, {}
    for no, line in enumerate(path.read_text(encoding='utf-8-sig').splitlines(), 1):
        if '||' not in line:
            continue
        h, b = line.split('||', 1)
        h = json.loads(h)
        b = b.rstrip('|')
        if not b:
            continue
        body = json.loads(b)
        if h['type'] == 'DOCHEAD':
            active = body.get('docType') == 'SCH_PAGE'
        if not active:
            continue
        if h['type'] == 'COMPONENT':
            comps[h['id']] = body
            lines[h['id']] = no
        elif h['type'] == 'ATTR' and body.get('parentId'):
            attrs.setdefault(body['parentId'], {})[body['key']] = body.get('value')
    return {a['Designator']: {'id': cid, 'line': lines[cid], 'attrs': a}
            for cid, a in attrs.items() if cid in comps and a.get('Designator')}

proof_path = POWER / '检查/统一审查/跨板并料复核.json'
proof = load(proof_path)
evidence = {'date': '2026-10-29', 'timezone': 'Asia/Shanghai',
    'scope': '四份自研现行原理图；FOC两板同版，只计一次针网；不改原生、BOM、PCB或固件',
    'method': '本轮离线SHA绑定、原生属性读回和已有官方ENET与现行合同比较。没有本轮官方重导出/DRC。',
    'priorProof': {'path': str(proof_path.relative_to(ROOT)), 'sha256': digest(proof_path)},
    'modules': {}}

evidence['priorArchives'] = {}
for key in ('sourceZip', 'rawResponseZip'):
    info = proof[key]
    p = POWER / info['path']
    evidence['priorArchives'][key] = {'path': info['path'], 'exists': p.is_file(),
        'sha256': digest(p) if p.is_file() else None,
        'hashMatchesProof': p.is_file() and digest(p) == info['sha256']}

for name, data in proof['modules'].items():
    files = []
    for f in data['after']['files']:
        p = POWER / f['path']
        actual = digest(p) if p.is_file() else None
        files.append({'path': f['path'], 'expected': f['sha256'], 'actual': actual,
                      'match': actual == f['sha256']})
    native_path = next(POWER / f['path'] for f in data['after']['files'] if f['path'].endswith('.esch2'))
    native = native_attrs(native_path)
    official = data['officialVerification']
    enet_path = POWER / official['enetPath']
    enet = load(enet_path)
    by_ref = {c['props']['Designator']: c for c in enet['components'].values()}
    native_missing = sorted(set(by_ref) - set(native))
    mpn_diff = []
    for ref, c in by_ref.items():
        n = native.get(ref, {}).get('attrs', {}).get('Manufacturer Part', '') or ''
        e = c['props'].get('Manufacturer Part', '') or ''
        if n != e:
            mpn_diff.append({'ref': ref, 'native': n, 'enet': e})
    m = {'files': files, 'allPriorSourceHashesMatch': all(f['match'] for f in files),
        'native': str(native_path.relative_to(ROOT)), 'nativeSha256': digest(native_path),
        'nativeDesignatorCount': len(native), 'enet': str(enet_path.relative_to(ROOT)),
        'enetSha256': digest(enet_path), 'enetHashMatchesPriorProof': digest(enet_path) == official['enetSha256'],
        'components': len(by_ref), 'pins': sum(len(c['pinInfoMap']) for c in by_ref.values()),
        'missingNativeRefs': native_missing, 'nativeVsEnetMpnDifferences': mpn_diff,
        'nativeRefLocations': {ref: {'line': item['line'], 'id': item['id']} for ref, item in native.items()},
        'priorDrcOnly': official.get('drc', official.get('strictDrc', {}))}
    with zipfile.ZipFile(POWER / proof['rawResponseZip']['path']) as archive:
        raw = json.loads(archive.read(official['verificationEntry']))
        assert raw['ok'] is True
        raw = raw['value']
        raw_drc = raw['drc']
        m['rawArchiveVerification'] = {
            'entry': official['verificationEntry'],
            'componentCountMatches': raw['componentCount'] == len(by_ref),
            'pinCountMatches': raw['pinCount'] == m['pins'],
            'enetByteTransportMatchesCurrentFile': bytes(raw['enetUtf8Bytes']) == enet_path.read_bytes(),
            'drcErrors': sum(row.get('type') == 'error' for row in raw_drc),
            'drcWarnings': sum(row.get('type') == 'warn' for row in raw_drc),
            'drcMatchesPublishedProof': raw_drc == m['priorDrcOnly']['errors'] + m['priorDrcOnly']['warnings']}
    if name in ('REG', 'PUMP'):
        contract_path = POWER / ('稳压模块/原理图准备/design-plan.json' if name == 'REG'
                                 else '水泵驱动模块/原理图准备/原理图连接契约.json')
        plan = load(contract_path)
        expected = {c['ref']: c for c in plan['components' if name == 'REG' else 'parts']}
        if name == 'REG':
            for boundary in plan.get('systemBoundaries', []):
                assert boundary['ref'] not in expected
                expected[boundary['ref']] = boundary
        differences, pin_rows = [], []
        for ref, part in expected.items():
            if ref not in by_ref:
                differences.append({'ref': ref, 'reason': 'missing_enet_component'})
                continue
            actual_pins = by_ref[ref]['pinInfoMap']
            if set(part['pins']) != set(actual_pins):
                differences.append({'ref': ref, 'reason': 'pin_set', 'expected': list(part['pins']), 'actual': list(actual_pins)})
            for pin, net in part['pins'].items():
                actual_net = actual_pins.get(pin, {}).get('net')
                ok = str(net or '').strip().upper() == str(actual_net or '').strip().upper() and pin in actual_pins
                row = {'ref': ref, 'pin': pin, 'expected': net, 'actual': actual_net, 'match': ok}
                pin_rows.append(row)
                if not ok:
                    differences.append(row)
        m['contract'] = {'path': str(contract_path.relative_to(ROOT)), 'sha256': digest(contract_path),
                         'parts': len(expected), 'pins': len(pin_rows), 'differences': differences,
                         'extraEnetComponents': sorted(set(by_ref)-set(expected)), 'pinRows': pin_rows}
    evidence['modules'][name] = m

evidence['totalUniqueSchematicComponents'] = sum(m['components'] for m in evidence['modules'].values())
evidence['totalUniqueSchematicPins'] = sum(m['pins'] for m in evidence['modules'].values())

# 重新计算边界值；没有使用典型公式冒充实测/保证值。
evidence['calculations'] = {
    'regulatorPmosCollector10kW': {str(v): (v-10)**2/10000 for v in [37.8, 43, 69.4]},
    'regulatorInductor20uH80AEnergyJ': 0.5*20e-6*80**2,
    'regulatorIdeal160uFCapRiseFrom19_6V': math.sqrt(19.6**2 + 20e-6*80**2/160e-6),
    'regulatorBalanceAt2_48V120ohmUpperA': 2.48/120,
    'pumpVoltageDividerRatio': (110000+110000+10000)/10000,
    'pumpDutyAtFull37_8V': 21/37.8,
    'pumpTripTypicalA': 1.65/(1500*212e-6),
    'notes': 'P管用10V名义齐纳和理想饱和NPN估算；断流能量假设全进160uF，仅作边界筛查；水泵斩波点为典型值。'}

extensions = {'.esch2', '.eprj3', '.kicad_sch', '.kicad_pcb', '.schdoc', '.pcbdoc', '.sch', '.brd', '.enet'}
inventory = []
for top in ROOT.iterdir():
    if not top.is_dir() or top.name.startswith('.'):
        continue
    for p in top.rglob('*'):
        if p.is_file() and p.suffix.lower() in extensions:
            rel = p.relative_to(ROOT).as_posix()
            if '/external/' in rel or '/tools/' in rel or '/upstream/' in rel:
                continue
            inventory.append(rel)
evidence['designFileInventory'] = sorted(inventory)

failures = []
for key, m in evidence['modules'].items():
    if not m['allPriorSourceHashesMatch'] or not m['enetHashMatchesPriorProof'] or m['missingNativeRefs']:
        failures.append(key + ': source_or_export_binding')
    if not all(v is True for k, v in m['rawArchiveVerification'].items()
               if k in ('componentCountMatches', 'pinCountMatches', 'enetByteTransportMatchesCurrentFile', 'drcMatchesPublishedProof')):
        failures.append(key + ': raw_archive_binding')
    if m.get('contract', {}).get('differences') or m.get('contract', {}).get('extraEnetComponents'):
        failures.append(key + ': pin_contract')
    for item in m['nativeVsEnetMpnDifferences']:
        if key == 'REG' and item == {'ref': 'SYS_BAT_BUS', 'native': '', 'enet': '2PINA-power'}:
            continue  # 非采购系统边界载体的库默认属性。
        failures.append(key + ': mpn:' + item['ref'])
if not all(v['hashMatchesProof'] for v in evidence['priorArchives'].values()):
    failures.append('prior_archive_hash')
evidence['staticAuditStatus'] = 'PASS_WITH_DECLARED_VIRTUAL_INTERFACE_ALIAS' if not failures else 'FAIL'
evidence['auditFailures'] = failures

OUT.mkdir(parents=True, exist_ok=True)
(OUT / '现行输入复核.json').write_text(json.dumps(evidence, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
for key, m in evidence['modules'].items():
    print(key, m['components'], m['pins'], 'source_hashes=',m['allPriorSourceHashesMatch'],
          'enet_hash=',m['enetHashMatchesPriorProof'], 'mpn_diff=',len(m['nativeVsEnetMpnDifferences']),
          'contract_diff=',len(m.get('contract',{}).get('differences',[])))
print('TOTAL',evidence['totalUniqueSchematicComponents'],evidence['totalUniqueSchematicPins'])
print(evidence['staticAuditStatus'])
sys.exit(1 if failures else 0)
