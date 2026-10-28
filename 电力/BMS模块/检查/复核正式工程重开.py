"""Read-only official CLI reopen; no netlist/BOM export and no GUI access."""
from pathlib import Path
import importlib.util, json, hashlib, datetime, time

BASE = Path(__file__).resolve().parent
M = BASE.parent
spec = importlib.util.spec_from_file_location('helper', BASE / 'tmp/layout_continue.py')
h = importlib.util.module_from_spec(spec)
spec.loader.exec_module(h)

def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

paths = [M / 'BMS-9S4P.eprj3', M / 'sch/Schematic1/P1.esch2', M / 'sch/Schematic1/Schematic1.ecfg']
before = {p.relative_to(M).as_posix(): digest(p) for p in paths}
expected = json.loads((BASE / 'Layout验证/current-snapshot.json').read_text(encoding='utf-8-sig'))
opened = h.command(['open', '--path', paths[0].as_posix(), '--headless', 'true'], 'root-after-cleanup-open')
sid = opened['sessionId']
try:
    time.sleep(6)
    h.invoke("let d;try{d=await eda.dmt_SelectControl.getCurrentDocumentInfo();}catch(e){}if(d?.uuid!=='350a4904c4ec87b3')await eda.dmt_EditorControl.openDocument('350a4904c4ec87b3');return {document:await eda.dmt_SelectControl.getCurrentDocumentInfo()};", 'root-after-cleanup-page', session=sid)
    state = h.invoke('''
const all=await eda.sch_PrimitiveComponent.getAll('part');
const rows=all.filter(c=>c.getState_Designator()).map(c=>({ref:c.getState_Designator(),id:c.getState_PrimitiveId(),mpn:c.getState_ManufacturerId()}));
return {document:await eda.dmt_SelectControl.getCurrentDocumentInfo(),rows};
''', 'root-after-cleanup-state', session=sid)
    pins = []
    for i in range(0, len(state['rows']), 25):
        ids = [x['id'] for x in state['rows'][i:i+25]]
        code = '''const out=[];for(const id of IDS){const c=await eda.sch_PrimitiveComponent.getAllPinsByPrimitiveId(id);out.push({id,pins:(c||[]).map(p=>({number:p.getState_PinNumber(),name:p.getState_PinName(),x:p.getState_X(),y:p.getState_Y(),nc:p.getState_NoConnected()}))});}return out;'''.replace('IDS', json.dumps(ids))
        pins.extend(h.invoke(code, f'root-after-cleanup-pins-{i//25+1}', session=sid))
    drc = h.invoke('return await eda.sch_Drc.check(true,false,true);', 'root-after-cleanup-drc', session=sid)
    expected_refs = {x['ref']: x for x in expected['rows']}
    actual_refs = {x['ref']: x for x in state['rows']}
    pins_by_id = {x['id']: x['pins'] for x in pins}
    diffs = []
    for ref in sorted(expected_refs.keys() | actual_refs.keys()):
        e, a = expected_refs.get(ref), actual_refs.get(ref)
        if e is None or a is None:
            diffs.append({'ref': ref, 'missing_or_extra': True})
            continue
        if e['mpn'] != a['mpn']:
            diffs.append({'ref': ref, 'mpn': [e['mpn'], a['mpn']]})
        def signature(ps):
            return sorted((str(p['number']), p['name'], p['x'], p['y'], p['nc']) for p in ps)
        if signature(e['pins']) != signature(pins_by_id.get(a['id'], [])):
            diffs.append({'ref': ref, 'pins_differ': True})
    after = {p.relative_to(M).as_posix(): digest(p) for p in paths}
    proof = {'checked_at': datetime.datetime.now().isoformat(timespec='seconds'), 'method': 'official CLI headless open, API pin batches and DRC; no GUI or CAD export', 'project': paths[0].as_posix(), 'source_sha256_before': before, 'source_sha256_after': after, 'source_unchanged': before == after, 'components': len(state['rows']), 'pins': sum(len(x['pins']) for x in pins), 'snapshot_differences': diffs, 'snapshot_match': not diffs, 'drc': drc, 'warning_details_reviewed': False}
    proof['status'] = 'PASS_READBACK' if proof['snapshot_match'] and proof['source_unchanged'] else 'FAIL_READBACK'
    (BASE / 'Layout验证/root-after-cleanup-reopen.json').write_text(json.dumps(proof, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(proof, ensure_ascii=False, indent=2))
finally:
    h.command(['session', 'close', '--session', sid, '--destroy'], 'root-after-cleanup-close')
