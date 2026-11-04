"""隔离副本四页官方只读复核，不保存或关闭用户会话。"""
import hashlib
import json
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'eda工程文件' / 'sc-regulator'
CHECKS = ROOT / '检查'
EXE = Path(r'C:\Program Files\lceda-pro\lceda-pro.exe')
PAGES = ['9546dca0bc8bce7f', '28c1f3740a9b8e02', '28c1f3740a9b8e03', '28c1f3740a9b8e04']

def call(*args, timeout=50):
    p = subprocess.run([str(EXE), *args], capture_output=True, timeout=timeout)
    if p.returncode:
        raise RuntimeError(f'CLI exit={p.returncode}; stdout={p.stdout.decode("utf-8", "replace")}; stderr={p.stderr.decode("utf-8", "replace")}')
    result = json.loads(p.stdout.decode('utf-8-sig'))
    if not result.get('ok'):
        raise RuntimeError(str(result))
    return result['value']

def digests(folder):
    return {p.relative_to(folder).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(folder.rglob('*')) if p.is_file()}

before = digests(SOURCE)
sid = None
diagnostic = None
with tempfile.TemporaryDirectory(prefix='schematic_readonly_', dir=CHECKS) as temp:
    clone = Path(temp) / 'sc-regulator'
    shutil.copytree(SOURCE, clone)
    assert before == digests(clone), 'copy mismatch'
    before_ids = {x['sessionId'] for x in call('session', 'list')}
    proc = subprocess.Popen([str(EXE), 'open', '--path', str(clone / 'sc-regulator.eprj3'), '--headless', 'true'],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        for _ in range(60):
            sessions = call('session', 'list')
            matches = [s for s in sessions if s['sessionId'] not in before_ids and s.get('origin') == 'cli'
                       and s.get('status') == 'alive' and Path(s.get('path', '')).resolve() == (clone/'sc-regulator.eprj3').resolve()]
            if len(matches) == 1:
                sid = matches[0]['sessionId']
                break
            time.sleep(0.5)
        if sid is None:
            raise RuntimeError('no uniquely identified owned headless session')
        time.sleep(6)  # session alive does not imply the editor project tree is ready
        diagnostic = call('invoke', '--session', sid, '--ext-uuid', 'eda', '--timeout', '6000', '--code',
                          'return {project:await eda.dmt_Project.getCurrentProjectInfo(),doc:await eda.dmt_SelectControl.getCurrentDocumentInfo(),tree:await eda.dmt_EditorControl.getSplitScreenTree()};',timeout=12)
        code = """const ids=['9546dca0bc8bce7f','28c1f3740a9b8e02','28c1f3740a9b8e03','28c1f3740a9b8e04'];
const wait=ms=>new Promise(r=>setTimeout(r,ms));const result=[];
for(const id of ids){await eda.dmt_EditorControl.openDocument(id);let doc;
for(let k=0;k<50;k++){await wait(200);doc=await eda.dmt_SelectControl.getCurrentDocumentInfo();if(doc?.uuid===id)break;}
if(doc?.uuid!==id)throw Error('page context '+id);const drc=await eda.sch_Drc.check(true,false,true);
const svg=await eda.sch_ManufactureData.getExportDocumentFile(undefined,'SVG',undefined,'Current Schematic Page');
if(!svg)throw Error('SVG export '+id);result.push({id,doc,drc,svg:await svg.text()});}
const net=await eda.sch_ManufactureData.getNetlistFile(undefined,'EasyEDA');
const bom=await eda.sch_ManufactureData.getBomFile(undefined,'csv');
return {project:await eda.dmt_Project.getCurrentProjectInfo(),pages:result,netlist:await net.text(),bom:bom?await bom.text():null};"""
        result = call('invoke', '--session', sid, '--ext-uuid', 'eda', '--timeout', '60000', '--code', code, timeout=70)
        net_text = result.pop('netlist')
        net = json.loads(net_text)
        bom = result.pop('bom')
        for i, page in enumerate(result['pages'], 1):
            svg_text = page.pop('svg')
            svg_path = CHECKS/f'本轮磁盘读回_{i:02d}.svg'
            svg_path.write_text(svg_text, encoding='utf-8')
            page['svg'] = str(svg_path.relative_to(ROOT))
            page['svg_sha256'] = hashlib.sha256(svg_path.read_bytes()).hexdigest()
        net_path = CHECKS/'本轮磁盘读回.enet'
        net_path.write_text(net_text, encoding='utf-8')
        bom_path = CHECKS/'本轮磁盘读回BOM.csv'
        if bom is not None:
            bom_path.write_text(bom, encoding='utf-8')
        result.update({'method':'read-only copied project; official CLI; no save', 'source_sha256':before,
                       'source_unchanged_during_read':before == digests(SOURCE), 'official_component_count':len(net.get('components',{})),
                       'official_bom_path':str(bom_path.relative_to(ROOT)) if bom else None,
                       'official_bom_sha256':hashlib.sha256(bom_path.read_bytes()).hexdigest() if bom else None,
                       'official_netlist_path':str(net_path.relative_to(ROOT)), 'official_netlist_sha256':hashlib.sha256(net_path.read_bytes()).hexdigest(),
                       'official_netlist_components':net.get('components',{}), 'session_owned':True})
        target = CHECKS/'本轮四页官方只读复核.json'
        target.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
        print(json.dumps({'path':str(target), 'source_unchanged':result['source_unchanged_during_read'],
                          'pages':[{ 'id':p['id'],'drc':p['drc']} for p in result['pages']],
                          'component_count':result['official_component_count']},ensure_ascii=False))
    except Exception as exc:
        failure={'method':'read-only copied project; official CLI; no save', 'status':'NOT_COMPLETED',
                 'source_sha256':before, 'source_unchanged_during_read':before == digests(SOURCE),
                 'session_owned':sid is not None, 'binding_diagnostic':diagnostic, 'error':str(exc)}
        (CHECKS/'本轮四页官方只读复核失败.json').write_text(json.dumps(failure,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
        print(json.dumps({'status':'NOT_COMPLETED','error':str(exc)},ensure_ascii=False))
    finally:
        if sid is not None:
            call('session', 'close', '--session', sid, '--destroy')
