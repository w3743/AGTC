"""Official EasyEDA API transaction for the four approved six-resistor chains.

Native files are only copied or written by the editor, never edited as text.
"""
import argparse
import hashlib
import importlib.util
import json
import shutil
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
MOD = HERE.parent
WORK = HERE / 'tmp' / 'approved_resistor_work'
NATIVE = MOD / 'eda工程文件'
OUT = HERE / '电阻归并原生落实'
CLI = Path(r'D:\lceda-pro\lceda-pro.exe')
spec = importlib.util.spec_from_file_location('builder', HERE / '单页eprj3施工.py')
builder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(builder)
OLD = ['R_BAL_MAINCLR_BASE', 'R_BAL_ENABLE_BASE', 'R_LD', 'R_PACK']

def dump(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')

def cli(args):
    p = subprocess.run([str(CLI), *args], capture_output=True, timeout=70)
    s = p.stdout.decode('utf-8-sig').strip()
    data = json.loads(s)
    if p.returncode or not data.get('ok'):
        raise RuntimeError(data)
    return data['value']

def api(session, code):
    return builder.invoke(CLI, session, code)

def export(session, name):
    init = api(session, "const f=await eda.sch_ManufactureData.getNetlistFile('approved.enet','EasyEDA');if(!f)throw Error('no netlist');globalThis.APPROVED_ENET=await f.text();return {length:globalThis.APPROVED_ENET.length};")
    value = ''.join(api(session, f'return globalThis.APPROVED_ENET.slice({i},{i+24000});') for i in range(0, init['length'], 24000))
    obj = json.loads(value)
    dump(OUT / name, obj)
    return obj

def prepare():
    if WORK.exists():
        raise RuntimeError('work directory already exists; reuse the recorded session')
    shutil.copytree(NATIVE, WORK)
    hashes = {str(p.relative_to(NATIVE)): hashlib.sha256(p.read_bytes()).hexdigest() for p in NATIVE.rglob('*') if p.is_file()}
    assert all(hashlib.sha256((WORK / k).read_bytes()).hexdigest() == v for k,v in hashes.items())
    dump(OUT / '原生修改前文件哈希.json', hashes)
    opened = cli(['open','--path',str(WORK/'BMS-9S4P.eprj3'),'--headless','true'])
    dump(OUT/'session.json', opened)
    print(json.dumps(opened,ensure_ascii=False))

def inspect(session):
    r=api(session, r"""
const pages=await eda.dmt_Schematic.getAllSchematicPagesInfo();
if(pages.length!==1||pages[0].name!=='P1')throw Error('unexpected page');
await eda.dmt_EditorControl.openDocument(pages[0].uuid);
const all=await eda.sch_PrimitiveComponent.getAll();
const physical=all.filter(c=>c.getState_Designator());
const wanted=physical.filter(c=>['R_BAL_MAINCLR_BASE','R_BAL_ENABLE_BASE','R_LD','R_PACK'].includes(c.getState_Designator())||['RMC06031K1%N','ABD03K1000FT'].includes(c.getState_ManufacturerId()));
const rows=[];
for(const c of wanted){rows.push({ref:c.getState_Designator(),id:c.getState_PrimitiveId(),mpn:c.getState_ManufacturerId(),name:c.getState_Name(),x:c.getState_X(),y:c.getState_Y(),rotation:c.getState_Rotation(),symbol:c.getState_Symbol(),footprint:c.getState_Footprint(),other:c.getState_OtherProperty(),pins:await eda.sch_PrimitiveComponent.getAllPinsByPrimitiveId(c.getState_PrimitiveId())});}
return {page:pages[0],count:physical.length,rows};
""")
    dump(OUT/'修改前API器件.json',r)
    net=export(session,'修改前官方网表.enet')
    assert len(net['components'])==213
    print(json.dumps(r,ensure_ascii=False))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','inspect']);p.add_argument('--session');a=p.parse_args()
    if a.action=='prepare': prepare()
    else: inspect(a.session)
