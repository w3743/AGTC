"""Bind final runtime metadata without changing electrical evidence or CAD."""
from pathlib import Path
import copy, datetime, hashlib, json, sys, zipfile
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tmp'))
import drc_fix as d
M = d.h.M
L = M / '检查/Layout验证'
E = M / '检查/DRC修复'
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p): return json.loads(p.read_text(encoding='utf-8-sig'))
def write(p, v): p.write_text(json.dumps(v, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')

baseline = '20261016_012150_用户DRC修复前'
root = M / 'BMS-9S4P.eprj3'
cfg = M / 'sch/Schematic1/Schematic1.ecfg'
with zipfile.ZipFile(M / '归档/历史与调试.zip') as z:
    old_root = json.loads(z.read(baseline + '/BMS-9S4P.eprj3'))
    old_cfg = z.read(baseline + '/sch/Schematic1/Schematic1.ecfg').decode('utf-8-sig')
current_root = read(root)
paths = []
def changes(a, b, path=''):
    if isinstance(a, dict) and isinstance(b, dict):
        assert a.keys() == b.keys(), path
        for k in a: changes(a[k], b[k], path + '/' + k)
    elif a != b:
        paths.append(path)
changes(old_root, current_root)
assert set(paths) == {
    '/profile/schematics/1c3e06662fdc8e9a/version',
    '/profile/schematics/1c3e06662fdc8e9a/updateTime',
    '/profile/sheets/350a4904c4ec87b3/version',
    '/profile/sheets/350a4904c4ec87b3/updateTime',
}, paths
def cfg_rows(text):
    result=[]
    for line in text.splitlines():
        head, body = line.split('||',1)
        h=json.loads(head); v=json.loads(body.rstrip('|'))
        if h['type']=='DOCHEAD': v.pop('client',None)
        result.append((h,v))
    return result
assert cfg_rows(old_cfg) == cfg_rows(cfg.read_text(encoding='utf-8'))
pub=read(L/'当前发布.json'); publication=read(L/'publication.json')
assert sha(M/'sch/Schematic1/P1.esch2') == pub['sources']['sch/Schematic1/P1.esch2']['sha256']
assert sha(M/'原理图准备/BOM.csv') == pub['sources']['原理图准备/BOM.csv']['sha256']
assert read(E/'restored-readback.json') == read(E/'final-snapshot.json')
archive = d.h.archive([root,cfg,L/'当前发布.json',L/'publication.json'], 'DRC最终运行元数据绑定前')
old_hashes={p:v['sha256'] for p,v in pub['sources'].items()}
for p in pub['sources']:
    pub['sources'][p]={'sha256':sha(M/p),'bytes':(M/p).stat().st_size}
pub['known_root_path_runtime']['subsequent_runtime_metadata'] = '检查/DRC修复/final-runtime-metadata.json'
pub['known_root_path_runtime']['note'] = 'P1/BOM and all snapshot pins/MPN unchanged after final trial rollback. Later root version/updateTime and ecfg client metadata were rebound after session close; see supplement.'
publication['files']={p:pub['sources'][p] for p in publication['files']}
publication['runtimeMetadataSupplement']='检查/DRC修复/final-runtime-metadata.json'
write(E/'final-runtime-metadata.json',{
    'checked_at':datetime.datetime.now().isoformat(),
    'status':'PASS_NON_ELECTRICAL_METADATA_REBIND',
    'session_closed':True,'archive':archive,
    'previous_published_sha256':old_hashes,
    'final_sha256':{p:v['sha256'] for p,v in pub['sources'].items()},
    'comparison_baseline_archive':baseline,
    'root_changed_paths_from_baseline':paths,
    'ecfg_semantically_equal_except_DOCHEAD_client':True,
    'p1_and_bom_unchanged':True,'final_snapshot_equals_restored_readback':True,
    'drc':read(E/'restored-readback.json')['drc'],
    'warning_details_reviewed':False,
    'note':'Original reopen report remains a historical observation at its recorded time. No electrical evidence or CAD was altered by this metadata binding.',
})
write(L/'当前发布.json',pub);write(L/'publication.json',publication)
print(json.dumps({'status':'PASS','root':sha(root),'cfg':sha(cfg)},ensure_ascii=False))
