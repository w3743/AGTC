"""显式清单归档；逐文件SHA读回后供PowerShell按清单清理。不会修改EDA。"""
from pathlib import Path
import argparse, collections, hashlib, json, os, stat, time, zipfile

ROOT = Path(__file__).resolve().parents[2]
RECEIPT = ROOT / '归档/目录整理.json'
PREFIX = '20261016_目录整理/'
BOARDS = ('BMS模块', 'FOC驱动模块', '稳压模块')

def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024), b''): h.update(b)
    return h.hexdigest()

def checked(rel):
    original=ROOT/rel
    for q in (original,*original.parents):
        if q==ROOT: break
        if q.lstat().st_file_attributes & stat.FILE_ATTRIBUTE_REPARSE_POINT: raise RuntimeError('拒绝reparse: '+str(q))
    p = original.resolve(strict=True)
    p.relative_to(ROOT)
    for q in (p, *p.parents):
        if q == ROOT: break
        if q.lstat().st_file_attributes & stat.FILE_ATTRIBUTE_REPARSE_POINT:
            raise RuntimeError('拒绝reparse: '+str(q))
    if not p.is_file(): raise RuntimeError('仅处理显式文件: '+str(p))
    return p

def write(obj):
    RECEIPT.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

def archive(dry=False):
    if RECEIPT.exists(): raise RuntimeError('已有清单，禁止重复执行')
    files=set(); keep=set(); notes=[]; backups=set()
    for plan in sorted((ROOT/'.agents').glob('cleanup_*_plan.json')):
        data=json.loads(plan.read_text(encoding='utf-8-sig'))
        def expand(items):
            out=set()
            for rel in items:
                p=ROOT/rel
                if p.is_file(): out.add(p.relative_to(ROOT).as_posix())
                elif p.is_dir(): out.update(x.relative_to(ROOT).as_posix() for x in p.rglob('*') if x.is_file())
                elif '**' in rel:
                    # Only literal directory/** patterns, never prose annotations.
                    parent=ROOT/rel.removesuffix('/**')
                    if rel.endswith('/**') and parent.is_dir(): out.update(x.relative_to(ROOT).as_posix() for x in parent.rglob('*') if x.is_file())
            return out
        files.update(expand(data.get('archive_files',[]))); keep.update(expand(data.get('preserve',[]))); notes.extend(data.get('notes',[])); backups.update(expand(data.get('backup_files',[])))
    files.difference_update(keep)
    files.update(backups)
    protected={}
    for board in BOARDS:
        for p in (ROOT/board).rglob('*'):
            rel=p.relative_to(ROOT).as_posix()
            if p.is_file() and ('/eda工程文件/' in rel or (board=='BMS模块' and (rel.endswith('.eprj3') and '/' not in rel[len(board)+1:] or rel.startswith(board+'/sch/'))) or rel.endswith('/原理图准备/BOM.csv')):
                protected[rel]=sha(p)
    if set(protected)&files: raise RuntimeError('清理命中正式工程或BOM')
    rows=[]; groups=collections.defaultdict(list)
    for rel in sorted(files):
        p=checked(rel)
        if '/.git/' in '/'+rel or rel.endswith('/.git'): raise RuntimeError('清理命中Git')
        group=rel.split('/')[0]
        if rel.startswith('归档/原始材料/'):
            old=rel.split('/')[2]
            if old in BOARDS: group=old
            elif old.startswith(('BMS','bms-')): group='BMS模块'
        dest=(group+'/归档/历史与调试.zip') if group in BOARDS else '归档/工程过程.zip'
        if rel==dest: raise RuntimeError('归档自身')
        groups[dest].append(rel)
    if dry:
        print(json.dumps({'protected_files':len(protected),'groups':{k:{'files':len(v),'bytes':sum((ROOT/p).stat().st_size for p in v)} for k,v in groups.items()}},ensure_ascii=False,indent=2));return
    model_hashes={}
    for board in BOARDS:
        for p in (ROOT/board/'仿真').rglob('*'):
            if p.is_file() and p.relative_to(ROOT).as_posix() not in files and p.suffix.lower() not in ('.md','.pyc'):
                model_hashes[p.relative_to(ROOT).as_posix()]=sha(p)
    baseline=list(p for p in ROOT.rglob('*') if p.is_file())
    result={'date':'2026-10-16','root':str(ROOT),'status':'ARCHIVING','restore':'每条记录的archive/member为恢复位置；本轮成员以电力根目录相对路径保存。旧成员命名不改。','protected_sha256':protected,'retained_model_sha256':model_hashes,'before':{'files':len(baseline),'bytes':sum(p.stat().st_size for p in baseline)},'notes':notes,'files':rows,'archives':{}}
    write(result)
    for dest,rels in groups.items():
        zpath=ROOT/dest; zpath.parent.mkdir(parents=True,exist_ok=True)
        print('ARCHIVE',dest,len(rels),flush=True)
        prior=sha(zpath) if zpath.exists() else None
        dedup={}
        with zipfile.ZipFile(zpath,'a',compression=zipfile.ZIP_DEFLATED,compresslevel=1,allowZip64=True) as z:
            names=set(z.namelist())
            for i,rel in enumerate(rels):
                p=checked(rel); h=sha(p); size=p.stat().st_size
                member=dedup.get(h)
                if member is None:
                    member=PREFIX+rel
                    if member in names: raise RuntimeError('成员冲突: '+member)
                    z.write(p,member); dedup[h]=member; names.add(member)
                rows.append({'path':rel,'bytes':size,'sha256':h,'archive':dest,'member':member,'readback':False,'cleanup':rel not in backups})
                if i%100==0 or size>100*1048576: print('  saved',i+1,'/',len(rels),rel,flush=True)
        with zipfile.ZipFile(zpath) as z:
            verified={}
            for row in (x for x in rows if x['archive']==dest):
                member=row['member']
                if member not in verified:
                    h=hashlib.sha256()
                    with z.open(member) as f:
                        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
                    verified[member]=h.hexdigest()
                if verified[member]!=row['sha256'] or sha(checked(row['path']))!=row['sha256']:
                    raise RuntimeError('源或归档SHA不匹配: '+row['path'])
                row['readback']=True
        result['archives'][dest]={'before_sha256':prior,'after_sha256':sha(zpath),'files':len(rels),'unique_entries':len(dedup),'all_sha256_readback':True}
        write(result)
        print('VERIFIED',dest,flush=True)
    result['status']='VERIFIED_READY_FOR_POWERSHELL_CLEANUP'
    result['file_count']=len(rows);result['source_bytes']=sum(x['bytes'] for x in rows)
    write(result)
    print('READY',len(rows),result['source_bytes'],flush=True)

def verify():
    d=json.loads(RECEIPT.read_text(encoding='utf-8'))
    fallback=ROOT/'.agents/cleanup_empty_moves.json'
    if fallback.exists():
        d['empty_directory_fallback']={'reason':'执行层策略拒绝额外删除空目录，按AGENTS约定可恢复移动；未换工具绕过删除。','moves':json.loads(fallback.read_text(encoding='utf-8-sig')),'verification':'源路径已不存在；目的树均为普通空目录，无文件和reparse point。'}
    failures=[]
    for rel,h in d['protected_sha256'].items():
        if not (ROOT/rel).is_file() or sha(ROOT/rel)!=h: failures.append(rel)
    remaining=[x['path'] for x in d['files'] if x.get('cleanup',True) and (ROOT/x['path']).exists()]
    d['protected_unchanged']=not failures;d['protected_failures']=failures
    model_failures=[p for p,h in d.get('retained_model_sha256',{}).items() if not (ROOT/p).is_file() or sha(ROOT/p)!=h]
    d['retained_models_unchanged']=not model_failures;d['model_failures']=model_failures
    d['remaining_original_files']=remaining
    archive_failures=[]
    for dest,info in d['archives'].items():
        actual=sha(ROOT/dest)
        if actual!=info.get('documents_sha256',info['after_sha256']): archive_failures.append(dest)
        info['final_sha256']=actual
    d['archive_failures']=archive_failures
    after=list(p for p in ROOT.rglob('*') if p.is_file())
    d['after']={'files':len(after),'bytes':sum(p.stat().st_size for p in after)}
    d['status']='PASS' if not failures and not remaining and not model_failures and not archive_failures else 'HOLD'
    write(d);print(json.dumps({'status':d['status'],'protected':len(d['protected_sha256']),'remaining':len(remaining)},ensure_ascii=False))
    if failures or remaining or model_failures or archive_failures: raise SystemExit(1)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['archive','verify','plan']);a=p.parse_args()
    verify() if a.mode=='verify' else archive(a.mode=='plan')
