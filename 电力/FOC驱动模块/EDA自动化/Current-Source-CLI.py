"""官方CLI操作当前FOC原理图；备份后启动，仅管理本工具创建的会话。"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import zipfile

ROOT = Path(__file__).resolve().parents[1]
CLI = r'D:\lceda-pro\lceda-pro.exe'
NATIVE = ROOT/'eda工程文件/FOC_G431_Driver'
PROJECT = NATIVE/'FOC_G431_Driver.eprj3'
OUT = ROOT/'检查/当前源'
STATE = OUT/'当前整改会话.json'
PAGE = '90aa6c46493624cf'
ENV = dict(os.environ, PYTHONIOENCODING='utf-8')

def sha(raw): return hashlib.sha256(raw).hexdigest()
def write(name, value):
    (OUT/name).write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
def call(*args, limit=80):
    p = subprocess.run([CLI, *args], capture_output=True, env=ENV, timeout=limit)
    raw = p.stdout.decode('utf-8-sig').strip()
    if not raw:
        if args[:2] == ('session','list') and p.returncode == 0: return []
        raise RuntimeError({'exitCode':p.returncode,'emptyStdout':True,'stderr':p.stderr.decode('utf-8',errors='replace')[:1200]})
    d = json.loads(raw)
    if not d.get('ok'): raise RuntimeError(d)
    return d['value']
def invoke(sid, code):
    return call('invoke','--session',sid,'--ext-uuid','eda','--timeout','30000','--code',code,limit=45)

def backup():
    files = [p for p in NATIVE.rglob('*') if p.is_file()]
    files += [p for p in (ROOT/'原理图准备').iterdir() if p.is_file()]
    files += [p for p in (ROOT/'eda工程文件/导出/当前交付').iterdir() if p.is_file()]
    rows = []
    stamp = '20261021_BOV_reference_before_'+str(time.time_ns())
    archive = ROOT/'归档/历史与调试.zip'
    existed = archive.exists()
    archive.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive,'a',zipfile.ZIP_DEFLATED) as z:
        for f in files:
            raw = f.read_bytes(); entry=stamp+'/'+f.relative_to(ROOT).as_posix()
            z.writestr(entry, raw); rows.append({'path':f.relative_to(ROOT).as_posix(),'entry':entry,'sha256':sha(raw)})
    with zipfile.ZipFile(archive) as z:
        for r in rows: assert sha(z.read(r['entry']))==r['sha256']
    result={'archive':archive.as_posix(),'prefix':stamp,'files':rows,'readback':'PASS',
            'created_new_archive':not existed,
            'scope':'本轮修改前快照；新建ZIP不恢复或替代此前已缺失的历史归档'}
    write('当前整改备份.json',result)
    return {'files':len(rows),'archiveReadback':'PASS'}

def start():
    previous = {x['sessionId'] for x in call('session','list')}
    with (OUT/'当前整改打开.json').open('wb') as stdout, (OUT/'当前整改打开错误.txt').open('wb') as stderr:
        proc = subprocess.Popen([CLI,'open','--path',PROJECT.as_posix(),'--headless','true'],stdout=stdout,stderr=stderr,env=ENV,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        try: proc.wait(timeout=25)
        except subprocess.TimeoutExpired: pass
    try:
        data=json.loads((OUT/'当前整改打开.json').read_text(encoding='utf-8-sig'))
        assert data['ok'];sid=data['value']['sessionId']
    except (ValueError,KeyError,AssertionError):
        new=[x for x in call('session','list') if x['sessionId'] not in previous and x.get('status')=='alive']
        if len(new)!=1:raise RuntimeError('无法唯一识别新会话，不接管其他窗口')
        sid=new[0]['sessionId']
    write(STATE.name,{'session':sid,'owned':True,'project':PROJECT.as_posix(),'page':PAGE})
    result=invoke(sid,"const o=await eda.dmt_EditorControl.openDocument('"+PAGE+"');const p=await eda.dmt_Schematic.getCurrentSchematicPageInfo();return {opened:o,page:p?.uuid,name:p?.name};")
    assert result['page'] == PAGE
    write('当前整改项目读回.json',result)
    return {'session':sid,'project':result}

if __name__=='__main__':
    OUT.mkdir(parents=True, exist_ok=True)
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('action',choices=['backup','start','run','close']);ap.add_argument('script',nargs='?');ap.add_argument('output',nargs='?');a=ap.parse_args()
    if a.action=='backup':result=backup()
    elif a.action=='start':result=start()
    else:
        s=json.loads(STATE.read_text(encoding='utf-8'));assert s['owned'];sid=s['session']
        if a.action=='close':result=call('session','close','--session',sid,'--destroy')
        else:
            if not a.script or not a.output:ap.error('run需要脚本和输出文件名')
            result=invoke(sid,Path(a.script).read_text(encoding='utf-8-sig'));write(a.output,result)
    if isinstance(result,dict) and any(k in result for k in ('base64','enet','content','components')):
        print(json.dumps({'resultSaved':a.output,'keys':list(result),'largePayloadPrinted':False},ensure_ascii=True))
    else:
        print(json.dumps(result,ensure_ascii=True)[:3000])
