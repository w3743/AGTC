"""Operate only a separately copied review candidate; never save the live project."""
import hashlib
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

root=Path(__file__).resolve().parents[1]
src=root/'eda工程文件/sc-regulator'
folder=root/'检查/候选_预充与超容隔离_R1'
clone=folder/'sc-regulator'
exe=Path(r'C:\Program Files\lceda-pro\lceda-pro.exe')
folder.mkdir(exist_ok=True)
if not clone.exists():
    shutil.copytree(src,clone)
    def hashes(p):
        return {x.relative_to(p).as_posix():hashlib.sha256(x.read_bytes()).hexdigest() for x in sorted(p.rglob('*')) if x.is_file()}
    assert hashes(src)==hashes(clone)
    (folder/'复制源哈希.json').write_text(json.dumps(hashes(src),ensure_ascii=False,indent=2),encoding='utf-8')

def call(*args,timeout=60):
    p=subprocess.run([str(exe),*args],capture_output=True,timeout=timeout)
    if p.returncode:
        raise RuntimeError(p.stdout.decode('utf-8','replace')+' '+p.stderr.decode('utf-8','replace'))
    d=json.loads(p.stdout.decode('utf-8-sig'))
    if not d.get('ok'): raise RuntimeError(str(d))
    return d['value']

path=(clone/'sc-regulator.eprj3').resolve()
before={x['sessionId'] for x in call('session','list')}
proc=subprocess.Popen([str(exe),'open','--path',str(path),'--headless','true'],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
sid=None
try:
    for _ in range(60):
        options=[x for x in call('session','list') if x['sessionId'] not in before and x.get('origin')=='cli'
                 and x.get('status')=='alive' and Path(x.get('path','')).resolve()==path]
        if len(options)==1:
            sid=options[0]['sessionId'];break
        time.sleep(.5)
    if sid is None: raise RuntimeError('isolated session unavailable')
    time.sleep(6)
    code=Path(sys.argv[1]).read_text(encoding='utf-8')
    ret=call('invoke','--session',sid,'--ext-uuid','eda','--timeout','60000','--code',code,timeout=70)
    name=sys.argv[2] if len(sys.argv)>2 else 'api结果.json'
    (folder/name).write_text(json.dumps(ret,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(ret,ensure_ascii=False)[:12000])
finally:
    if sid:
        call('session','close','--session',sid,'--destroy')
