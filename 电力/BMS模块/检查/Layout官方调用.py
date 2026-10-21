"""Official editor on an isolated, hash-checked copy for the Layout handoff."""
from pathlib import Path
import hashlib, json, shutil, subprocess, sys
sys.stdout.reconfigure(encoding='utf8')
ROOT=Path(__file__).resolve().parent.parent
OUT=ROOT/'检查/Layout收尾'
WORK=ROOT/'检查/tmp/layout_handoff_work'
CLI=Path('D:/lceda-pro/lceda-pro.exe')
def command(args):
    r=subprocess.run([str(CLI),*args],capture_output=True,timeout=90)
    raw=r.stdout.decode('utf-8-sig',errors='replace').strip()
    try: obj=json.loads(raw)
    except Exception: raise RuntimeError((r.returncode,raw[:1500],r.stderr.decode('utf8',errors='replace')))
    if r.returncode or not obj.get('ok'):raise RuntimeError(obj)
    return obj
def invoke(code):
    s=json.loads((OUT/'session.json').read_text(encoding='utf8'))['sessionId']
    return command(['invoke','--session',s,'--ext-uuid','eda','--timeout','60000','--code',code])
if __name__=='__main__':
    OUT.mkdir(exist_ok=True)
    mode=sys.argv[1]
    if mode=='open':
        assert not WORK.exists(),'Do not overwrite an active copy'
        shutil.copytree(ROOT/'eda工程文件',WORK)
        hashes={}
        for p in (ROOT/'eda工程文件').rglob('*'):
            if p.is_file():
                rel=p.relative_to(ROOT/'eda工程文件')
                h=hashlib.sha256(p.read_bytes()).hexdigest()
                assert h==hashlib.sha256((WORK/rel).read_bytes()).hexdigest()
                hashes[rel.as_posix()]=h
        (OUT/'初始原生SHA256.json').write_text(json.dumps(hashes,indent=2),encoding='utf8')
        r=command(['open','--path',str(WORK/'BMS-9S4P.eprj3'),'--headless','true'])
        (OUT/'session.json').write_text(json.dumps(r['value'],indent=2),encoding='utf8')
        print(json.dumps(r,ensure_ascii=False))
    elif mode=='invoke':
        r=invoke(sys.stdin.read())
        (OUT/sys.argv[2]).write_text(json.dumps(r,ensure_ascii=False,indent=2),encoding='utf8')
        print(json.dumps(r,ensure_ascii=False))
    elif mode=='close':
        s=json.loads((OUT/'session.json').read_text(encoding='utf8'))['sessionId']
        print(json.dumps(command(['session','close','--session',s,'--destroy'])))
    elif mode=='reopen':
        r=command(['open','--path',str(WORK/'BMS-9S4P.eprj3'),'--headless','true'])
        (OUT/'session.json').write_text(json.dumps(r['value'],indent=2),encoding='utf8')
        print(json.dumps(r,ensure_ascii=False))
    else: raise ValueError(mode)
