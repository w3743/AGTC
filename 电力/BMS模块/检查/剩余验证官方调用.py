"""Official CLI on a hash-checked isolated native copy; never auto-publish."""
from pathlib import Path
import hashlib, json, shutil, subprocess, sys
sys.stdout.reconfigure(encoding='utf8')

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / '检查/剩余验证'
WORK = ROOT / '检查/tmp/remaining_validation_work'
CLI = Path(r'D:\lceda-pro\lceda-pro.exe')

def command(args):
    p = subprocess.run([str(CLI), *args], capture_output=True, timeout=70)
    raw = p.stdout.decode('utf-8-sig', errors='replace').strip()
    try: result = json.loads(raw)
    except Exception: raise RuntimeError((p.returncode, raw[:1000], p.stderr.decode('utf8', errors='replace')))
    if p.returncode or not result.get('ok'): raise RuntimeError(result)
    return result

def invoke(code):
    session = json.loads((OUT/'session.json').read_text(encoding='utf8'))['sessionId']
    return command(['invoke','--session',session,'--ext-uuid','eda','--timeout','60000','--code',code])

if __name__ == '__main__':
    OUT.mkdir(exist_ok=True)
    mode = sys.argv[1]
    if mode == 'open':
        assert not WORK.exists(), 'Do not overwrite an active work copy'
        shutil.copytree(ROOT/'eda工程文件', WORK)
        hashes = {}
        for p in (ROOT/'eda工程文件').rglob('*'):
            if p.is_file():
                rel = p.relative_to(ROOT/'eda工程文件')
                a = hashlib.sha256(p.read_bytes()).hexdigest()
                assert a == hashlib.sha256((WORK/rel).read_bytes()).hexdigest()
                hashes[rel.as_posix()] = a
        (OUT/'初始原生SHA256.json').write_text(json.dumps(hashes,indent=2),encoding='utf8')
        result = command(['open','--path',str(WORK/'BMS-9S4P.eprj3'),'--headless','true'])
        (OUT/'open.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8')
        (OUT/'session.json').write_text(json.dumps(result['value'],ensure_ascii=False,indent=2),encoding='utf8')
        print(json.dumps(result,ensure_ascii=False))
    elif mode == 'invoke':
        result = invoke(sys.stdin.read())
        dest = OUT / sys.argv[2]
        dest.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8')
        print(json.dumps({'saved':str(dest),'value':result['value']},ensure_ascii=False))
    elif mode == 'reopen':
        assert (WORK/'BMS-9S4P.eprj3').is_file()
        result = command(['open','--path',str(WORK/'BMS-9S4P.eprj3'),'--headless','true'])
        (OUT/'session.json').write_text(json.dumps(result['value'],ensure_ascii=False,indent=2),encoding='utf8')
        print(json.dumps(result,ensure_ascii=False))
    elif mode == 'close':
        session=json.loads((OUT/'session.json').read_text(encoding='utf8'))['sessionId']
        print(json.dumps(command(['session','close','--session',session,'--destroy']),ensure_ascii=False))
