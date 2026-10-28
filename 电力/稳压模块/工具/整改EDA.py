"""本模块官方CLI整改适配器；不操作PCB，保存每次API返回证据。"""
from pathlib import Path
import argparse, hashlib, json, subprocess, zipfile

MODULE = Path(__file__).resolve().parents[1]
ROOT = MODULE.parent
CLI = r'D:\lceda-pro\lceda-pro.exe'
PROJECT = MODULE / 'eda工程文件/sc-regulator/sc-regulator.eprj3'
PAGE = '9546dca0bc8bce7f'
OUT = MODULE / '检查/原理整改'
OUT.mkdir(parents=True, exist_ok=True)
STATE = OUT / 'session.json'

def cli(*args, timeout=80):
    p = subprocess.run([CLI, *args], capture_output=True, timeout=timeout)
    d = json.loads(p.stdout.decode('utf-8-sig'))
    if not d.get('ok'):
        raise RuntimeError(str(d))
    return d['value']

def save(name, d):
    (OUT / name).write_text(json.dumps(d, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')

def invoke(sid, code, timeout=60000):
    return cli('invoke','--session',sid,'--ext-uuid','eda','--timeout',str(timeout),'--code',code)

def start():
    before = {s['sessionId'] for s in cli('session','list')}
    # Electron子进程可能长期持有继承的PIPE；open使用文件接收，避免run的
    # 超时清理再次communicate时无限等待。只接管本次新建会话。
    with (OUT/'open-response.json').open('wb') as out, (OUT/'open-stderr.log').open('wb') as err:
        process=subprocess.Popen([CLI,'open','--path',PROJECT.as_posix(),'--headless','true'],stdout=out,stderr=err)
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            pass
    try:
        envelope=json.loads((OUT/'open-response.json').read_text(encoding='utf-8-sig'))
        assert envelope['ok']
        sid=envelope['value']['sessionId']
    except (ValueError,KeyError,AssertionError):
        new = [s for s in cli('session','list') if s['sessionId'] not in before and s.get('status')=='alive']
        if len(new)!=1:
            raise RuntimeError('启动超时且不能唯一识别新建会话；拒绝选择其他编辑器。')
        sid = new[0]['sessionId']
    save('session.json', {'session':sid,'project':PROJECT.as_posix(),'owned':True})
    result = invoke(sid, "const p=await eda.dmt_Project.getCurrentProjectInfo();const o=await eda.dmt_EditorControl.openDocument('"+PAGE+"');const d=await eda.dmt_SelectControl.getCurrentDocumentInfo();return {project:p?.name,uuid:p?.uuid,opened:o,document:d};")
    save('opened.json',result)
    assert result['document']['uuid']==PAGE
    return {'session':sid,'opened':True}

def baseline():
    target=OUT/'修改前源.zip'
    if target.exists():
        return {'baseline':'already_present'}
    paths=[PROJECT, MODULE/'原理图准备/BOM.csv',MODULE/'原理图准备/design-plan.json']
    paths+=list((MODULE/'eda工程文件/sc-regulator/sch').rglob('*'))
    rows=[]
    with zipfile.ZipFile(target,'x',zipfile.ZIP_DEFLATED) as z:
        for p in paths:
            if p.is_file():
                raw=p.read_bytes(); name=p.relative_to(ROOT).as_posix()
                z.writestr(name,raw);rows.append({'path':name,'sha256':hashlib.sha256(raw).hexdigest(),'bytes':len(raw)})
    with zipfile.ZipFile(target) as z:
        assert z.testzip() is None
        assert all(hashlib.sha256(z.read(r['path'])).hexdigest()==r['sha256'] for r in rows)
    save('修改前SHA.json',rows)
    return {'backup':target.name,'files':len(rows),'roundtrip':'PASS'}

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['start','baseline','run','close'])
    p.add_argument('script',nargs='?');p.add_argument('output',nargs='?')
    a=p.parse_args()
    if a.action=='start': result=start()
    elif a.action=='baseline': result=baseline()
    else:
        state=json.loads(STATE.read_text(encoding='utf-8'));sid=state['session']
        if a.action=='close': result=cli('session','close','--session',sid,'--destroy')
        else:
            if not a.script or not a.output: p.error('run需要脚本路径和证据文件名')
            result=invoke(sid,Path(a.script).read_text(encoding='utf-8-sig'))
            save(a.output,result)
    print(json.dumps(result,ensure_ascii=True)[:4500])
