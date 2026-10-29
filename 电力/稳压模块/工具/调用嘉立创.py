"""本机官方CLI适配器。只接管本工具新建的headless会话；不操作PCB。"""
from pathlib import Path
import argparse,csv,json,subprocess,shutil
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'检查/接线实施'
OUT.mkdir(parents=True,exist_ok=True)
CLI=Path(shutil.which('lceda-pro') or r'C:\Program Files\lceda-pro\lceda-pro.exe')
PROJECT=ROOT/'eda工程文件/sc-regulator/sc-regulator.eprj3'
STATE=OUT/'会话.json'
def call(args,timeout=60):
    p=subprocess.run([str(CLI),*args],capture_output=True,timeout=timeout)
    s=p.stdout.decode('utf-8-sig').strip()
    d=json.loads(s)
    if not d.get('ok'):raise RuntimeError(str(d))
    return d['value']
def write(path,data):path.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def start():
    existing={s['sessionId'] for s in call(['session','list'])}
    with (OUT/'打开工程返回.json').open('wb') as stdout,(OUT/'打开工程日志.txt').open('wb') as stderr:
        p=subprocess.Popen([str(CLI),'open','--path',str(PROJECT),'--headless','true'],stdout=stdout,stderr=stderr)
        try:p.wait(timeout=30)
        except subprocess.TimeoutExpired:pass
    sessions=call(['session','list'])
    fresh=[s for s in sessions if s['sessionId'] not in existing and s.get('status')=='alive' and s.get('origin')=='cli']
    if len(fresh)!=1:raise RuntimeError('不能唯一识别本次新建会话：'+str(fresh))
    state={'sessionId':fresh[0]['sessionId'],'owned':True,'project':str(PROJECT),'headless':True}
    write(STATE,state)
    return state
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['start','run','close'])
    p.add_argument('--script');p.add_argument('--output');p.add_argument('--apply',action='store_true')
    p.add_argument('--timeout',type=int,default=180000)
    p.add_argument('--refs',help='仅器件实施脚本使用，逗号分隔位号')
    a=p.parse_args()
    if not CLI.is_file():raise FileNotFoundError(CLI)
    if a.action=='start':result=start()
    else:
        state=json.loads(STATE.read_text(encoding='utf-8'))
        assert state['owned'] is True
        if a.action=='close':
            result=call(['session','close','--session',state['sessionId'],'--destroy'])
            state['status']='closed';write(STATE,state)
        else:
            assert state.get('status')!='closed','请先start创建本工具专用会话'
            assert a.script and a.output,'run必须指定--script和--output'
            data=json.loads((ROOT/'原理图准备/软件接线输入.json').read_text(encoding='utf-8'))
            data['apply']=a.apply
            if a.refs:data['only_refs']=a.refs.split(',')
            for name,key in [('实际元件库绑定.json','bindings'),('选型BOM.csv',None)]:
                if key:data[key]=json.loads((ROOT/'原理图准备'/name).read_text(encoding='utf-8'))
            data['bom']=list(csv.DictReader((ROOT/'原理图准备/选型BOM.csv').open(encoding='utf-8-sig')))
            std=ROOT/'检查/器件属性标准化计划.json'
            if std.exists():data['standardization']=json.loads(std.read_text(encoding='utf-8'))
            # 当前页切换属于软件阶段，随后独立脚本核对UUID；不切换其他用户会话。
            call(['invoke','--session',state['sessionId'],'--ext-uuid','eda','--code',
                  "await eda.dmt_EditorControl.openDocument('9546dca0bc8bce7f');return await eda.dmt_SelectControl.getCurrentDocumentInfo();"])
            # 4.1.93官方--code-file避免Windows 32767字符参数限制；内层块只为本次调用注入完整合同。
            body='{ const __CLI__ = {args:'+json.dumps(data,ensure_ascii=False)+'};\n'+Path(a.script).read_text(encoding='utf-8-sig')+'\n}'
            script_path=OUT/'本次调用体.js'
            script_path.write_text(body,encoding='utf-8')
            result=call(['invoke','--session',state['sessionId'],'--ext-uuid','eda','--timeout',str(a.timeout),
                         '--code-file',str(script_path.resolve())],a.timeout/1000+15)
            write(OUT/a.output,result)
    encoded=json.dumps(result,ensure_ascii=False)
    print(encoded if len(encoded)<3000 else json.dumps({'status':result.get('status') if isinstance(result,dict) else 'RESULT_SAVED','output':a.output,'characters':len(encoded)},ensure_ascii=False))
if __name__=='__main__':main()
