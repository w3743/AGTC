"""Use the official editor on a private current-source copy; never edit native syntax."""
from pathlib import Path
import base64, hashlib, json, shutil, subprocess, sys
sys.stdout.reconfigure(encoding='utf8')
M=Path(__file__).resolve().parent.parent
O=M/'检查/原理图逻辑重排'
W=M/'检查/tmp/schematic_logic_work'
CLI=Path('D:/lceda-pro/lceda-pro.exe')
PAGE='350a4904c4ec87b3'
def command(args):
    r=subprocess.run([str(CLI),*args],capture_output=True,timeout=110)
    raw=r.stdout.decode('utf-8-sig',errors='replace').strip()
    try:d=json.loads(raw)
    except Exception:raise RuntimeError((r.returncode,raw[:2000],r.stderr.decode('utf8',errors='replace')[:2000]))
    if r.returncode or not d.get('ok'):raise RuntimeError(d)
    return d
def invoke(code):
    s=json.loads((O/'session.json').read_text(encoding='utf8'))['sessionId']
    return command(['invoke','--session',s,'--ext-uuid','eda','--timeout','60000','--code',code])['value']
def write(name,d):
    (O/name).write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
def download(expression,name,binary=False):
    target=O/name
    d=invoke('const f=await '+expression+';if(!f)throw Error("empty export");try{return {saved:await eda.sys_FileSystem.saveFileToFileSystem('+json.dumps(target.as_posix())+',f,undefined,true),size:f.size,name:f.name};}catch(e){return {saved:false,error:String(e)};}')
    if d.get('saved') is True and target.is_file():
        assert len(target.read_bytes())==d['size']
    else:
        code='const f=await '+expression+';if(!f)throw Error("empty export");'
        if binary:code+='const b=new Uint8Array(await f.arrayBuffer());let s="";for(let i=0;i<b.length;i+=16384)s+=String.fromCharCode(...b.subarray(i,i+16384));globalThis.LOGIC_EXPORT=btoa(s);'
        else:code+='globalThis.LOGIC_EXPORT=await f.text();'
        meta=invoke(code+'return {size:f.size,length:globalThis.LOGIC_EXPORT.length};')
        payload=''.join(invoke(f'return globalThis.LOGIC_EXPORT.slice({i},{i+48000});') for i in range(0,meta['length'],48000))
        data=base64.b64decode(payload,validate=True) if binary else payload.encode('utf8')
        if binary:assert len(data)==meta['size']
        target.write_bytes(data)
    return {'path':name,'bytes':target.stat().st_size,'sha256':hashlib.sha256(target.read_bytes()).hexdigest()}
if __name__=='__main__':
    O.mkdir(exist_ok=True)
    mode=sys.argv[1]
    if mode=='open':
        assert not W.exists(),'Do not overwrite a work copy'
        shutil.copytree(M/'eda工程文件',W)
        hashes={}
        for p in (M/'eda工程文件').rglob('*'):
            if p.is_file():
                rel=p.relative_to(M/'eda工程文件');h=hashlib.sha256(p.read_bytes()).hexdigest()
                assert h==hashlib.sha256((W/rel).read_bytes()).hexdigest();hashes[rel.as_posix()]=h
        write('正式源初始SHA256.json',hashes)
        write('正式配套初始SHA256.json',{str(p.relative_to(M)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [M/'原理图准备/BOM.csv',M/'检查/当前施工输入.json',M/'检查/BMS-9S4P.enet']})
        r=command(['open','--path',(W/'BMS-9S4P.eprj3').as_posix(),'--headless','true']);write('session.json',r['value']);print(json.dumps(r,ensure_ascii=False))
    elif mode=='invoke':
        r=invoke(sys.stdin.read());write(sys.argv[2],r);print(json.dumps(r,ensure_ascii=False))
    elif mode=='reopen':
        r=command(['open','--path',(W/'BMS-9S4P.eprj3').as_posix(),'--headless','true']);write('session.json',r['value']);print(json.dumps(r,ensure_ascii=False))
    elif mode=='close':
        s=json.loads((O/'session.json').read_text(encoding='utf8'))['sessionId']
        print(json.dumps(command(['session','close','--session',s,'--destroy']),ensure_ascii=False))
    elif mode=='snapshot':
        r=invoke('''let d;try{d=await eda.dmt_SelectControl.getCurrentDocumentInfo();}catch(e){}if(d?.uuid!=='''+json.dumps(PAGE)+''')await eda.dmt_EditorControl.openDocument('''+json.dumps(PAGE)+''');const rows=[];for(const c of (await eda.sch_PrimitiveComponent.getAll()).filter(c=>c.getState_Designator())){const ps=await eda.sch_PrimitiveComponent.getAllPinsByPrimitiveId(c.getState_PrimitiveId());rows.push({ref:c.getState_Designator(),id:c.getState_PrimitiveId(),x:c.getState_X(),y:c.getState_Y(),rotation:c.getState_Rotation(),mirror:c.getState_Mirror(),name:c.getState_Name(),mpn:c.getState_ManufacturerId(),component:c.getState_Component(),symbol:c.getState_Symbol(),footprint:c.getState_Footprint(),bom:c.getState_AddIntoBom(),pcb:c.getState_AddIntoPcb(),pins:ps.map(p=>({number:p.getState_PinNumber(),name:p.getState_PinName(),x:p.getState_X(),y:p.getState_Y(),rotation:p.getState_Rotation(),nc:p.getState_NoConnected()}))});}return {project:await eda.dmt_Project.getCurrentProjectInfo(),document:await eda.dmt_SelectControl.getCurrentDocumentInfo(),rows};''')
        write(sys.argv[2] if len(sys.argv)>2 else '几何快照.json',r);print(json.dumps({'parts':len(r['rows']),'project':r['project'],'document':r['document']},ensure_ascii=False))
    elif mode=='export':
        prefix=sys.argv[2]
        a=download("eda.sch_ManufactureData.getNetlistFile('BMS-9S4P.enet','EasyEDA')",prefix+'.enet')
        b=download("eda.sch_ManufactureData.getExportDocumentFile('BMS_current_raw.pdf','PDF',{theme:'Default',lineWidth:'Default',displayAttributesAsMenu:false,size:'Original Size'},'Current Schematic Page',{range:'All'})",prefix+'.pdf',True)
        write(prefix+'_导出.json',[a,b]);print(json.dumps([a,b],ensure_ascii=False))
    else:raise ValueError(mode)
