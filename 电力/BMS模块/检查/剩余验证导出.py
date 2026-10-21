"""Export official API results from the isolated current work session."""
from pathlib import Path
import base64, hashlib, importlib.util, json, sys

P=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('api',P/'剩余验证官方调用.py')
api=importlib.util.module_from_spec(spec);spec.loader.exec_module(api)
O=api.OUT
def call(code):return api.invoke(code)['value']
def write(name,obj):
    (O/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf8')
def download(expression,name,binary=False):
    # Write only into the isolated export directory, then independently verify size.
    # Native publication is handled separately after archive/hash checks.
    target=O/name
    direct=call('const f=await '+expression+';if(!f)throw Error("no export");try{return {saved:await eda.sys_FileSystem.saveFileToFileSystem('+json.dumps(str(target))+',f,undefined,true),size:f.size,name:f.name};}catch(e){return {saved:false,error:String(e)};}')
    if direct.get('saved') is True and target.is_file():
        data=target.read_bytes();assert len(data)==direct['size']
        return {**direct,'path':name,'sha256':hashlib.sha256(data).hexdigest()}
    code='const f=await '+expression+';if(!f)throw Error("no export");'
    if binary:
        code+='const b=new Uint8Array(await f.arrayBuffer());let s="";for(let i=0;i<b.length;i+=16384)s+=String.fromCharCode(...b.subarray(i,i+16384));globalThis.REMAINING_EXPORT=btoa(s);'
    else:code+='globalThis.REMAINING_EXPORT=await f.text();'
    meta=call(code+'return {name:f.name,size:f.size,length:globalThis.REMAINING_EXPORT.length};')
    payload=''.join(call(f'return globalThis.REMAINING_EXPORT.slice({i},{i+48000});') for i in range(0,meta['length'],48000))
    data=base64.b64decode(payload,validate=True) if binary else payload.encode('utf8')
    if binary:assert len(data)==meta['size']
    (O/name).write_bytes(data)
    return {**meta,'path':name,'sha256':hashlib.sha256(data).hexdigest()}

if __name__=='__main__':
    mode=sys.argv[1]
    call("await eda.dmt_EditorControl.openDocument('350a4904c4ec87b3');return true;")
    if mode=='snapshot':
        r=call('''const rows=[];for(const c of (await eda.sch_PrimitiveComponent.getAll()).filter(c=>c.getState_Designator())){const ps=await eda.sch_PrimitiveComponent.getAllPinsByPrimitiveId(c.getState_PrimitiveId());rows.push({ref:c.getState_Designator(),id:c.getState_PrimitiveId(),mpn:c.getState_ManufacturerId(),name:c.getState_Name(),component:c.getState_Component(),symbol:c.getState_Symbol(),footprint:c.getState_Footprint(),x:c.getState_X(),y:c.getState_Y(),pcb:c.getState_AddIntoPcb(),bom:c.getState_AddIntoBom(),pins:ps.map(p=>({number:String(p.pinNumber),noConnected:p.getState_NoConnected()}))});}return {rows,drc:await eda.sch_Drc.check(true,false,true)};''')
        write('保存重开器件核对.json',r)
        result=download("eda.sch_ManufactureData.getNetlistFile('BMS-9S4P.enet','EasyEDA')",'final_reopened.enet')
        print(json.dumps({'parts':len(r['rows']),'drc':r['drc'],'netlist':result},ensure_ascii=False))
    elif mode=='deliverables':
        result=[]
        result.append(download("eda.sch_ManufactureData.getExportDocumentFile('BMS_current_raw.pdf','PDF',{theme:'Default',lineWidth:'Default',displayAttributesAsMenu:false,size:'Original Size'},'Current Schematic Page',{range:'All'})",'BMS_current_raw.pdf',True))
        project=call('return (await eda.dmt_Project.getCurrentProjectInfo()).uuid;')
        result.append(download('eda.sys_FileManager.getProjectFileByProjectUuid('+json.dumps(project)+',"BMS-9S4P",undefined,"epro2")','BMS-9S4P.epro2',True))
        write('官方导出SHA256.json',result)
        print(json.dumps(result,ensure_ascii=False))
    else:raise ValueError(mode)
