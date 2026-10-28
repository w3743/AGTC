"""通过嘉立创EDA官方API逐页更新；不创建PCB。"""
import json, subprocess, pathlib, sys, base64
ROOT = pathlib.Path(__file__).resolve().parents[1]
SESSION = sys.argv[1]
CLI = r'D:\lceda-pro\lceda-pro.exe'
def invoke(code):
    p = subprocess.run([CLI, 'invoke', '--session', SESSION, '--ext-uuid', 'eda', '--timeout', '60000', '--code', code], capture_output=True)
    raw = p.stdout.decode('utf-8-sig', errors='replace').strip()
    j = json.loads(raw)
    if not j.get('ok'): raise RuntimeError(j)
    return j['value']
def load(name):
    return json.loads((ROOT/'检查'/name).read_text(encoding='utf-8-sig'))
if __name__ == '__main__':
    mapping = load('Layout器件映射.json')
    entries = mapping if isinstance(mapping, list) else mapping['components']
    for e in load('开关构建清单.json'):
        entries.append({**e,'xy':{'x':e['x'],'y':e['y']},'unresolved':[],'carrierReason':'External manual switch; no PCB footprint.'})
    originals = {e['ref']:e for e in load('原理图构建清单.json')}
    passiveRecord=ROOT/'检查'/'通用阻容落实记录.json'
    selectedPassives={e['ref'] for e in load('通用阻容落实记录.json')['parts']} if passiveRecord.exists() else set()
    for e in entries:
        e.update(x=e['xy']['x'],y=e['xy']['y'])
        original = originals.get(e['ref'],{})
        e['keepExisting']=original.get('libType')=='3'
        if original.get('forceRecreate'): e['keepExisting']=False
        if e['ref'] in ('J_BAL_SERVICE','D_CASCADE_CS','D_CASCADE_SCK','D_CASCADE_SDI','U_WDT_OPTO1','U_WDT_OPTO2'): e['keepExisting']=False
        if e['ref'] in ('U_PSU','R_RUN_PD','R_PRE_CTRL_PD','R_BMS_LOCAL_OK_PD','R_PRE_REQ_PD','R_PRE_READY_PD'): e['keepExisting']=False  # Rebuild selected exact library devices with verified footprints.
        if e['ref'] in ('J_CORE_P1','J_CORE_P2'): e['keepExisting']=True
        if e['ref'] in ('L_BAL_PWR1','R_BREG_FEED','D_BAL_TVS1','C_BAL_IN_BULK','C_BAL_OUT_BULK'): e['keepExisting']=False
        if e['ref'] in selectedPassives: e['keepExisting']=False
        if e.get('sourceFile')=='风扇元件连接.json': e['keepExisting']=False
        for k in ('value','note','subPartName','rotation','addIntoBom'):
            if k in original: e[k]=original[k]
        # 源连接变化不是器件库错误；使用当前源连接重建。
        e['unresolved']=[i for i in e['unresolved'] if i['code']!='source_native_pin_mismatch']
        if e['ref'] in ('J_BAT_HV','J_PACK_HV'):
            e['pins']['3']=None
            e['addIntoBom']=False
        if e['ref']=='RSHUNT' and 'I_BAT' in e['pins']:
            p=e['pins']; e['pins']={'1':p['I_BAT'],'2':p['I_LOAD'],'3':p['V_BAT'],'4':p['V_LOAD']}
    pages = load('原理图页面映射.json')
    (ROOT/'检查'/'当前施工输入.json').write_text(json.dumps(entries,ensure_ascii=False,indent=2),encoding='utf-8')
    wanted = sys.argv[2:] or list(pages)
    outcomes = []
    for group in wanted:
        es = [e for e in entries if e['pageGroup'] == group]
        if not es: continue
        # Windows命令行总长度有限；逐小批传入运行窗口，不截断设计备注。
        encoded=base64.b64encode(json.dumps(es,ensure_ascii=False,separators=(',',':')).encode('utf-8')).decode('ascii')
        invoke('globalThis.BMS_PAGE_DATA="";return true;')
        for start in range(0,len(encoded),24000):
            invoke('globalThis.BMS_PAGE_DATA+='+json.dumps(encoded[start:start+24000])+';return true;')
        invoke('globalThis.BMS_PAGE_ENTRIES=JSON.parse(new TextDecoder().decode(Uint8Array.from(atob(globalThis.BMS_PAGE_DATA),c=>c.charCodeAt(0))));return true;')
        code = "const entries=globalThis.BMS_PAGE_ENTRIES;\n" + r'''
const result=[];
let stage='open';
try{
const project=await eda.dmt_Project.getCurrentProjectInfo();
if(project?.uuid!=='f3b5f17f6b76f52ea94c3da976e24511a61863492a292d84aa98bcb199920378')throw Error('wrong current project');
await eda.dmt_EditorControl.openDocument(PAGE_ID);
const doc=await eda.dmt_SelectControl.getCurrentDocumentInfo();
const page=await eda.dmt_Schematic.getCurrentSchematicPageInfo();
if(doc?.documentType!==1||doc?.uuid!==PAGE_ID||page?.parentSchematicUuid!=='42dbb456fb4dd67a')throw Error('wrong schematic document');
let existing=await eda.sch_PrimitiveComponent.getAll();
const planned=[];
// 先创建与核对替换件，成功后才移除旧器件和导线。
try{
for(const e of entries){
 stage='plan '+e.ref;
 const old=existing.find(c=>c.getState_Designator()===e.ref);
 let c=old, fresh=false;
 if((e.libType==='3' && !e.unresolved.length && (!e.keepExisting||!old||(e.addIntoPcb&&!old.getState_Footprint())))||!old||e.ref==='J_BAL_SERVICE'){
  try{
   const descriptor=e.libType==='3'?await eda.lib_Device.get(e.libUuid,e.libraryUuid||'0819f05c4eef4c71ace90d822a990e87'):{libraryUuid:e.libraryUuid||'0819f05c4eef4c71ace90d822a990e87',uuid:e.libUuid,libraryType:e.libType};
   if(!descriptor)throw Error('device descriptor');
   c=await eda.sch_PrimitiveComponent.create(descriptor,e.x+14000,e.y+14000,undefined,e.rotation||0,false,e.addIntoBom!==false,e.addIntoPcb);
  }catch(err){throw Error(e.ref+' create '+e.libUuid+' '+String(err));}
  if(!c)throw Error('create '+e.ref);
  fresh=true;
 }
 if(!c) throw Error('missing existing '+e.ref);
 stage='pins '+e.ref;
 let pins;
 try{pins=await eda.sch_PrimitiveComponent.getAllPinsByPrimitiveId(c.getState_PrimitiveId());}catch(err){if(fresh)await eda.sch_PrimitiveComponent.delete(c.getState_PrimitiveId());throw err;}
 if(fresh&&e.pinRemap){for(const p of pins){if(e.pinRemap[p.pinNumber]){p.pinNumber=e.pinRemap[p.pinNumber];await p.done();}}}
 const keys=pins.map(p=>String(p.pinNumber));
 const absent=Object.keys(e.pins).filter(k=>e.pins[k]!==null&&!keys.includes(k));
 const extra=keys.filter(k=>!Object.hasOwn(e.pins,k));
 if(absent.length||extra.length){
  if(fresh)await eda.sch_PrimitiveComponent.delete(c.getState_PrimitiveId());
  throw Error(e.ref+' pin map missing='+absent+' extra='+extra);
 }
 planned.push({e,old,c,pins,fresh});
}
}catch(err){
 for(const item of planned)if(item.fresh)await eda.sch_PrimitiveComponent.delete(item.c.getState_PrimitiveId());
 throw err;
}
stage='delete ports';
for(const p of existing.filter(c=>c.getState_ComponentType()==='netport'))await eda.sch_PrimitiveComponent.delete(p.getState_PrimitiveId());
stage='delete wires';
for(const w of await eda.sch_PrimitiveWire.getAll())await eda.sch_PrimitiveWire.delete(w.getState_PrimitiveId());
const intended=new Set(entries.map(e=>e.ref));
for(const old of existing.filter(c=>c.getState_ComponentType()==='part'&&!intended.has(c.getState_Designator())))await eda.sch_PrimitiveComponent.delete(old.getState_PrimitiveId());
for(const item of planned){
 const {e,old,c,pins,fresh}=item;
 stage='modify '+e.ref;
 if(fresh&&old)await eda.sch_PrimitiveComponent.delete(old.getState_PrimitiveId());
 const properties={'Design note':e.note||'','Electrical manifest':e.ref,'Footprint carrier':e.carrierReason||'','Library status':e.unresolved.length?'UNRESOLVED':'device-associated'};
 if(e.ref.startsWith('T_BAL')&&e.unresolved.length)properties.Footprint='';
 await eda.sch_PrimitiveComponent.modify(c.getState_PrimitiveId(),{x:e.x,y:e.y,designator:e.ref,name:e.value||e.mpn||e.ref,manufacturerId:e.mpn||null,addIntoPcb:Boolean(e.addIntoPcb),addIntoBom:e.addIntoBom!==false,uniqueId:e.ref,otherProperty:properties});
 const movedPins=await eda.sch_PrimitiveComponent.getAllPinsByPrimitiveId(c.getState_PrimitiveId());
 for(const p of movedPins){
  stage='connect '+e.ref+'/'+p.pinNumber;
  const net=e.pins[String(p.pinNumber)];
  p.noConnected=net===null;await p.done();
  if(net===null)continue;
  const r=((p.rotation%360)+360)%360,d=25;
  const xx=p.x+(r===0?d:r===180?-d:0),yy=p.y+(r===90?d:r===270?-d:0);
  await eda.sch_PrimitiveWire.create([p.x,p.y,xx,yy],net);
  await eda.sch_PrimitiveComponent.createNetPort('BI',net,xx,yy,r===180?0:r===0?180:r===90?270:90);
 }
 result.push({ref:e.ref,id:c.getState_PrimitiveId(),footprint:c.getState_Footprint(),fresh,pcb:Boolean(e.addIntoPcb)});
}
const saved=await eda.sch_Document.save();
if(!saved)throw Error('schematic save failed');
return {saved,result};
}catch(err){return {error:String(err),stage,result};}
'''.replace('PAGE_ID',json.dumps(pages[group]))
        try:
            out=invoke(code)
            outcomes.append({'group':group,**out})
            if out.get('error'):
                print(group,out['stage'],ascii(out['error']),flush=True)
                break
            print(group,'saved',len(out['result']),flush=True)
        except Exception as e:
            outcomes.append({'group':group,'error':str(e)})
            print(group,'ERROR',str(e),flush=True)
            break
    (ROOT/'检查'/'原生更新证据.json').write_text(json.dumps(outcomes,ensure_ascii=False,indent=2),encoding='utf-8')
