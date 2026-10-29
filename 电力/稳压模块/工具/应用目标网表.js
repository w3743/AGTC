// 软件阶段执行；输入=原理图准备/软件接线输入.json。
// 前提：已根据库绑定表放齐目标器件并移除退役件。
// 仅调用SCH/DMT API，不触及PCB。目标unique-id由实际导出反查，不用离线位号冒充内部ID。
const args = typeof __CLI__ !== 'undefined' ? __CLI__.args : undefined;
if (!args?.contract?.components) throw new Error('请将软件接线输入.json作为CLI --args传入');
const page = args.target_page_uuid || '9546dca0bc8bce7f';
await eda.dmt_EditorControl.openDocument(page);
const doc = await eda.dmt_SelectControl.getCurrentDocumentInfo();
if (doc?.uuid !== page) throw new Error('当前图页不符，未执行接线');
const file = await eda.sch_ManufactureData.getNetlistFile('before-apply.enet','EasyEDA');
if (!file) throw new Error('无法取得实际官方网表，未执行接线');
const actual = JSON.parse(await file.text());
const target = args.contract.components;
const byRef = new Map();
for (const [id,c] of Object.entries(actual.components || {})) {
  const ref=c.props?.Designator;
  if (!ref) continue;
  if (byRef.has(ref)) throw new Error('重复位号 '+ref);
  byRef.set(ref,{id,c});
}
const wanted=new Set(target.map(c=>c.ref));
const issues=[];
for (const ref of byRef.keys()) if(!wanted.has(ref)) issues.push({ref,error:'extra_component_remove_retired_first'});
for (const t of target) {
  const live=byRef.get(t.ref);
  if(!live) {issues.push({ref:t.ref,error:'missing_component'});continue;}
  if(t.manufacturer_part && live.c.props?.['Manufacturer Part']!==t.manufacturer_part)
    issues.push({ref:t.ref,error:'MPN_mismatch',actual:live.c.props?.['Manufacturer Part'],expected:t.manufacturer_part});
  if(t.supplier_part && live.c.props?.['Supplier Part']!==t.supplier_part)
    issues.push({ref:t.ref,error:'supplier_code_mismatch',actual:live.c.props?.['Supplier Part'],expected:t.supplier_part});
  const expectedPins=new Set(t.pins.map(p=>String(p.number)));
  const actualPins=Object.keys(live.c.pinInfoMap || {});
  if(actualPins.length!==expectedPins.size || actualPins.some(p=>!expectedPins.has(p)))
    issues.push({ref:t.ref,error:'library_pin_set_mismatch',actual:actualPins,expected:[...expectedPins]});
}
if(issues.length) return {status:'PRECONDITION_FAILED_NO_MUTATION',issues};
const routing={component:{}};
let changed=0,NC=0;
for(const t of target) {
  const {id,c}=byRef.get(t.ref);
  routing.component[id]={pinInfoMap:{}};
  for(const p of t.pins) {
    const n=String(p.number),net=p.connection==='NC'?'':p.net;
    if(p.connection==='NC')NC++;
    if((c.pinInfoMap[n].net||'')!==net)changed++;
    c.pinInfoMap[n].net=net;
    routing.component[id].pinInfoMap[n]={name:c.pinInfoMap[n].name,number:n,net,props:{'Pin Number':n}};
  }
}
// 默认只返回计划；软件接线阶段传apply:true执行。
if(args.apply!==true) return {status:'READY_TO_APPLY_NOT_EXECUTED',components:target.length,changedPins:changed,NC};
// 拓扑大幅变更，明确重建当前页导线和网络标识，不能依赖autoRouting隐式清理旧线。
const oldWires=await eda.sch_PrimitiveWire.getAll();
if(oldWires.length&&!await eda.sch_PrimitiveWire.delete(oldWires.map(w=>w.getState_PrimitiveId())))throw new Error('旧线清理失败');
const prims=await eda.sch_PrimitiveComponent.getAll(undefined,false);
const oldPorts=prims.filter(c=>!c.getState_Designator());
if(oldPorts.length&&!await eda.sch_PrimitiveComponent.delete(oldPorts))throw new Error('旧网络标识清理失败');
for(const c of prims.filter(c=>c.getState_Designator())){
 const t=target.find(x=>x.ref===c.getState_Designator());
 if(!t)continue;
 const pins=await eda.sch_PrimitiveComponent.getAllPinsByPrimitiveId(c.getState_PrimitiveId());
 for(const p of pins||[]){
  const shouldNC=t.pins.find(x=>String(x.number)===p.getState_PinNumber()).connection==='NC';
  if(p.getState_NoConnected()!==shouldNC)await p.toAsync().setState_NoConnected(shouldNC).done();
 }
}
await eda.sch_Netlist.setNetlist('EasyEDA',JSON.stringify(actual));
const routingResult=await eda.sch_Document.autoRouting({netlist:routing});
if(!await eda.sch_Document.save()) throw new Error('保存失败，不能认为接线完成');
const afterFile=await eda.sch_ManufactureData.getNetlistFile('after-apply.enet','EasyEDA');
if(!afterFile) throw new Error('接线后官方导出失败，未验收');
const afterText=await afterFile.text();
const after=JSON.parse(afterText);
const afterRefs=new Map(Object.values(after.components||{}).filter(c=>c.props?.Designator).map(c=>[c.props.Designator,c]));
const differences=[];
for(const t of target) for(const p of t.pins) {
  const v=afterRefs.get(t.ref)?.pinInfoMap?.[String(p.number)]?.net||'';
  if(v!==(p.net||''))differences.push({ref:t.ref,pin:p.number,actual:v,expected:p.net||''});
}
return {status:differences.length?'APPLIED_NETS_MISMATCH_REVIEW_REQUIRED':'APPLIED_PIN_MATCH_REOPEN_DRC_AND_VISUAL_REVIEW_REQUIRED',
        components:target.length,changedPins:changed,NC,routingResult,differences,officialNetlist:afterText};
