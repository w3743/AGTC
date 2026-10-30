// 嘉立创 4.1.93 四页官方只读验收；期望页与跨页网来自连接合同。
const cfg=__CLI__?.args?.contract?.drawing;
if(!cfg||!cfg.bookUuid||!Array.isArray(cfg.pages)||cfg.pages.length!==4||!Array.isArray(cfg.crossPageNets))throw new Error('MISSING_FOUR_PAGE_DRAWING_CONTRACT');
const wait=ms=>new Promise(r=>setTimeout(r,ms));
const safe=(o,m)=>{try{return o&&typeof o[m]==='function'?o[m]():undefined}catch(e){return null}};
const sortedRefs=refs=>refs.map(x=>x==null?'':String(x).trim()).filter(Boolean).sort();
const sameRefs=(a,b)=>a.length===b.length&&a.every((x,i)=>x===b[i]);
const result={pages:[],project:null,schematic:null,network:null,errors:[]};
for(const p of cfg.pages) await eda.dmt_EditorControl.openDocument(p.uuid);
for(const p of cfg.pages){
 if(!Array.isArray(p.refs)||new Set(sortedRefs(p.refs)).size!==p.refs.length)throw new Error(`INVALID_PAGE_REFERENCE_CONTRACT:${p.uuid}`);
 const expectedRefs=sortedRefs(p.refs);
 await eda.dmt_EditorControl.openDocument(p.uuid);
 let comps=[],doc=null,pi=null,actualRefs=[],ready=false;
 for(let i=0;i<30;i++){
  await wait(500); doc=await eda.dmt_SelectControl.getCurrentDocumentInfo();
  pi=await eda.dmt_Schematic.getCurrentSchematicPageInfo();
  if(doc?.uuid===p.uuid&&pi?.uuid===p.uuid&&pi?.parentSchematicUuid===cfg.bookUuid){
   comps=await eda.sch_PrimitiveComponent.getAll(undefined,false)||[];
   actualRefs=sortedRefs(comps.map(c=>safe(c,'getState_Designator')));
   if(new Set(actualRefs).size===actualRefs.length&&sameRefs(actualRefs,expectedRefs)){ready=true;break;}
  }
 }
 if(!ready)throw new Error(`PAGE_CONTEXT_OR_REFERENCE_SET_MISMATCH:${p.uuid};doc=${doc?.uuid};page=${pi?.uuid};book=${pi?.parentSchematicUuid};actual=${JSON.stringify(actualRefs)};expected=${JSON.stringify(expectedRefs)}`);
 const row={uuid:p.uuid,title:p.title,doc,pageInfo:pi,components:[],marks:[],wires:[],drc:null};
 for(const c of comps){
  const ref=String(safe(c,'getState_Designator')||'').trim();if(!ref){row.marks.push({id:safe(c,'getState_PrimitiveId'),type:safe(c,'getState_ComponentType'),net:safe(c,'getState_Net')});continue;}
  const id=safe(c,'getState_PrimitiveId'),device=safe(c,'getState_Component');
  const pins=await eda.sch_PrimitiveComponent.getAllPinsByPrimitiveId(id)||[];
  row.components.push({ref,id,device,pins:pins.map(x=>({number:String(safe(x,'getState_PinNumber')||''),name:String(safe(x,'getState_PinName')||''),nc:safe(x,'getState_NoConnected')}))});
 }
 try{
  const wanted=new Map(row.marks.map(m=>[m.id,m]));
  for(const a of await eda.sch_PrimitiveAttribute.getAll()||[]){
   const parent=safe(a,'getState_ParentPrimitiveId'),mark=wanted.get(parent),key=safe(a,'getState_Key');
   if(mark&&['Name','Global Net Name'].includes(key)){mark.attributes??={};mark.attributes[key]=safe(a,'getState_Value');}
  }
 }catch(e){row.markAttributesError=String(e)}
 try{const wires=await eda.sch_PrimitiveWire.getAll();row.wires=(wires||[]).map(w=>({id:safe(w,'getState_PrimitiveId'),net:safe(w,'getState_Net')}));}catch(e){row.wiresError=String(e)}
 try{let f=await eda.sch_ManufactureData.getNetlistFile(undefined,'EasyEDA');row.enet=await f.text();}catch(e){row.enetError=String(e)}
 try{let f=await eda.sch_ManufactureData.getBomFile(undefined,'csv');row.bom=f?await f.text():null;}catch(e){row.bomError=String(e)}
 try{let f=await eda.sch_ManufactureData.getExportDocumentFile(p.title+'.svg','SVG',undefined,'Current Schematic Page');row.svg=await f.text();}catch(e){row.svgError=String(e)}
 try{row.drc=await eda.sch_Drc.check(true,false,true);}catch(e){row.drcError=String(e)}
 result.pages.push(row);
}
result.project=await eda.dmt_Project.getCurrentProjectInfo();
result.schematic=await eda.dmt_Schematic.getCurrentSchematicInfo();
result.networkApi={status:'UNSUPPORTED_CLIENT_4_1_93',note:'getCurrentProjectAllNets is documented for v4.2 and later; single official full-book ENET is retained instead.'};
return result;
