const args=__CLI__.args;
const doc=await eda.dmt_SelectControl.getCurrentDocumentInfo();
if(doc?.uuid!==args.target_page_uuid)throw new Error('目标页不符');
const wanted=new Map(args.contract.components.map(c=>[c.ref,c]));
const bindings=new Map(args.bindings.components.map(c=>[c.ref,c]));
const bom=new Map(args.bom.map(c=>[c.Reference,c]));
const all=await eda.sch_PrimitiveComponent.getAll(undefined,false);
const actual=new Map(all.filter(c=>c.getState_Designator()).map(c=>[c.getState_Designator(),c]));
const positions={
 R_DUMP_EXT:[3470,1990,0],PSU5_EXT:[650,2120,0],
 U_CELL_MUX1:[1450,1080,0],U_CELL_MUX2:[2650,480,0],U_SIGNAL_ISO:[4000,1000,0],
 U_GATE_OD:[4000,2200,0],U_ISO_DRV:[4000,2850,0],
 Q_SC_ISO1:[4400,3050,0],Q_SC_ISO2:[4700,3050,0],
 D_SC_LO:[2200,2680,0],D_SC_HI:[2200,3050,0],D_DUMP_FW:[3370,2200,0],
 Q_DUMP_SW:[4000,1800,0],C_ISO_BST:[4240,2750,270],
 R_ADC_A_PD:[1750,1080,270],R_ADC_B_PD:[2950,480,270],
 R_ARM_PD:[4200,2120,270],R_ISO_GS:[4550,2940,270],R_ISO_TGUP:[4250,2910,0],
 R_DUMP_HW_PU:[4300,1840,270],R_DUMP_SW_BASE:[4200,1740,0],
 R_BUS_OV_TOP1:[2800,2120,0],R_BUS_OV_TOP2:[2950,2120,0],R_BUS_OV_BOT:[3100,2120,270]
};
const removed=[],placed=[],updated=[],pinIssues=[];
// 所有现行采购属性均以采用BOM覆盖，不继承载体采购/功率信息。
for(const [ref,c]of actual)if(!wanted.has(ref)){
 if(!await eda.sch_PrimitiveComponent.delete(c))throw new Error('删除失败 '+ref);
 removed.push(ref);
}
for(const [ref,t]of wanted){
 if(args.only_refs&&!args.only_refs.includes(ref))continue;
 const b=bom.get(ref),bind=bindings.get(ref);let c=actual.get(ref);
 const old=c;
 const exactIdentity=c&&(c.getState_ManufacturerId()||'')===(b['Manufacturer part']||'')&&
  (!b['Supplier part']||c.getState_SupplierId()===b['Supplier part']);
 if(!c || !exactIdentity){
  const p=old?[old.getState_X(),old.getState_Y(),old.getState_Rotation()]:positions[ref];
  if(!p)throw new Error('缺少坐标 '+ref);
  if(old&&!await eda.sch_PrimitiveComponent.delete(old))throw new Error('替换删除失败 '+ref);
  c=await eda.sch_PrimitiveComponent.create({libraryUuid:bind.library_uuid,uuid:bind.device_uuid},p[0],p[1],undefined,p[2],old?.getState_Mirror()||false,true,b['Convert to PCB']==='Yes');
  if(!c)throw new Error('放置失败 '+ref);
  placed.push(ref);
 }
 c=await eda.sch_PrimitiveComponent.modify(c,{designator:ref,name:b['Value or candidate'],
  manufacturer:b.Manufacturer||null,manufacturerId:b['Manufacturer part']||null,
  supplier:b.Supplier||null,supplierId:b['Supplier part']||null,
  addIntoBom:true,addIntoPcb:b['Convert to PCB']==='Yes',
  otherProperty:{'Selected Footprint':b.Footprint,'Scope':b.Scope,'Design Note':b.Note||'',
   'Manufacturer Part':b['Manufacturer part']||'','Supplier Part':b['Supplier part']||''}});
 if(!c)throw new Error('属性修改失败 '+ref);
 updated.push(ref);
 const pins=await eda.sch_PrimitiveComponent.getAllPinsByPrimitiveId(c.getState_PrimitiveId());
 const pnums=(pins||[]).map(p=>String(p.getState_PinNumber())).sort();
 const expected=t.pins.map(p=>String(p.number)).sort();
 if(JSON.stringify(pnums)!==JSON.stringify(expected))pinIssues.push({ref,actual:pnums,expected,
   pins:(pins||[]).map(p=>({number:p.getState_PinNumber(),name:p.getState_PinName()}))});
}
if(!await eda.sch_Document.save())throw new Error('保存失败');
return {status:pinIssues.length?'PLACED_PIN_SET_ISSUES':'PARTS_PLACED_PIN_SETS_MATCH',removed,placed,updatedCount:updated.length,pinIssues};
