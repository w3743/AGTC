// One-shot official API change. Exact designator/primitive preconditions, no area deletion.
const page='90aa6c46493624cf';
const opened=await eda.dmt_EditorControl.openDocument(page);
if(!String(opened).startsWith(page+'@'))throw new Error('wrong page');
const all=await eda.sch_PrimitiveComponent.getAll(undefined,false);
const byRef=new Map(all.filter(c=>c.getState_Designator()).map(c=>[c.getState_Designator(),c]));
if(byRef.size!==167||byRef.has('U7'))throw new Error('source is not expected 167-object continuation');
if(['R_OVREF_TOP','R_OVREF_BOT','R_OV_HYS'].some(r=>byRef.has(r)))throw new Error('obsolete components remain');
const expected={U3:'94115299a5a573d3',R_OVTOP:'3dcdfe8483c90fa2'};
for(const [r,id]of Object.entries(expected))if(byRef.get(r)?.getState_PrimitiveId()!==id)throw new Error('identity mismatch '+r);
const wireEdits={
  // Preserve the existing BUSOV_N port, pull-up and PNP connections; detach old output/feedback.
  b39ed3a7e5fdaf4a:[1515,1770,1515,1720,1515,1770,1450,1770,1450,1770,1450,1640,1450,1640,1445,1640,1515,1720,1500,1720,1015,1770,1015,1860,1015,1860,1030,1860,1450,1770,1015,1770],
  // The divider and C0G filter remain; old TLV1704 input branch is removed.
  '62aebd91bccc654c':[1210,1505,1190,1505,1210,1740,1210,1505,1260,1740,1210,1740,1260,1660,1260,1740,1190,1505,1190,1520,1190,1480,1190,1505],
  // Ground branch formerly ending at the deleted reference bottom resistor.
  c569e504c0b98f21:[1260,1575,1325,1575,1260,1605,1260,1575,1190,1605,1260,1605,1190,1560,1190,1605,1260,1605,1260,1620]
};
for(const id of Object.keys(wireEdits))if(!await eda.sch_PrimitiveWire.get(id))throw new Error('missing wire '+id);
const catalog={};
for(const key of ['TLV6700DDCR']){
  const rows=await eda.lib_Device.search(key,undefined,undefined,undefined,5,1);
  const exact=rows.filter(r=>r.manufacturerId===key);
  if(exact.length!==1)throw new Error('ambiguous exact library '+key);
  catalog[key]=exact[0];
}
catalog.C1608X7R1A104K080AE=await eda.lib_Device.get('4e48d71315ce4d4c9f2ca294a3b1380b','0819f05c4eef4c71ace90d822a990e87');
if(catalog.C1608X7R1A104K080AE?.name!=='CAP_0603')throw new Error('wrong existing passive carrier');
const rcl=byRef.get('U1_BUCK_RCL');
if(rcl.getState_ManufacturerId()!=='RC0603FR-071ML')throw new Error('reused 1M identity changed');
const removed=[];
for(const [id,line]of Object.entries(wireEdits)){
  const old=await eda.sch_PrimitiveWire.get(id);const net=old.getState_Net()||'GND_SYS';
  await eda.sch_PrimitiveWire.delete([id]);
  const segs=[];for(let i=0;i<line.length;i+=4)segs.push(line.slice(i,i+4));
  if(!await eda.sch_PrimitiveWire.create(segs,net))throw new Error('wire recreate failed '+id);
}
await eda.sch_PrimitiveComponent.modify(byRef.get('R_OVTOP'),{
  name:'1M/1%',manufacturer:rcl.getState_Manufacturer(),manufacturerId:rcl.getState_ManufacturerId(),
  supplier:rcl.getState_Supplier(),supplierId:rcl.getState_SupplierId(),
  otherProperty:{...rcl.getState_OtherProperty(),'Candidate Component Type':'BUS_OV_sense_divider_top',
    'Source Candidate Value / Notes':'1M 1%, 100ppm/C; BUS to VBUS_OV_SENSE; independent TLV6700 400mV reference',
    'Library Binding':'verified equivalent R0603 CAD carrier; actual procurement RC0603FR-071ML'}
});
const created=[];
for(const [ref,key,x,y]of [['U7','TLV6700DDCR',1390,1450],['C_U7','C1608X7R1A104K080AE',1550,1450]]){
  const c=await eda.sch_PrimitiveComponent.create(catalog[key],x,y,undefined,0,false,true,true);
  if(!c)throw new Error('create failed '+ref);
  const base=ref==='C_U7'?byRef.get('U3_C1'):c;
  await eda.sch_PrimitiveComponent.modify(c,{designator:ref,name:ref==='U7'?'TLV6700DDCR':'100nF/10V',manufacturerId:key,
    manufacturer:base.getState_Manufacturer(),supplier:base.getState_Supplier(),supplierId:base.getState_SupplierId(),
    otherProperty:{...base.getState_OtherProperty(),'Candidate / Manufacturer Part ID':key,'Library Binding':ref==='U7'?'exact-library':'generic-footprint-carrier',
      'Source Candidate Value / Notes':ref==='U7'?'BUS_OV independent 400mV reference; OUTB active-low high BUS':'100nF local U7 VDD bypass'}});
  const ps=await eda.sch_PrimitiveComponent.getAllPinsByPrimitiveId(c.getState_PrimitiveId());
  created.push({ref,id:c.getState_PrimitiveId(),pins:ps.map(p=>({pin:p.getState_PinNumber(),x:p.getState_X(),y:p.getState_Y(),rotation:p.getState_Rotation()}))});
}
return {stage:'components_and_exact_wire_edits',removed,created,wireEdits:Object.keys(wireEdits),requiresConnectionStage:true};
