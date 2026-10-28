const page='90aa6c46493624cf';
await eda.dmt_EditorControl.openDocument(page);
const cs=await eda.sch_PrimitiveComponent.getAll(undefined,false);
const refs=new Map(cs.filter(c=>c.getState_Designator()).map(c=>[c.getState_Designator(),c]));
if(refs.size!==169||refs.get('U7')?.getState_PrimitiveId()!=='ecce9bf834f12403'||refs.get('C_U7')?.getState_PrimitiveId()!=='84609f30528ce224')throw new Error('wrong connection-stage source');
const pinMap={};for(const ref of ['U3','U7','C_U7'])pinMap[ref]=new Map((await eda.sch_PrimitiveComponent.getAllPinsByPrimitiveId(refs.get(ref).getState_PrimitiveId())).map(p=>[String(p.getState_PinNumber()),p]));
const targets=[
  {net:'VBUS_OV_SENSE',line:[[1435,1440,1465,1440],[1465,1440,1465,1390]],port:[1465,1390,90]},
  {net:'DUMP_5V',line:[[1435,1450,1530,1450],[1500,1450,1500,1430]],port:[1500,1430,90]},
  {net:'BUSOV_N',line:[[1435,1460,1495,1460],[1495,1460,1495,1520]],port:[1495,1520,270]},
  {net:'GND_SYS',line:[[1345,1440,1315,1440],[1345,1450,1315,1450],[1315,1440,1315,1490]],port:[1315,1490,270]},
  {net:'GND_SYS',line:[[1570,1450,1590,1450],[1590,1450,1590,1490]],port:[1590,1490,270]},
  {net:'GND_SYS',line:[[770,1740,810,1740]],port:[810,1740,180]},
  {net:'DUMP_5V',line:[[770,1750,850,1750]],port:[850,1750,180]}
];
// Repeat calls stop before adding duplicate ports.
for(const t of targets)if(cs.some(c=>!c.getState_Designator()&&c.getState_Net()===t.net&&c.getState_X()===t.port[0]&&c.getState_Y()===t.port[1]))throw new Error('connection stage already applied');
for(const t of targets){
  if(!await eda.sch_PrimitiveWire.create(t.line,t.net))throw new Error('wire create failed '+t.net);
  if(!await eda.sch_PrimitiveComponent.createNetPort('BI',t.net,t.port[0],t.port[1],t.port[2],false))throw new Error('port create failed '+t.net);
}
for(const [ref,pin]of [['U3','13'],['U7','1']])await pinMap[ref].get(pin).setState_NoConnected(true).done();
const saved=await eda.sch_Document.save();
return {stage:'connected_and_saved',saved,components:refs.size,expectedPins:506,NC:['U3.13','U7.1'],connections:targets};
