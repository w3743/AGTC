const trace=[];
async function step(name,promise){
 const result=await Promise.race([promise,new Promise((_,reject)=>setTimeout(()=>reject(Error('timeout '+name)),5500))]);
 trace.push(name);
 return result;
}
try{
const page = '28c1f3740a9b8e02';
await eda.dmt_EditorControl.openDocument(page);
for(let i=0;i<40;i++){
  await new Promise(r=>setTimeout(r,150));
  if((await eda.dmt_SelectControl.getCurrentDocumentInfo())?.uuid===page)break;
}
if((await eda.dmt_SelectControl.getCurrentDocumentInfo())?.uuid!==page)throw Error('page binding');
const libraryUuid='0819f05c4eef4c71ace90d822a990e87';
const wires=await eda.sch_PrimitiveWire.getAll();
const old=wires.find(w=>w.getState_PrimitiveId()==='0ad1f430d5c0e9b0');
if(!old||old.getState_Net()!=='DUMP_PWM_LIMIT')throw Error('original PB7 branch');
const components=await eda.sch_PrimitiveComponent.getAll(undefined,false);
if(components.some(c=>c.getState_Designator()==='U_DUMP_SW_ALLOW'))throw Error('already applied');
const parts=[
 ['U_DUMP_SW_ALLOW','a47978760be14967b7f6b45d2b518be1','SN74LVC1G08DCKR',2960,540,{1:'DUMP_PWM_LIMIT',2:'LPV_QUAL_LOCAL',3:'GND_SYS',4:'DUMP_PWM_ALLOWED',5:'5V_EXT'}],
 ['R_DUMP_LPV_PD','76d105d271254800b23e914f247a8575','RS-03K1002FT',2780,325,{1:'LPV_QUAL_LOCAL',2:'GND_SYS'}],
 ['R_DUMP_PWM_PD','76d105d271254800b23e914f247a8575','RS-03K1002FT',3090,325,{1:'DUMP_PWM_LIMIT',2:'GND_SYS'}],
 ['C_DUMP_SW_ALLOW','79c2bdde96324f8fa321b4c5c00fcd5e','CGA3E2X7R1H104KT0Y0S',3380,325,{1:'5V_EXT',2:'GND_SYS'}]
];
async function connect(x,y,angle,net,distance=55){
 const rad=angle*Math.PI/180, ex=Math.round(x+distance*Math.cos(rad)),ey=Math.round(y+distance*Math.sin(rad));
 if(!await step('wire '+net,eda.sch_PrimitiveWire.create([x,y,ex,ey],net)))throw Error('wire '+net);
 if(!await step('port '+net,eda.sch_PrimitiveComponent.createNetPort('BI',net,ex,ey,angle,false)))throw Error('port '+net);
}
const placed=[];
for(const [ref,uuid,name,x,y,map] of parts){
 let c=await step('create '+ref,eda.sch_PrimitiveComponent.create({libraryUuid,uuid},x,y,undefined,0,false,true,true));
 if(!c)throw Error('create '+ref);
 c=await step('modify '+ref,eda.sch_PrimitiveComponent.modify(c,{designator:ref,name}));
 if(!c)throw Error('ref '+ref);
 const pins=await step('pins '+ref,eda.sch_PrimitiveComponent.getAllPinsByPrimitiveId(c.getState_PrimitiveId()));
 if(pins.length!==Object.keys(map).length||pins.some(p=>!map[p.getState_PinNumber()]))throw Error('library pins '+ref);
 for(const pin of pins)await connect(pin.getState_X(),pin.getState_Y(),pin.getState_Rotation(),map[pin.getState_PinNumber()]);
 placed.push({ref,mpn:name,pins:pins.map(p=>[p.getState_PinNumber(),p.getState_PinName?.(),map[p.getState_PinNumber()]])});
}
if(!await step('cut',eda.sch_PrimitiveWire.delete([old.getState_PrimitiveId()])))throw Error('cut original PB7 branch');
await connect(1740,500,0,'DUMP_PWM_LIMIT',15);
await connect(1810,500,180,'DUMP_PWM_ALLOWED',15);
await eda.sch_PrimitiveText.create(2720,690,'HOLD / DO NOT POWER - LPV receiver and early enable qualification pending',0,null,null,13);
await eda.sch_PrimitiveText.create(2720,650,'PB7 and locally qualified LPV gate software dump only',0,null,null,13);
const saved=await step('save',eda.sch_Document.save());
return {placed,saved,trace};
}catch(e){return {error:String(e),trace};}