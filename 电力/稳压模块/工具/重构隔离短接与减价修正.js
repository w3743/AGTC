const lib='0819f05c4eef4c71ace90d822a990e87';
async function open(page){
 await eda.dmt_EditorControl.openDocument(page);
 await new Promise(r=>setTimeout(r,1700));
 if((await eda.dmt_SelectControl.getCurrentDocumentInfo())?.uuid!==page)throw Error('page '+page);
}
async function connect(net,line,angle){
 if(!await eda.sch_PrimitiveWire.create(line,net))throw Error('wire '+net);
 if(!await eda.sch_PrimitiveComponent.createNetPort('BI',net,line.at(-2),line.at(-1),angle,false))throw Error('port '+net);
}
async function save(){
 const ok=await Promise.race([eda.sch_Document.save(),new Promise(r=>setTimeout(()=>r(false),10000))]);
 if(!ok)throw Error('save');
}
async function passive(ref,name,uuid){
 const old=(await eda.sch_PrimitiveComponent.getAll(undefined,false)).find(c=>c.getState_Designator()===ref);
 if(!old)throw Error(ref);
 const [x,y,a]=[old.getState_X(),old.getState_Y(),old.getState_Rotation()];
 const oldPins=(await eda.sch_PrimitiveComponent.getAllPinsByPrimitiveId(old.getState_PrimitiveId())).map(p=>[p.getState_PinNumber(),p.getState_X(),p.getState_Y()]);
 const c=await eda.sch_PrimitiveComponent.create({libraryUuid:lib,uuid},x,y,undefined,a,false,true,true);
 if(!c)throw Error('create '+ref);
 const pins=(await eda.sch_PrimitiveComponent.getAllPinsByPrimitiveId(c.getState_PrimitiveId())).map(p=>[p.getState_PinNumber(),p.getState_X(),p.getState_Y()]);
 if(pins.length!==oldPins.length||pins.some(p=>!oldPins.some(old=>JSON.stringify(old)===JSON.stringify(p))))throw Error('pin anchors '+ref);
 if(!await eda.sch_PrimitiveComponent.delete(old))throw Error('delete '+ref);
 if(!await eda.sch_PrimitiveComponent.modify(c,{designator:ref,name}))throw Error('set '+ref);
}
const result={};
await open('de55c77c5dc18e8f');
const all=await eda.sch_PrimitiveComponent.getAll(undefined,false),ws=await eda.sch_PrimitiveWire.getAll();
const wshort='aa35c8240e88b848', pshort=['d6ad2b6e1a8fc764','b95018a6a4b6f3c1'];
if(!ws.some(w=>w.getState_PrimitiveId()===wshort)||pshort.some(id=>!all.some(c=>c.getState_PrimitiveId()===id)))throw Error('short source');
if(!await eda.sch_PrimitiveWire.delete([wshort])||!await eda.sch_PrimitiveComponent.delete(pshort))throw Error('cut short');
await connect('GND_SYS',[1870,480,1870,445,1890,445,1890,420],270);
await connect('SC_ARM_LATCHED',[1900,365,1855,365,1855,395,1785,395],180);
await passive('R_LPV_FEED','RC0603FR-074K7L','7f373bdc80e0446ab3e47e7c4abe868f');

const old=all.find(c=>c.getState_Designator()==='Q_DRV_SWITCH');
const oldPins=await eda.sch_PrimitiveComponent.getAllPinsByPrimitiveId(old.getState_PrimitiveId());
const target={G:'DRV_PMOS_G',S:'12V_PROT',D:'12V_DRV'};
const has=(coords,x,y)=>{for(let i=0;i<coords.length;i+=2)if(coords[i]===x&&coords[i+1]===y)return true;return false;};
const oldWires=[],oldPorts=[];
for(const p of oldPins){
 const net=target[p.getState_PinName()],x=p.getState_X(),y=p.getState_Y();
 const hit=ws.filter(w=>w.getState_Net()===net&&has(w.getState_Line().flat(),x,y));
 if(hit.length!==1)throw Error('old switch wire '+net);
 const port=all.filter(c=>!c.getState_Designator()&&c.getState_Name()===net&&has(hit[0].getState_Line().flat(),c.getState_X(),c.getState_Y()));
 if(port.length!==1)throw Error('old switch port '+net);
 oldWires.push(hit[0].getState_PrimitiveId());oldPorts.push(port[0].getState_PrimitiveId());
}
if(!await eda.sch_PrimitiveWire.delete(oldWires)||!await eda.sch_PrimitiveComponent.delete([...oldPorts,old.getState_PrimitiveId()]))throw Error('cut old switch');
let pmos=await eda.sch_PrimitiveComponent.create({libraryUuid:lib,uuid:'cc240f919f9d4bc8bbfc845a777e68b4'},3230,340,undefined,0,false,true,true);
if(!pmos)throw Error('new switch');
pmos=await eda.sch_PrimitiveComponent.modify(pmos,{designator:'Q_DRV_SWITCH',name:'DMP3098L-7'});
if(!pmos)throw Error('new switch ref');
result.switch=[];
for(const p of await eda.sch_PrimitiveComponent.getAllPinsByPrimitiveId(pmos.getState_PrimitiveId())){
 const x=p.getState_X(),y=p.getState_Y(),a=p.getState_Rotation(),rad=a*Math.PI/180,net=target[p.getState_PinName()];
 if(!net)throw Error('new switch pin '+p.getState_PinName());
 await connect(net,[x,y,Math.round(x+85*Math.cos(rad)),Math.round(y+85*Math.sin(rad))],a);
 result.switch.push([p.getState_PinNumber(),p.getState_PinName(),net]);
}
await eda.sch_PrimitiveText.create(2000,113,'After any rail fault, wait for valid TPSI secondary rails before a fresh ARM edge',0,null,null,12);
await save();
try{result.pageRename=await eda.dmt_Schematic.modifySchematicPageName('de55c77c5dc18e8f','05 Early Guard and SC Isolation');}catch(e){result.pageRename=String(e);}
await open('28c1f3740a9b8e02');
await passive('R_DUMP_LPV_PD','RC0603FR-0722KL','58496dac16a440139f5cb6c5e234f264');
await save();
return result;