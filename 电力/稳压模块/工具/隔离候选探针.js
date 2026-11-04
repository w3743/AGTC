await eda.dmt_EditorControl.openDocument('28c1f3740a9b8e02');
await new Promise(r=>setTimeout(r,1500));
const all=await eda.sch_PrimitiveComponent.getAll(undefined,false);
const take=['R_DRV_GAIN','R_ARM_PD','R_PB5_PERMIT','R_QP_COL'];
let models=[];
for(const ref of take){let c=all.find(x=>x.getState_Designator()===ref);models.push({ref,device:c?.getState_Component(),x:c?.getState_X(),y:c?.getState_Y()});}
let c=await eda.sch_PrimitiveComponent.create({libraryUuid:'0819f05c4eef4c71ace90d822a990e87',uuid:'bb4ad8735a9a4337a96861b2dd8db439'},3000,800,undefined,0,false,true,true);
if(!c)throw Error('candidate device create failed');
let pins=await eda.sch_PrimitiveComponent.getAllPinsByPrimitiveId(c.getState_PrimitiveId());
await eda.dmt_EditorControl.openDocument('28c1f3740a9b8e03');await new Promise(r=>setTimeout(r,1000));
let c150=(await eda.sch_PrimitiveComponent.getAll(undefined,false)).find(x=>x.getState_Designator()==='R_BUS_HI');
models.push({ref:'R_BUS_HI',device:c150?.getState_Component()});
return {models,pins:pins.map(p=>({number:p.getState_PinNumber(),name:p.getState_PinName(),x:p.getState_X(),y:p.getState_Y(),rotation:p.getState_Rotation(),nc:p.getState_NoConnected()}))};
