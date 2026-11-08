// Names are placed outward from the pin; symbols retain their electrical anchor.
const cs=await eda.sch_PrimitiveComponent.getAll();let n=0;
for(const c of cs.filter(c=>c.getState_ComponentType()==='netport')){
 const x=c.getState_X(),y=c.getState_Y(),rot=c.getState_Rotation();
 const left=rot===180 || (Math.abs(x-810)<.1&&Math.abs(y-1980)<.1) || (Math.abs(x-790)<.1&&Math.abs(y-1200)<.1) || (Math.abs(x-2160)<.1&&Math.abs(y-1490)<.1);
 for(const a of await eda.sch_PrimitiveAttribute.getAll(c.getState_PrimitiveId())){
  if(a.getState_Key()==='Name'&&a.getState_ParentPrimitiveId()===c.getState_PrimitiveId()){
   await eda.sch_PrimitiveAttribute.modify(a.getState_PrimitiveId(),{x:x+(left?-48:48),y,rotation:0,alignMode:left?8:2,valueVisible:true,fontSize:null});n++;
  }
 }
}
await eda.sch_Document.save();return {ports:n};
