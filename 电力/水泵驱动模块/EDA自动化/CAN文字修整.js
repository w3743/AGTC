const cs=await eda.sch_PrimitiveComponent.getAll();let out=[];
for(const c of cs){
 if(c.getState_ComponentType()==='netport'&&c.getState_X()===2660){for(const a of await eda.sch_PrimitiveAttribute.getAll(c.getState_PrimitiveId()))if(a.getState_Key()==='Name'&&a.getState_ParentPrimitiveId()===c.getState_PrimitiveId()){await eda.sch_PrimitiveAttribute.modify(a.getState_PrimitiveId(),{x:2660,y:2042,alignMode:5});out.push('CAN_STB name');}}
 if(c.getState_Designator()==='J_CAN'){for(const a of await eda.sch_PrimitiveAttribute.getAll(c.getState_PrimitiveId()))if(['Name','Value'].includes(a.getState_Key())&&a.getState_ParentPrimitiveId()===c.getState_PrimitiveId()){await eda.sch_PrimitiveAttribute.modify(a.getState_PrimitiveId(),{x:3040,alignMode:8});out.push(a.getState_Key());}}
}
await eda.sch_Document.save();return out;
