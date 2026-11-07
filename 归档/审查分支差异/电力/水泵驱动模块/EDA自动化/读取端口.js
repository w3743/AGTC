const cs=await eda.sch_PrimitiveComponent.getAll();const out=[];
for(const c of cs.filter(c=>c.getState_ComponentType()==='netport')){
 const attrs=await eda.sch_PrimitiveAttribute.getAll(c.getState_PrimitiveId());
 out.push({id:c.getState_PrimitiveId(),x:c.getState_X(),y:c.getState_Y(),rot:c.getState_Rotation(),attrs:attrs.filter(a=>a.getState_ParentPrimitiveId()===c.getState_PrimitiveId()).map(a=>({id:a.getState_PrimitiveId(),key:a.getState_Key(),value:a.getState_Value(),x:a.getState_X(),y:a.getState_Y()}))});
}return out;
