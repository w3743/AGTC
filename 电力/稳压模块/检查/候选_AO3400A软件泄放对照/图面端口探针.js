await eda.dmt_EditorControl.openDocument('28c1f3740a9b8e02');
await new Promise(r=>setTimeout(r,1800));
const all=await eda.sch_PrimitiveComponent.getAll(undefined,false);
const ports=all.filter(c=>['DUMP_PWM_LIMIT','DUMP_PWM_ALLOWED'].includes(c.getState_Name())).map(c=>({id:c.getState_PrimitiveId(),name:c.getState_Name(),x:c.getState_X(),y:c.getState_Y(),rotation:c.getState_Rotation()}));
const wires=(await eda.sch_PrimitiveWire.getAll()).filter(w=>['DUMP_PWM_LIMIT','DUMP_PWM_ALLOWED'].includes(w.getState_Net())).map(w=>({id:w.getState_PrimitiveId(),net:w.getState_Net(),line:w.getState_Line()}));
return {ports,wires};
