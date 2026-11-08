await eda.dmt_EditorControl.openDocument('28c1f3740a9b8e02');await new Promise(r=>setTimeout(r,1500));
const comp=(await eda.sch_PrimitiveComponent.getAll()).filter(c=>c.getState_X()>2700).map(c=>({id:c.getState_PrimitiveId(),ref:c.getState_Designator(),x:c.getState_X(),y:c.getState_Y(),type:c.getState_ComponentType(),net:c.getState_Net?.()}));
const wires=(await eda.sch_PrimitiveWire.getAll()).filter(w=>{const v=w.getState_Line().flat();return v[0]>2700}).map(w=>({id:w.getState_PrimitiveId(),net:w.getState_Net(),line:w.getState_Line()}));
return {comp,wires};
