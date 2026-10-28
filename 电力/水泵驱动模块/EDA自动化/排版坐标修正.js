const wires=await eda.sch_PrimitiveWire.getAll();if(wires.length)await eda.sch_PrimitiveWire.delete(wires);
const texts=await eda.sch_PrimitiveText.getAll();if(texts.length)await eda.sch_PrimitiveText.delete(texts);
const cs=await eda.sch_PrimitiveComponent.getAll();const nonParts=cs.filter(c=>c.getState_ComponentType()!=='part');
await eda.sch_PrimitiveComponent.delete(nonParts);
for(const c of cs.filter(c=>c.getState_ComponentType()==='part')){const r=c.getState_Designator();await eda.sch_PrimitiveComponent.modify(c.getState_PrimitiveId(),{y:2500-c.getState_Y(),...(['C_5V_IN','C_5V_OUT'].includes(r)?{rotation:0}:{})});}
globalThis.wiredPumpRefs=[];globalThis.pumpNotesCreated=false;await eda.sch_Document.save();return {wires:wires.length,texts:texts.length,deleted:nonParts.length};
