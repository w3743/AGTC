await eda.dmt_EditorControl.openDocument('28c1f3740a9b8e02');
await new Promise(r=>setTimeout(r,1800));
const c=(await eda.sch_PrimitiveComponent.getAll(undefined,false)).find(x=>x.getState_Designator()==='R_DUMP_IN_PD');
if(!c||c.getState_ManufacturerId()!=='RC0603FR-0747KL')throw Error('unexpected part');
if(!await eda.sch_PrimitiveComponent.modify(c,{rotation:270}))throw Error('rotation');
if(!await eda.sch_Document.save())throw Error('save');
return {saved:true,ref:'R_DUMP_IN_PD',rotation:270};
