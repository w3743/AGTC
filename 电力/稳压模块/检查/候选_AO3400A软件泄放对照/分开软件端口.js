await eda.dmt_EditorControl.openDocument('28c1f3740a9b8e02');
await new Promise(r=>setTimeout(r,1800));
const defs=[
 {net:'DUMP_PWM_LIMIT',wire:'bf5434776a86e3a1',port:'55764f8bdb4c8521',line:[1740,500,1740,560,1680,560],angle:180},
 {net:'DUMP_PWM_ALLOWED',wire:'ee4fd0cb787312b8',port:'3250d358ec5fa747',line:[1810,500,1810,440,1750,440],angle:180}
];
const all=await eda.sch_PrimitiveComponent.getAll(undefined,false), wires=await eda.sch_PrimitiveWire.getAll();
for(const d of defs){
 if(!all.some(x=>x.getState_PrimitiveId()===d.port&&x.getState_Name()===d.net)||!wires.some(x=>x.getState_PrimitiveId()===d.wire&&x.getState_Net()===d.net))throw Error('source mismatch '+d.net);
}
for(const d of defs){
 if(!await eda.sch_PrimitiveWire.delete([d.wire])||!await eda.sch_PrimitiveComponent.delete([d.port]))throw Error('remove '+d.net);
 if(!await eda.sch_PrimitiveWire.create(d.line,d.net))throw Error('wire '+d.net);
 if(!await eda.sch_PrimitiveComponent.createNetPort('BI',d.net,d.line.at(-2),d.line.at(-1),d.angle,false))throw Error('port '+d.net);
}
if(!await eda.sch_Document.save())throw Error('save');
return {status:'SAVED',nets:defs.map(x=>x.net)};
