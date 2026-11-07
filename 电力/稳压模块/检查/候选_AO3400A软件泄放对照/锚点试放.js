await eda.dmt_EditorControl.openDocument('28c1f3740a9b8e02');
await new Promise(r=>setTimeout(r,1800));
if((await eda.dmt_SelectControl.getCurrentDocumentInfo())?.uuid!=='28c1f3740a9b8e02')throw Error('wrong page');
const defs=[
 ['Q_DUMP_SW','b68850111fa243d3b18c9ddb8fac8962'],
 ['R_DUMP_SW_BASE','5e03c61d36774a939987ce73b1a35063'],
 ['R_DUMP_IN_PD','6a676f819e3646c8b3b03380076e4c20']
];
const all=await eda.sch_PrimitiveComponent.getAll(undefined,false), result=[];
for(const [ref,uuid] of defs){
 const old=all.find(c=>c.getState_Designator()===ref);if(!old)throw Error(ref);
 const placed=await eda.sch_PrimitiveComponent.create({libraryUuid:'0819f05c4eef4c71ace90d822a990e87',uuid},old.getState_X(),old.getState_Y(),undefined,old.getState_Rotation(),old.getState_Mirror(),true,true);
 if(!placed)throw Error('create '+ref);
 async function pins(c){return (await eda.sch_PrimitiveComponent.getAllPinsByPrimitiveId(c.getState_PrimitiveId())).map(p=>({n:p.getState_PinNumber(),name:p.getState_PinName(),x:p.getState_X(),y:p.getState_Y()})).sort((a,b)=>String(a.n).localeCompare(String(b.n)))}
 result.push({ref,old:await pins(old),proposed:await pins(placed)});
}
return {saved:false,result};
