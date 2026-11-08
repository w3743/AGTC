await eda.dmt_EditorControl.openDocument('28c1f3740a9b8e02');
await new Promise(r=>setTimeout(r,1800));
if((await eda.dmt_SelectControl.getCurrentDocumentInfo())?.uuid!=='28c1f3740a9b8e02')throw Error('wrong page');
const defs=[
 {ref:'Q_DUMP_SW',uuid:'b68850111fa243d3b18c9ddb8fac8962',name:'AO3400A'},
 {ref:'R_DUMP_SW_BASE',uuid:'5e03c61d36774a939987ce73b1a35063',name:'100Ω'},
 {ref:'R_DUMP_IN_PD',uuid:'6a676f819e3646c8b3b03380076e4c20',name:'47kΩ',rotation:90}
];
const all=await eda.sch_PrimitiveComponent.getAll(undefined,false), result=[];
async function pins(c){return (await eda.sch_PrimitiveComponent.getAllPinsByPrimitiveId(c.getState_PrimitiveId())).map(p=>({n:String(p.getState_PinNumber()),name:p.getState_PinName(),x:p.getState_X(),y:p.getState_Y()})).sort((a,b)=>a.n.localeCompare(b.n))}
for(const d of defs){
 const old=all.find(c=>c.getState_Designator()===d.ref);if(!old)throw Error(d.ref);
 const oldPins=await pins(old);
 let replacement=await eda.sch_PrimitiveComponent.create({libraryUuid:'0819f05c4eef4c71ace90d822a990e87',uuid:d.uuid},old.getState_X(),old.getState_Y(),undefined,d.rotation??old.getState_Rotation(),old.getState_Mirror(),true,true);
 if(!replacement)throw Error('create '+d.ref);
 const newPins=await pins(replacement);
 if(JSON.stringify(oldPins.map(({n,x,y})=>({n,x,y})))!==JSON.stringify(newPins.map(({n,x,y})=>({n,x,y}))))throw Error('pin anchors differ '+d.ref+JSON.stringify({oldPins,newPins}));
 if(!await eda.sch_PrimitiveComponent.delete(old))throw Error('delete '+d.ref);
 replacement=await eda.sch_PrimitiveComponent.modify(replacement,{designator:d.ref,name:d.name});
 if(!replacement)throw Error('rename '+d.ref);
 result.push({ref:d.ref,oldPins,newPins,manufacturer:replacement.getState_ManufacturerId()});
}
if(!await eda.sch_PrimitiveText.create(1550,610,'HOLD - AO3400A comparison; gate drive and turn-off not qualified',0,null,null,12))throw Error('text');
if(!await eda.sch_Document.save())throw Error('save');
return {status:'SAVED_ISOLATED_HOLD',result};
