await eda.dmt_EditorControl.openDocument('28c1f3740a9b8e02');
await new Promise(r=>setTimeout(r,1800));
let all=await eda.sch_PrimitiveComponent.getAll(undefined,false);
if(!all.some(x=>x.getState_Designator()==='U_OV_EARLY'))throw Error('missing IC');
const lib='0819f05c4eef4c71ace90d822a990e87';
const parts=[
 {ref:'R_OV_EARLY_TOP',uuid:'d67a2388e10148ee92276df0b0fe46a6',name:'150kΩ',x:2820,y:1060},
 {ref:'R_OV_EARLY_MID',uuid:'b003a1f0d1e342bead8e0fa975a8f532',name:'1kΩ',x:2820,y:990},
 {ref:'R_OV_EARLY_BOT',uuid:'e78687933cd54a68b95e75fa28dfee3d',name:'100kΩ',x:2820,y:920}
];
let created=[];
for(const r of parts){
 if(all.some(x=>x.getState_Designator()===r.ref))throw Error('already exists '+r.ref);
 let z=await eda.sch_PrimitiveComponent.create({libraryUuid:lib,uuid:r.uuid},r.x,r.y,undefined,0,false,true,true);
 if(!z)throw Error('create failed '+r.ref);
 z=await eda.sch_PrimitiveComponent.modify(z,{designator:r.ref,name:r.name});
 if(!z)throw Error('modify failed '+r.ref);created.push(r.ref);
}
const result=await Promise.race([eda.sch_Document.save(),new Promise(r=>setTimeout(()=>r('SAVE_TIMEOUT'),9000))]);
return {created,save:result,status:'REVIEW_ONLY_UNWIRED'};
