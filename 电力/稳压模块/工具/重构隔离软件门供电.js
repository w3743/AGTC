await eda.dmt_EditorControl.openDocument('28c1f3740a9b8e02');
await new Promise(r=>setTimeout(r,1700));
const all=await eda.sch_PrimitiveComponent.getAll(undefined,false),ws=await eda.sch_PrimitiveWire.getAll();
const rows=[
 ['d1ed140105f66eb1','d7d89abe931d593d','DUMP_PWM_LIMIT',[2940,560,2900,560,2900,620,2780,620]],
 ['89f3aa0104666c2d','ccd426344ee5b967','LPV_QUAL_LOCAL',[2940,550,2860,550,2860,585,2720,585]],
 ['f7c33ad0c62cfb40','4737fde2479a2947','GND_SYS',[2940,540,2750,540]],
 ['2354e2ecb05ecabc','e142a4b4ae67ed77','DUMP_PWM_ALLOWED',[2940,530,2860,530,2860,495,2720,495]],
 ['cc54bee4092c374d','b6da32fefe23782d','3V3_CORE',[2940,520,2900,520,2900,460,2780,460]],
 ['804983c9247a07b3','1839699e936c260a','3V3_CORE',[3360,325,3305,325]]
];
if(rows.some(([w,c])=>!ws.some(p=>p.getState_PrimitiveId()===w)||!all.some(p=>p.getState_PrimitiveId()===c)))throw Error('branch IDs differ');
if(!await eda.sch_PrimitiveWire.delete(rows.map(r=>r[0])))throw Error('old wires');
if(!await eda.sch_PrimitiveComponent.delete(rows.map(r=>r[1])))throw Error('old ports');
for(const [, , net, line] of rows){
 if(!await eda.sch_PrimitiveWire.create(line,net))throw Error('wire '+net);
 if(!await eda.sch_PrimitiveComponent.createNetPort('BI',net,line.at(-2),line.at(-1),180,false))throw Error('port '+net);
}
const saved=await Promise.race([eda.sch_Document.save(),new Promise(r=>setTimeout(()=>r('SAVE_TIMEOUT'),8000))]);
return {saved,nets:rows.map(r=>r[2])};