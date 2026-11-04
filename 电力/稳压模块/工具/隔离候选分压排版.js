await eda.dmt_EditorControl.openDocument('28c1f3740a9b8e02');await new Promise(r=>setTimeout(r,1700));
const portIds=['ded6c73bf1b5ef5c','f1eca5fcabd3449f','b911a3432c4c8c52','654c259c388b9d97','de6e4199a1a66d0d','7e280fb8c835d150'];
const wireIds=['37ac975b526d3baa','1ece81a1a74b5cfd','6736762b3c2ed96c','bcfecafbcaa791bb','431aabfe4702e7ec','6a376d91ee555b1d'];
if(!await eda.sch_PrimitiveComponent.delete(portIds))throw Error('delete ports');
if(!await eda.sch_PrimitiveWire.delete(wireIds))throw Error('delete wires');
const specs={R_OV_EARLY_TOP:{'1':'OV_EARLY_REF','2':'OV_EARLY_H'},R_OV_EARLY_MID:{'1':'OV_EARLY_H','2':'OV_EARLY_L'},R_OV_EARLY_BOT:{'1':'OV_EARLY_L','2':'GND_SYS'}};
const all=await eda.sch_PrimitiveComponent.getAll(undefined,false);
let wired=[];
for(const [ref,map] of Object.entries(specs)){
 let c=all.find(x=>x.getState_Designator()===ref);c=await eda.sch_PrimitiveComponent.modify(c,{x:3000});
 for(const p of await eda.sch_PrimitiveComponent.getAllPinsByPrimitiveId(c.getState_PrimitiveId())){
  const num=String(p.getState_PinNumber()),net=map[num],x=Math.round(p.getState_X()),y=Math.round(p.getState_Y()),a=p.getState_Rotation()*Math.PI/180;
  const ex=Math.round(x+200*Math.cos(a)),ey=Math.round(y+200*Math.sin(a));
  if(!await eda.sch_PrimitiveWire.create([x,y,ex,ey],net))throw Error('wire '+ref+num);
  if(!await eda.sch_PrimitiveComponent.createNetPort('BI',net,ex,ey,p.getState_Rotation(),false))throw Error('port '+ref+num);
  wired.push({ref,pin:num,net});
 }
}
return {saved:await eda.sch_Document.save(),wired};
