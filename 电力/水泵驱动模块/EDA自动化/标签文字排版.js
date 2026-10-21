// Wire endpoint order is normalized by EasyEDA; locate the pin rather than assuming order.
const cs=await eda.sch_PrimitiveComponent.getAll();
const ports=cs.filter(c=>c.getState_ComponentType()!=='part');
if(ports.length)await eda.sch_PrimitiveComponent.delete(ports.map(c=>c.getState_PrimitiveId()));
const pins=new Set();
for(const c of cs.filter(c=>c.getState_ComponentType()==='part')){
 for(const p of await eda.sch_PrimitiveComponent.getAllPinsByPrimitiveId(c.getState_PrimitiveId()))pins.add(`${p.x},${p.y}`);
}
let n=0;
for(const w of await eda.sch_PrimitiveWire.getAll()){
 const [xa,ya,xb,yb]=w.getState_Line();
 const aIsPin=pins.has(`${xa},${ya}`),bIsPin=pins.has(`${xb},${yb}`);
 if(aIsPin===bIsPin)throw Error('Ambiguous wire endpoints: '+w.getState_PrimitiveId());
 const [px,py,x,y]=aIsPin?[xa,ya,xb,yb]:[xb,yb,xa,ya];
 for(const a of await eda.sch_PrimitiveAttribute.getAll(w.getState_PrimitiveId())){
  if(a.getState_Key()!=='Name')continue;
  await eda.sch_PrimitiveAttribute.modify(a.getState_PrimitiveId(),{
   x:x+(x<px?-4:4),y,rotation:0,alignMode:x<px?8:2,fontSize:null,valueVisible:true});n++;
 }
}
await eda.sch_Document.save();return {wireLabels:n};
