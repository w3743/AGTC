const cs=await eda.sch_PrimitiveComponent.getAll(),byRef=new Map(cs.filter(c=>c.getState_ComponentType()==='part').map(c=>[c.getState_Designator(),c]));
const out=[];globalThis.wiredPumpRefs=globalThis.wiredPumpRefs||[];
for(const p of pumpContract.parts.slice(globalThis.wireStart,globalThis.wireStart+10)){
 if(wiredPumpRefs.includes(p.ref))continue;
 const c=byRef.get(p.ref);if(!c)continue;
 const pins=await eda.sch_PrimitiveComponent.getAllPinsByPrimitiveId(c.getState_PrimitiveId());
 for(const pin of pins){
  const net=p.pins[pin.pinNumber];if(net===undefined)throw Error('unknown pin '+p.ref+':'+pin.pinNumber);
  if(net===null){pin.noConnected=true;await pin.done();continue;}
  const rot=((pin.rotation%360)+360)%360;
  const dx=rot===0?35:rot===180?-35:0,dy=rot===90?35:rot===270?-35:0;
  const w=await eda.sch_PrimitiveWire.create([pin.x,pin.y,pin.x+dx,pin.y+dy],net);if(!w)throw Error('wire '+p.ref+':'+pin.pinNumber);
  const port=await eda.sch_PrimitiveComponent.createNetPort('BI',net,pin.x+dx,pin.y+dy,rot);if(!port)throw Error('port '+net);
 }
 wiredPumpRefs.push(p.ref);out.push(p.ref);
}
await eda.sch_Document.save();return out;
