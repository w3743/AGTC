const args=__CLI__.args;
const targets=new Map(args.contract.components.map(c=>[c.ref,c]));
const prims=(await eda.sch_PrimitiveComponent.getAll(undefined,false)).filter(c=>c.getState_Designator());
const done=[],connections=[];
const seen=new Set();
for(const c of prims){
 const ref=c.getState_Designator();if(args.only_refs&&!args.only_refs.includes(ref))continue;
 const t=targets.get(ref);if(!t)throw new Error('额外器件 '+ref);
 const targetPins=new Map(t.pins.map(p=>[String(p.number),p]));
 const pins=await eda.sch_PrimitiveComponent.getAllPinsByPrimitiveId(c.getState_PrimitiveId());
 for(const p of pins||[]){
  const want=targetPins.get(p.getState_PinNumber());if(!want)throw new Error('额外引脚 '+ref+':'+p.getState_PinNumber());
  const nc=want.connection==='NC';
  if(p.getState_NoConnected()!==nc)await p.toAsync().setState_NoConnected(nc).done();
  if(nc)continue;
  const x=Math.round(p.getState_X()*1000)/1000,y=Math.round(p.getState_Y()*1000)/1000;
  const r=((p.getState_Rotation()%360)+360)%360;
  const key=[x,y,want.net].join('|');if(seen.has(key))continue;seen.add(key);
  const ex=Math.round((x+30*Math.cos(r*Math.PI/180))*1000)/1000;
  const ey=Math.round((y+30*Math.sin(r*Math.PI/180))*1000)/1000;
  const wire=await eda.sch_PrimitiveWire.create([x,y,ex,ey],want.net);
  if(!wire)throw new Error('创建导线失败 '+ref+':'+want.number);
  const flag=await eda.sch_PrimitiveComponent.createNetPort('BI',want.net,ex,ey,r,false);
  if(!flag)throw new Error('创建网络端口失败 '+ref+':'+want.number);
  connections.push({ref,pin:want.number,net:want.net,x,y,ex,ey});
 }
 done.push(ref);
}
if(!await eda.sch_Document.save())throw new Error('保存失败');
return {status:'EXPLICIT_PIN_CONNECTIONS_SAVED',refs:done,wireAndPortPairs:connections.length,connections};
