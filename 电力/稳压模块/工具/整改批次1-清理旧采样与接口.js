throw new Error('已作废：曾误删均衡导线，禁止执行；使用待审精确ID方案并先复核')
const page='9546dca0bc8bce7f';
const doc=await eda.dmt_EditorControl.openDocument(page);
if((await eda.dmt_SelectControl.getCurrentDocumentInfo())?.uuid!==page) throw new Error('Wrong schematic');
const all=await eda.sch_PrimitiveComponent.getAll();
const comps=all.filter(c=>c.getState_Designator());
const ref=c=>c.getState_Designator();
const id=c=>c.getState_PrimitiveId();
const obsoleteRefs=new Set();
for(const c of comps){
 const r=ref(c);
 if(/^R_T[1-7]_(HI|LO|TEST)$/.test(r)||/^C_T[1-7]_FILTER$/.test(r)||/^R_TEST_(HI|LO)$/.test(r)||r==='C_TEST_FB') obsoleteRefs.add(r);
}
const ports=all.filter(c=>!c.getState_Designator()).map(c=>({c,id:id(c),net:c.getState_Net(),x:c.getState_X(),y:c.getState_Y(),type:c.getState_ComponentType()}));
const wires=await eda.sch_PrimitiveWire.getAll();
const wireRows=wires.map(w=>({w,id:w.getState_PrimitiveId(),net:w.getState_Net(),line:w.getState_Line()}));
const inOldTop=w=>{
 const line=w.line;
 if(!Array.isArray(line)||line.length<4||line.length%2) return false;
 const xs=[],ys=[];
 for(let i=0;i<line.length;i+=2){xs.push(Number(line[i]));ys.push(Number(line[i+1]));}
 return Math.min(...xs)>=1000 && Math.max(...xs)<=3500 && Math.min(...ys)>=0 && Math.max(...ys)<=1400;
};
const topWireIds=wireRows.filter(r=>inOldTop({line:r.line})).map(r=>r.id);
// Remove old top labels on the retired ADC/TEST networks and old tap-divider stubs.
const deletePorts=ports.filter(p=>{
 if(p.type==='netflag' && p.net==='GND_SYS' && p.x>=1000 && p.x<=3500 && p.y<=1400) return true;
 if(/^AD_T[1-8]$/.test(p.net||'')) return true;
 if((p.net==='TEST'||p.net==='TEST_FB') && p.y<1400) return true;
 if((p.net==='TEST'||p.net==='TEST_FB') && p.x<1000 && p.y>=1500 && p.y<=1700) return true;
 if(p.net==='NTC_MAIN' && p.x===870 && p.y===1625) return true;
 if(/^T[1-7]$/.test(p.net||'') && p.x>=1000 && p.x<3500 && p.y<1100) return true;
 if((p.net==='SC_PLUS'||p.net==='SC_MINUS'||p.net==='BUS_ADC') && p.x>=1000 && p.x<3500 && p.y<1400) return true;
 return false;
});
// Exact old connector stubs for removed AD_T2..AD_T7 and P1 TEST_FB: leave all other header nets intact.
const connectorIds={J_CORE_P1:'37153759bf08ab84',J_CORE_P2:'49c2d54e5601d4b7'};
const removedConnectorPorts=ports.filter(p=>((/^AD_T[1-7]$/.test(p.net||''))||(p.net==='TEST_FB'))&&p.x<1000&&p.y>=1500&&p.y<=1700&&!(p.net==='AD_T1'&&p.x===870&&p.y===1555));
const headerPins=[];
for(const [r,pid] of Object.entries(connectorIds)){
 const c=comps.find(z=>ref(z)===r); if(!c) throw new Error('Missing '+r);
 const ps=await eda.sch_PrimitiveComponent.getAllPinsByPrimitiveId(id(c));
 headerPins.push(...ps.map(p=>({ref:r,pin:p.getState_PinNumber(),x:p.getState_X(),y:p.getState_Y(),obj:p})));
}
const removeHeaderWireIds=[];
for(const p of removedConnectorPorts){
 const pin=headerPins.find(q=>q.y===p.y && Math.abs(q.x-p.x)<=100);
 if(pin){
  const w=wireRows.find(r=>{
   const a=r.line; let hasP=false,hasPin=false;
   for(let i=0;i<a.length;i+=2){if(Number(a[i])===p.x&&Number(a[i+1])===p.y)hasP=true;if(Number(a[i])===pin.x&&Number(a[i+1])===pin.y)hasPin=true;}
   return hasP&&hasPin;
  });
  if(w) removeHeaderWireIds.push(w.id);
 }
}
const removeWireIds=[...new Set([...topWireIds,...removeHeaderWireIds])];
if(removeWireIds.length){const ok=await eda.sch_PrimitiveWire.delete(removeWireIds);if(!ok)throw new Error('wire delete failed');}
const oldComponents=comps.filter(c=>obsoleteRefs.has(ref(c))).map(id);
if(oldComponents.length){const ok=await eda.sch_PrimitiveComponent.delete(oldComponents);if(!ok)throw new Error('component delete failed');}
const removePortIds=deletePorts.map(p=>p.id);
if(removePortIds.length){const ok=await eda.sch_PrimitiveComponent.delete(removePortIds);if(!ok)throw new Error('port delete failed');}
// Retarget the remaining physical core-header signals to the verified local pin map.
const p1=headerPins.filter(p=>p.ref==='J_CORE_P1');
const p2=headerPins.filter(p=>p.ref==='J_CORE_P2');
const pin=(arr,n)=>arr.find(p=>p.pin===String(n));
for(const n of [13,14,15,16,17,18]){const p=pin(p2,n);if(p){p.obj.setState_NoConnected(true);await p.obj.done();}}
const p1test=pin(p1,8); if(p1test){p1test.obj.setState_NoConnected(true);await p1test.obj.done();}
// Existing short stubs already reach these physical pads; replace any old ports at their former ends.
const replacementPorts=ports.filter(p=>p.net==='TEST'||p.net==='TEST_FB'||p.net==='AD_T8'||p.net==='AD_T1'||p.net==='NTC_MAIN');
const connectorReplacements=[];
const repl=[
 {ref:'J_CORE_P1',pin:7,net:'SC_SUM',x:130,y:1615},
 {ref:'J_CORE_P2',pin:5,net:'BQ_I2C_SCL',x:650,y:1625},
 {ref:'J_CORE_P2',pin:6,net:'BQ_I2C_SDA',x:870,y:1625},
 {ref:'J_CORE_P2',pin:20,net:'NTC_MAIN',x:870,y:1555}
];
for(const spec of repl){
 // The old ports on the same connector stubs were deleted above; recreate at verified port endpoints.
 const p=await eda.sch_PrimitiveComponent.createNetPort('BI',spec.net,spec.x,spec.y,0,false);
 if(!p)throw new Error('createNetPort failed '+spec.net);
 connectorReplacements.push({net:spec.net,x:spec.x,y:spec.y});
}
return {deletedComponents:[...obsoleteRefs].sort(),deletedWireCount:removeWireIds.length,deletedPortCount:removePortIds.length,deletedWireIds:removeWireIds,noConnect:[...Array(6)].map((_,i)=>'P2-'+(13+i)).concat(['P1-8']),createdConnectorPorts:connectorReplacements,opened:doc};
