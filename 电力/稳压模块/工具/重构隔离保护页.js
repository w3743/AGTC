const libraryUuid='0819f05c4eef4c71ace90d822a990e87';
const sheet='de55c77c5dc18e8f';
const d={
 sup:['TPS386000RGPR','abb958575acd40bdb219a35d4b0a9710'],
 ldo:['TLV76033DBZR','a391dc6c49854902b675dd9bd9b803b5'],
 diode:['1N5819HW-7-F','0059c9b5b06f4ed8867fd93a655c4730'],
 hold:['CGA6P3X7R1E226M250AB','b66aaa7771904ca09226d313eac0b6e0'],
 c100:['CGA3E2X7R1H104KT0Y0S','79c2bdde96324f8fa321b4c5c00fcd5e'],
 c1:['C3216X7R1E105K160AA','892f1d0ff001405b964c7bdb2aeca943'],
 c3:['C3225X7R1H335K250AB','adc2e66942694592a9c28650c1f97887'],
 c10:['GRM32ER71H106KA12L','a3e3bb9434764fd3b1400aa501ddb48c'],
 rt10:['RT0603BRD0710KL','3840b7e554aa441f90223c083eebd79a'],
 rt261:['RT0603BRD07261KL','755913fee6314f17b6550461a0187019'],
 rt220:['RT0603BRD07220KL','34e92f98c8ce4256835dffbf80cd8be9'],
 rt61:['RT0603BRD0761K9L','7c4f2ac7f3504b8bb89611fcde5df1bc'],
 r10:['RS-03K1002FT','76d105d271254800b23e914f247a8575'],
 r100:['RC0603FR-07100KL','7696e5ef97934e7985ae08c81ed833f2'],
 r47:['RC0603FR-0747KL','6a676f819e3646c8b3b03380076e4c20'],
 r4k7:['RC0603FR-074K7L','7f373bdc80e0446ab3e47e7c4abe868f'],
 r2k2:['RC0603FR-072K2L','ade6d2876a0a4121a46710d229a48250'],
 r1k8:['RC0603FR-071K8L','60b2681288624fbdaf3e44e5e423b210'],
 r1k:['RC0603FR-071KL','3a88329dfeba4ae2b7e97c1498192bee'],
 r7k32:['RT0603BRD077K32L','64acd838c976436688f7cb2fae4bb21a'],
 r68:['RC0603FR-0768KL','70c5368d9811439b918fd252eae677e6'],
 r1m:['RC0603FR-071ML','bc2e4cd70e5e422283434c6e2e781c5a'],
 gate3:['SN74LVC1G11DCKR','27376aeb8ef24d0493ee69f282c2787f'],
 gate:['SN74LVC1G08DCKR','a47978760be14967b7f6b45d2b518be1'],
 buf:['SN74LVC1G17DCKR','f9b1d634b1a2401c9f74aa801de63a0b'],
 od:['SN74LVC1G07DCKR','67a19913e43441b7a92c69cf6021efe7'],
 latch:['SN74LVC1G74DCTR','fed0630d099c47cba67ad27e46dc3cca'],
 iso:['TPSI3052QDWZRQ1','dc2a50eaab274c81a2c06a7c6abe2518'],
 npn:['MMBT5551','f7f17a35ca194afdbda992d103e80c1c'],
 pmos:['DMP3010LK3-13','34753b4af0424383bf91dae10dc7db2b'],
 opt:['LTV-817S-TA1-C','94b92fd4314c44a989f28fa11fba3d03'],
 minload:['25121WF4300T4E','514575261a1a4000846f6494b08e6cc0'],
 jack:['2541WV-2X01P','27efa1c393e94187ace0a64f34309c33']
};
const G='GND_SYS', P='12V_PROT', H='3V3_GUARD', C='3V3_CORE', I='ISO_MID';
const part=(ref,key,x,y,pins)=>({ref,key,x,y,pins});
const pass=(ref,key,x,y,a,b)=>part(ref,key,x,y,{1:a,2:b});
const and=(ref,x,y,a,b,out)=>part(ref,'gate',x,y,{1:a,2:b,3:G,4:out,5:H});
const b={
 minload:[pass('R_AUX_MINLOAD_TRIM','r10',760,1260,P,G)],
 ramp:[
  pass('R_SVS_RAMP','r1k',1180,1190,H,'3V3_SVS'),
  pass('C_SVS_RAMP','c1',1550,1190,'3V3_SVS',G),
  pass('R_AUX_MINLOAD','minload',760,1190,P,G)
 ],
 held:[
  part('D_GUARD_HOLD','diode',420,1090,{'A':P,'K':'12V_GUARD_HOLD'}),
  pass('C_GUARD_HOLD','hold',760,1090,'12V_GUARD_HOLD',G),
  part('U_GUARD_BIAS','ldo',1130,1100,{1:H,2:'12V_GUARD_HOLD',3:G}),
  pass('C_GUARD_LDO','c1',1540,1090,H,G),
  part('U_GUARD_SVS','sup',1130,770,{1:H,2:null,3:null,4:null,5:null,6:G,7:'UV_FORMAL5_S',8:'UV_CORE_S',9:'UV_DRV_S',10:'UV_PROT_S',11:G,12:G,13:null,14:H,15:'GUARD_RESET_N',16:'SC_RAIL_OK',17:'SC_RAIL_OK',18:'LPV_QUAL_LOCAL',19:null,20:G,21:G,EP:G,PAD:G}),
  pass('C_GUARD_SVS','c100',1580,835,H,G)
 ],
 div:[
  pass('R_UV_PROT_H','rt261',410,880,P,'UV_PROT_S'),pass('R_UV_PROT_L','rt10',410,820,'UV_PROT_S',G),
  pass('R_UV_DRV_H','rt220',410,745,'12V_DRV','UV_DRV_S'),pass('R_UV_DRV_L','rt10',410,685,'UV_DRV_S',G),
  pass('R_UV_CORE_H','rt61',410,610,C,'UV_CORE_S'),pass('R_UV_CORE_L','rt10',410,550,'UV_CORE_S',G),
  pass('R_UV_5V_H','r100',410,475,'5V_EXT','UV_FORMAL5_S'),pass('R_UV_5V_L','rt10',410,415,'UV_FORMAL5_S',G),
  pass('R_GUARD_RESET_PU','r68',1550,740,H,'GUARD_RESET_N'),pass('R_GUARD_RESET_PD','r1m',1550,680,'GUARD_RESET_N',G),
  pass('R_SC_RAIL_PU','r4k7',1550,620,H,'SC_RAIL_OK')
 ],
 iso:[
  part('U_SC_CLR','gate3',2010,1100,{1:'GUARD_OK',2:G,3:'SC_RAIL_OK',4:'SC_CLEAR_N',5:H,6:'LPV_QUAL_LOCAL'}),
  part('U_SC_ARM_FF','latch',2370,1080,{1:'PA9_TIM1_CH2',2:H,3:null,4:G,5:'SC_ARM_LATCHED',6:'SC_CLEAR_N',7:H,8:H}),
  and('U_SC_GATE_CMD',2720,1100,'SC_ARM_LATCHED','DRV_FAULT_N','SC_ISO_EN'),
  part('U_SC_ISO_GD','iso',3210,1080,{1:'SC_ISO_EN',2:'ISO_PXFR',3:H,4:G,5:I,6:'ISO_VDDM',7:'ISO_VDDH',8:'ISO_TGUP'}),
  pass('C_ISO_DIV1','c3',3580,1190,'ISO_VDDH','ISO_VDDM'),pass('C_ISO_DIV2','c10',3580,1120,'ISO_VDDM',I),
  pass('R_ISO_PXFR','r7k32',3570,1040,'ISO_PXFR',G),pass('C_ISO_PRI_BULK','c10',3570,960,H,G),
  pass('C_ISO_PRI_HF','c100',3570,885,H,G),pass('C_SC_ARM_FF','c100',2340,890,H,G),
  pass('C_SC_CLR','c100',2010,900,H,G),pass('C_SC_CMD','c100',2690,900,H,G),
  pass('R_SC_EN_PD','r47',3100,865,'SC_ISO_EN',G)
 ],
 interfaces:[
  part('U_GUARD_BUF','buf',1900,735,{1:null,2:'GUARD_RESET_N',3:G,4:'GUARD_OK',5:H}),
  pass('C_GUARD_BUF','c100',1900,570,H,G),pass('R_GUARD_OUT_PD','r47',1900,495,'GUARD_OK',G),
  pass('R_READY_BASE','r1k',2250,785,'GUARD_OK','READY_BASE'),
  part('Q_READY_LED','npn',2530,770,{1:'READY_BASE',2:G,3:'READY_LED_K'}),
  part('U_EARLY_READY_OPTO','opt',2930,770,{1:'READY_LED_A',2:'READY_LED_K',3:'B0_READY_RETURN',4:'EARLY_GUARD_READY_N'}),
  pass('R_READY_LED','r1k8',3310,760,P,'READY_LED_A'),
  part('J_EARLY_READY','jack',3560,760,{1:'EARLY_GUARD_READY_N',2:'B0_READY_RETURN'}),
  part('U_LPV_OPTO','opt',2870,560,{1:'BMS_LPV_LED_A',2:'BMS_LPV_LED_K',3:'LPV_QUAL_LOCAL',4:'LPV_OPTO_FEED'}),
  pass('R_LPV_FEED','r2k2',3230,540,C,'LPV_OPTO_FEED'),
  part('J_LPV_IN','jack',3550,540,{1:'BMS_LPV_LED_A',2:'BMS_LPV_LED_K'}),
  part('J_GUARD_12V','jack',330,260,{1:P,2:G})
 ],
 drv:[
  part('U_DRV_LPV_BUF','buf',1860,510,{1:null,2:'LPV_QUAL_LOCAL',3:G,4:'LPV_DRV_LOCAL',5:C}),
  pass('C_DRV_LPV_BUF','c100',2250,525,C,G),
  pass('R_DRV_SWITCH_BASE','r1k',2200,355,'LPV_DRV_LOCAL','DRV_SWITCH_BASE'),
  pass('R_DRV_SWITCH_PD','r47',2200,265,'DRV_SWITCH_BASE',G),
  part('Q_DRV_SWITCH_CTL','npn',2520,350,{1:'DRV_SWITCH_BASE',2:G,3:'DRV_PMOS_PULL'}),
  pass('R_DRV_PMOS_SER','r2k2',2830,350,'DRV_PMOS_PULL','DRV_PMOS_G'),
  pass('R_DRV_PMOS_PU','r10',2860,260,P,'DRV_PMOS_G'),
  part('Q_DRV_SWITCH','pmos',3230,345,{1:'DRV_PMOS_G',2:'12V_DRV',3:P,4:'12V_DRV'}),
  part('U_SC_BRIDGE_KILL','od',1920,355,{1:null,2:'SC_ARM_LATCHED',3:G,4:'DRV_FAULT_N',5:H}),
  pass('C_SC_BRIDGE_KILL','c100',1920,210,H,G)
 ]
};
async function wait(ms){await new Promise(r=>setTimeout(r,ms));}
async function bounded(name,op){return await Promise.race([op,new Promise((_,reject)=>setTimeout(()=>reject(Error(name)),6500))]);}
async function open(uuid){
 await eda.dmt_EditorControl.openDocument(uuid);
 for(let i=0;i<40;i++){await wait(150);if((await eda.dmt_SelectControl.getCurrentDocumentInfo())?.uuid===uuid){await wait(350);return;}}
 throw Error('open '+uuid);
}
async function connect(line,angle,net){
 const x=line.at(-2),y=line.at(-1);
 if(!await bounded('wire '+net,eda.sch_PrimitiveWire.create(line,net)))throw Error('wire '+net);
 if(!await bounded('port '+net,eda.sch_PrimitiveComponent.createNetPort('BI',net,x,y,angle,false)))throw Error('port '+net);
}
async function place(p){
 const [name,uuid]=d[p.key];
 let c=await bounded('create '+p.ref,eda.sch_PrimitiveComponent.create({libraryUuid,uuid},p.x,p.y,undefined,0,false,true,true));
 if(!c)throw Error('create '+p.ref);
 c=await bounded('modify '+p.ref,eda.sch_PrimitiveComponent.modify(c,{designator:p.ref,name}));
 if(!c)throw Error('modify '+p.ref);
 const pins=await eda.sch_PrimitiveComponent.getAllPinsByPrimitiveId(c.getState_PrimitiveId());
 const check=pins.map(pin=>{
  const n=String(pin.getState_PinNumber()),name=pin.getState_PinName(),net=Object.hasOwn(p.pins,n)?p.pins[n]:p.pins[name];
  if(net===undefined)throw Error('missing mapping '+p.ref+' '+n+' '+name);
  return {pin,n,name,net,x:pin.getState_X(),y:pin.getState_Y(),a:pin.getState_Rotation()};
 });
 for(const row of check){
  if(row.net===null){await bounded('NC '+p.ref+'.'+row.n,row.pin.toAsync().setState_NoConnected(true).done());continue;}
  const {x,y,a,net}=row, rad=a*Math.PI/180;
  const side=check.filter(other=>other.a===a&&other.net!==null).sort((l,r)=>a===0||a===180?l.y-r.y:l.x-r.x);
  const index=side.indexOf(row), center=(side.length-1)/2, shift=Math.round((index-center)*27);
  const tx=Math.round(x+(50+(center-Math.abs(index-center))*14)*Math.cos(rad));
  const ty=Math.round(y+(50+(center-Math.abs(index-center))*14)*Math.sin(rad));
  const dx=Math.round(65*Math.cos(rad)),dy=Math.round(65*Math.sin(rad));
  const horizontal=a===0||a===180;
  const mx=horizontal?tx:tx+shift,my=horizontal?ty+shift:ty;
  const line=shift===0?[x,y,tx+dx,ty+dy]:[x,y,tx,ty,mx,my,mx+dx,my+dy];
  await connect(line,a,net);
 }
 return {ref:p.ref,mpn:name,pins:check.map(({n,name,net})=>[n,name,net])};
}
await open(sheet);
const trace=[];
try{
 const old=new Set((await eda.sch_PrimitiveComponent.getAll(undefined,false)).map(x=>x.getState_Designator()));
 for(const stage of STAGES){
  for(const p of b[stage]){
   if(old.has(p.ref))throw Error('already exists '+p.ref);
   trace.push(await place(p));
  }
 }
 if(STAGES.includes('ramp')){
  const all=await eda.sch_PrimitiveComponent.getAll(undefined,false);
  const ws=await eda.sch_PrimitiveWire.getAll();
  const locate=(line,x,y)=>{for(let i=0;i<line.length;i+=2)if(line[i]===x&&line[i+1]===y)return true;return false;};
  for(const [ref,n] of [['U_GUARD_SVS','1'],['U_GUARD_SVS','14'],['C_GUARD_SVS','1'],['R_GUARD_RESET_PU','1']]){
   const c=all.find(c=>c.getState_Designator()===ref);
   const pin=(await eda.sch_PrimitiveComponent.getAllPinsByPrimitiveId(c.getState_PrimitiveId())).find(p=>p.getState_PinNumber()===n);
   const candidates=ws.filter(w=>w.getState_Net()===H&&locate(w.getState_Line().flat(),pin.getState_X(),pin.getState_Y()));
   if(candidates.length!==1)throw Error('unique startup branch '+ref+'.'+n+':'+candidates.length);
   const w=candidates[0],coords=w.getState_Line().flat();
   const ps=all.filter(c=>!c.getState_Designator()&&c.getState_Name()===H&&locate(coords,c.getState_X(),c.getState_Y()));
   if(ps.length!==1)throw Error('unique startup port '+ref+'.'+n+':'+ps.length);
   const port=ps[0],x=port.getState_X(),y=port.getState_Y();
   const key=(a,b)=>a+','+b, links=new Map();
   function edge(a,b){if(!links.has(a))links.set(a,[]);links.get(a).push(b);}
   const stride=coords.length%4===0?4:2;
   for(let i=0;i+3<coords.length;i+=stride){const a=key(coords[i],coords[i+1]),b=key(coords[i+2],coords[i+3]);edge(a,b);edge(b,a);}
   const start=key(pin.getState_X(),pin.getState_Y()),goal=key(x,y),queue=[[start]],seen=new Set([start]);let route;
   while(queue.length){const path=queue.shift(),last=path.at(-1);if(last===goal){route=path;break;}for(const next of links.get(last)||[]){if(seen.has(next))continue;seen.add(next);queue.push([...path,next]);}}
   if(!route)throw Error('broken startup route '+ref+'.'+n+' '+JSON.stringify(coords));
   const clean=route.flatMap(x=>x.split(',').map(Number));
   if(!await eda.sch_PrimitiveWire.delete([w.getState_PrimitiveId()]))throw Error('cut old startup wire');
   if(!await eda.sch_PrimitiveComponent.delete([port.getState_PrimitiveId()]))throw Error('cut old startup port');
   await connect(clean,port.getState_Rotation(),'3V3_SVS');
  }
  await eda.sch_PrimitiveText.create(220,1320,'HOLD / DO NOT POWER - independent early protection and SC turn-off candidate',0,null,null,21);
  await eda.sch_PrimitiveText.create(220,1275,'AUX primary in battery enclosure; all timing and backfeed limits require bench qualification',0,null,null,14);
  await eda.sch_PrimitiveText.create(1970,1275,'SC gate driver: TPSI3052, 3-wire, Kelvin source return',0,null,null,14);
 }
 const saved=await bounded('save',eda.sch_Document.save());
 return {stages:STAGES,saved,trace};
}catch(e){return {stages:STAGES,error:String(e),trace};}