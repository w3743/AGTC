// Current schematic re-layout. Only run against the pump page, after a verified backup.
// Coordinates below are page-reading coordinates: x right, y down.
const H=2400;
const contract=globalThis.pumpLayoutContract;
if(!contract)throw Error('pumpLayoutContract missing');
const cs=(await eda.sch_PrimitiveComponent.getAll()).filter(c=>c.getState_ComponentType()==='part');
const by=new Map(cs.map(c=>[c.getState_Designator(),c]));
if(by.size!==49||!by.has('U_DRV'))throw Error('Wrong document');
const cp=new Map(contract.parts.map(p=>[p.ref,p]));
const stage=globalThis.pumpLayoutStage;
async function place(ref,x,y,rotation=0){
 const c=by.get(ref);if(!c)throw Error(ref);
 await eda.sch_PrimitiveComponent.modify(c.getState_PrimitiveId(),{x,y:H-y,rotation});
}
async function orient(ref,x,y,vertical=false){
 const c=by.get(ref),ps=await eda.sch_PrimitiveComponent.getAllPinsByPrimitiveId(c.getState_PrimitiveId());
 const a=ps.find(p=>p.pinNumber==='1'),b=ps.find(p=>p.pinNumber==='2');
 if(!a||!b||ps.length!==2)throw Error('not two-pin '+ref);
 const current=c.getState_Rotation();
 const angle=Math.atan2(b.y-a.y,b.x-a.x)*180/Math.PI;
 const target=vertical?-90:0;
 await place(ref,x,y,((current+target-angle)%360+360)%360);
}
const pins={};
async function readPins(){
 for(const c of (await eda.sch_PrimitiveComponent.getAll()).filter(c=>c.getState_ComponentType()==='part')){
  const ref=c.getState_Designator();pins[ref]={};
  for(const p of await eda.sch_PrimitiveComponent.getAllPinsByPrimitiveId(c.getState_PrimitiveId())){
   pins[ref][p.pinNumber]=[Math.round(p.x*100)/100,Math.round((H-p.y)*100)/100];
  }
 }
}
const P=(r,n)=>pins[r][String(n)];
const net=(r,n)=>cp.get(r).pins[String(n)];
let wireCount=0;
async function W(n,...pts){
 const flat=pts.flat();if(flat.length<4)throw Error('short path');
 const clean=[];
 for(let i=0;i<flat.length;i+=2){if(i&&flat[i]===flat[i-2]&&flat[i+1]===flat[i-1])continue;clean.push(flat[i],H-flat[i+1]);}
 if(clean.length<4)return;
 const w=await eda.sch_PrimitiveWire.create(clean,n);if(!w)throw Error('wire '+n+' '+JSON.stringify(pts));
 const wireId=w.getState_PrimitiveId();
 if(typeof wireId==='string'&&wireId.length>0){
  for(const a of await eda.sch_PrimitiveAttribute.getAll(wireId))if(a.getState_ParentPrimitiveId()===wireId&&a.getState_Key()==='Name')await eda.sch_PrimitiveAttribute.modify(a.getState_PrimitiveId(),{valueVisible:false});
 }
 wireCount++;return w;
}
async function label(n,x,y,align=2){
 // Standard CAD electrical symbols, anchored to the actual wire endpoint.
 if(n==='CANH'||n==='CANL'||n==='MOTOR_A'||n==='MOTOR_B')return;
 if(n==='GND_SYS'){await groundAt(x,y);return;}
 if(['BUS_M','+5V','+3V3','DRV_DVDD'].includes(n)){await power(n,x,y);return;}
 const dir=n.startsWith('ADC_')||n.toUpperCase()==='DRV_NFAULT'?'OUT':'IN';
 await eda.sch_PrimitiveComponent.createNetPort(dir,n,x,H-y,align===8?180:0,false);
}
async function tag(ref,num,len=50){
 const p=P(ref,num),c=by.get(ref),ps=await eda.sch_PrimitiveComponent.getAllPinsByPrimitiveId(c.getState_PrimitiveId());
 const pin=ps.find(p=>p.pinNumber===String(num)),a=((pin.rotation%360)+360)%360;
 const dx=a===0?len:a===180?-len:0,dy=a===90?-len:a===270?len:0;
 const q=[p[0]+dx,p[1]+dy],n=net(ref,num);await W(n,p,q);
 if(n==='GND_SYS'){
  await eda.sch_PrimitiveComponent.createNetFlag('Ground',n,q[0],H-q[1],dx<0?270:90,false);
 }else if(['BUS_M','+5V','+3V3','DRV_DVDD'].includes(n)){
  await eda.sch_PrimitiveComponent.createNetPort('BI',n,q[0],H-q[1],a,false);
 }else{
  const isMcu=ref.startsWith('J_MCU_');
  const mcuOut=['DRV_IN1','DRV_IN2','DRV_SLEEP','CAN_TX','CAN_STB'].includes(n);
  const chipOut=['CAN_RX','DRV_nFAULT','DRV_NFAULT'].includes(n);
  const dir=isMcu?(mcuOut?'OUT':'IN'):(chipOut?'OUT':'IN');
  await eda.sch_PrimitiveComponent.createNetPort(dir,n,q[0],H-q[1],a,false);
 }
}
async function groundAt(x,y){
 const g=await eda.sch_PrimitiveComponent.createNetFlag('Ground','GND_SYS',x,H-y,0,false);
 if(!g)throw Error('ground flag');
}
async function ground(ref,num,y){const p=P(ref,num);await W('GND_SYS',p,[p[0],y]);await groundAt(p[0],y);}
async function power(n,x,y){
 const g=await eda.sch_PrimitiveComponent.createNetPort('BI',n,x,H-y,0,false);
 if(!g)throw Error('power flag '+n);
}
async function T(x,y,s,size=10,bold=false){await eda.sch_PrimitiveText.create(x,H-y,s,0,null,null,size,bold,false,false,1);}
if(stage===0){
 globalThis.pumpLogicalLabels=[];
 // Delete only graphical wiring/text/net flags; retain exact part identities and attributes.
 const ws=await eda.sch_PrimitiveWire.getAll();if(ws.length)await eda.sch_PrimitiveWire.delete(ws.map(w=>w.getState_PrimitiveId()));
 const ts=await eda.sch_PrimitiveText.getAll();if(ts.length)await eda.sch_PrimitiveText.delete(ts.map(t=>t.getState_PrimitiveId()));
 const flags=(await eda.sch_PrimitiveComponent.getAll()).filter(c=>c.getState_ComponentType()!=='part');
 if(flags.length)await eda.sch_PrimitiveComponent.delete(flags.map(c=>c.getState_PrimitiveId()));
 await place('J_BUS',180,380);
 await orient('C_VM_BULK',430,420,true);await orient('D_BUS',640,420,true);
 await place('U_5V',1140,380);
 await orient('C_5V_IN',850,470,true);await orient('C_5V_IN_HF',1000,470,true);
 await orient('D_5V',1320,480,true);await orient('L_5V',1430,400);
 await orient('C_5V_OUT',1840,480,true);await orient('C_5V_OUT_HF',2010,480,true);
 await orient('R_FB_TOP',1580,470,true);await orient('R_FB_BOTTOM',1580,590,true);await orient('C_FB_FF',1710,470,true);
 await place('U_CAN',2460,400);
 await orient('R_CAN_TXD_PU',2270,310,true);await orient('R_CAN_STB_PU',2610,310,true);
 await orient('C_CAN_VCC',2300,580,true);await orient('C_CAN_VIO',2480,580,true);
 await place('D_CAN',2840,490,180);await place('J_CAN',3030,395,90);await orient('R_TERM',2740,405,true);
 await place('U_DRV',900,1200);
 await place('J_MOTOR',300,1220,180);
 await orient('C_DRV_VM1',270,980,true);await orient('C_DRV_VM2',470,980,true);
 await orient('C_DRV_VCP',850,990);await orient('C_DRV_CP',1110,1060,true);
 await orient('R_SLEEP',1070,880,true);await orient('R_IN1_PD',1370,880,true);await orient('R_IN2_PD',1670,880,true);
 await orient('R_FAULT_PU',1850,880,true);
 await orient('R_IPROPI',1150,1300,true);await orient('R_ADC_I',1380,1240);await orient('C_ADC_I',1540,1300,true);
 await orient('R_VREF_TOP',1740,1110,true);await orient('R_VREF_BOTTOM',1740,1280,true);await orient('C_VREF',1900,1280,true);
 await orient('C_DRV_DVDD',1110,1490,true);await orient('C_DRV_VCC',1430,1490,true);
 await orient('R_BUS_TOP1',2230,910);await orient('R_BUS_TOP2',2440,910);await orient('R_BUS_BOTTOM',2600,1020,true);
 await orient('R_ADC_BUS',2740,910);await orient('C_ADC_BUS',2890,1020,true);
 await orient('R_NTC_PULL',2350,1220);await orient('NTC_BOARD',2600,1320,true);await orient('C_ADC_NTC',2890,1320,true);
 await place('J_MCU_A',2340,1660);await place('J_MCU_B',2810,1660);
 await readPins();globalThis.pumpLogicalPinCoords=pins;
 return {stage,parts:by.size,pins};
}
await readPins();
if(stage===1){
 await T(100,60,'水泵驱动 / 21 V 有刷直流水泵',24,true);
 await T(100,110,'9S BUS 22.5–37.8 V  |  单向 PWM + 电制动  |  CAN 控制 / 电流、电压、板温、故障回报',12);
 await T(100,210,'01  母线入口与浪涌吸收',16,true);
 await T(810,210,'02  辅助电源：BUS → 5 V / 0.3 A',16,true);
 await T(2160,210,'03  CAN 接口与保护',16,true);
 await W('BUS_M',P('J_BUS',1),[120,P('J_BUS',1)[1]],[120,300],[710,300]);await label('BUS_M',710,300);
 await W('GND_SYS',P('J_BUS',2),[140,P('J_BUS',2)[1]],[140,560],[680,560]);await groundAt(680,560);
 for(const r of ['C_VM_BULK','D_BUS']){await W('BUS_M',[P(r,1)[0],300],P(r,1));await W('GND_SYS',P(r,2),[P(r,2)[0],560]);}
 await T(100,640,'F_BUS：线束串接 5 A / 58 VDC 保险及保险座。',10);
 await T(100,665,'BUS_M 为保险后母线；两路桥输出均不接地。',10);
 // Buck power path and local ground rail.
 await W('BUS_M',[810,420],P('U_5V',1));await label('BUS_M',810,420,8);
 for(const r of ['C_5V_IN','C_5V_IN_HF']){await W('BUS_M',[P(r,1)[0],420],P(r,1));await W('GND_SYS',P(r,2),[P(r,2)[0],650]);}
 await W('XL_SW',P('U_5V',2),[1320,400],P('L_5V',1));await W('XL_SW',[1320,400],P('D_5V',1));
 await W('+5V',P('L_5V',2),[2070,400]);await label('+5V',2070,400);
 for(const r of ['C_5V_OUT','C_5V_OUT_HF']){await W('+5V',[P(r,1)[0],400],P(r,1));await W('GND_SYS',P(r,2),[P(r,2)[0],650]);}
 await W('GND_SYS',P('D_5V',2),[1320,650]);
 await W('GND_SYS',[820,650],[2040,650]);await groundAt(2040,650);
 // Input-side ground pins, EN and exposed pad; no long return through feedback.
 for(const n of [3,6])await W('GND_SYS',P('U_5V',n),[1080,P('U_5V',n)[1]]);
 await W('GND_SYS',[1080,370],[1080,390],[1080,650]);
 await W('GND_SYS',P('U_5V',5),[1230,P('U_5V',5)[1]],[1230,310]);await groundAt(1230,310);
 await W('+5V',[1580,400],P('R_FB_TOP',1));await W('+5V',[1710,400],P('C_FB_FF',1));
 await W('XL_FB',P('R_FB_TOP',2),[1580,530],P('R_FB_BOTTOM',1));
 await W('XL_FB',P('C_FB_FF',2),[1710,530],[1580,530]);
 await W('XL_FB',P('U_5V',4),[1260,P('U_5V',4)[1]],[1260,530],[1580,530]);
 await W('GND_SYS',P('R_FB_BOTTOM',2),[1580,650]);
 await T(810,700,'开关节点 → 续流二极管 / 电感 → 输出电容；下方为反馈分压与前馈电容。',10);
}
if(stage===2){
 // CAN TX / STB default bias and a visible differential route to the connector.
 await W('CAN_TX',P('U_CAN',1),[2200,P('U_CAN',1)[1]]);await label('CAN_TX',2200,P('U_CAN',1)[1],8);
 await W('CAN_TX',P('R_CAN_TXD_PU',2),[2270,P('U_CAN',1)[1]]);
 await W('+3V3',P('R_CAN_TXD_PU',1),[2270,260]);await power('+3V3',2270,260);
 await tag('U_CAN',4,140);
 await W('CAN_STB',P('U_CAN',8),[2660,P('U_CAN',8)[1]]);await label('CAN_STB',2660,P('U_CAN',8)[1]);
 await W('CAN_STB',P('R_CAN_STB_PU',2),[2610,P('U_CAN',8)[1]]);
 await W('+3V3',P('R_CAN_STB_PU',1),[2610,260]);await power('+3V3',2610,260);
 // CANH upper rail / CANL lower rail, crossing-free across the termination and TVS.
 const h=P('U_CAN',7),l=P('U_CAN',6);
 await W('CANH',h,[2680,h[1]],[2680,370],[2940,370]);
 await W('CANL',l,[2660,l[1]],[2660,440],[2960,440]);
 await W('CANH',P('R_TERM',1),[2740,370]);await W('CANL',P('R_TERM',2),[2740,440]);
 await W('CANH',[P('D_CAN',1)[0],370],P('D_CAN',1));await W('CANL',[P('D_CAN',2)[0],440],P('D_CAN',2));
 const jh=P('J_CAN',2),jl=P('J_CAN',3),jg=P('J_CAN',1);
 await W('CANH',[2940,370],[2940,jh[1]],jh);await W('CANL',[2960,440],[2960,jl[1]],jl);
 await ground('D_CAN',3,550);await W('GND_SYS',jg,[2980,jg[1]],[2980,550]);await groundAt(2980,550);
 await label('CANH',2900,370);await label('CANL',2900,440);
 await tag('U_CAN',2,35);await tag('U_CAN',3,60);await tag('U_CAN',5,50);
 for(const [r,n] of [['C_CAN_VCC','+5V'],['C_CAN_VIO','+3V3']]){await W(n,P(r,1),[P(r,1)[0],530]);await power(n,P(r,1)[0],530);await ground(r,2,640);}
 await T(2700,615,'R_TERM 默认 DNP',10,true);await T(2700,640,'仅总线物理末端装 120 Ω',10);
 await T(2160,700,'TX / RX / STB 跨区连接到 MCU；VCC = 5 V，VIO = 3.3 V。',10);
}
if(stage===3){
 await T(100,790,'04  并桥功率驱动与本地配置',16,true);
 await T(1010,810,'控制信号默认下拉；nFAULT 上拉',11,true);
 for(const [r,n] of [['R_SLEEP','DRV_SLEEP'],['R_IN1_PD','DRV_IN1'],['R_IN2_PD','DRV_IN2']]){
  const p=P(r,1);await W(n,p,[p[0],840]);await label(n,p[0],840);await ground(r,2,930);
 }
 await W('+3V3',P('R_FAULT_PU',1),[1850,840]);await power('+3V3',1850,840);
 await W('DRV_nFAULT',P('R_FAULT_PU',2),[1850,930]);await label('DRV_nFAULT',1850,930);
 // Local VM bypass and charge-pump storage.
 await W('BUS_M',[210,920],[530,920]);await label('BUS_M',210,920,8);
 await W('GND_SYS',[270,1040],[470,1040]);await groundAt(470,1040);
 for(const r of ['C_DRV_VM1','C_DRV_VM2']){await W('BUS_M',[P(r,1)[0],920],P(r,1));await W('GND_SYS',P(r,2),[P(r,2)[0],1040]);}
 await W('DRV_VCP',P('U_DRV',1),[800,P('U_DRV',1)[1]],[800,990],P('C_DRV_VCP',1));
 await W('BUS_M',P('C_DRV_VCP',2),[920,990]);await label('BUS_M',920,990);
 await W('DRV_CPH',P('U_DRV',44),[1020,P('U_DRV',44)[1]],[1020,P('C_DRV_CP',1)[1]],P('C_DRV_CP',1));
 await W('DRV_CPL',P('U_DRV',43),[1110,P('U_DRV',43)[1]],P('C_DRV_CP',2));
 // Combined bridge output rails: all six pins per output visibly meet their rail.
 for(const n of [4,5,6,17,18,19])await W('MOTOR_A',P('U_DRV',n),[650,P('U_DRV',n)[1]]);
 await W('MOTOR_A',[650,P('U_DRV',4)[1]],[650,1360]);
 for(const n of [7,8,9,14,15,16])await W('MOTOR_B',P('U_DRV',n),[600,P('U_DRV',n)[1]]);
 await W('MOTOR_B',[600,P('U_DRV',7)[1]],[600,1390]);
 await W('MOTOR_A',P('J_MOTOR',1),[500,P('J_MOTOR',1)[1]],[500,1360],[650,1360]);
 await W('MOTOR_B',P('J_MOTOR',2),[450,P('J_MOTOR',2)[1]],[450,1390],[600,1390]);
 await label('MOTOR_A',650,1360);await label('MOTOR_B',600,1390);
 await T(100,1450,'OUT1 / OUT2 并桥：每个输出的六个引脚汇接到同一电机端。',10);
 await T(100,1475,'交叉导线无连接点时不相连；电机两端均为桥输出。',10);
 // Stock symbol keeps manufacturer pin order. Use short named boundaries for interleaved power/config pins.
 for(const n of [2,3,10,13,20,21,22,23,25,26,27,35,37,40,41,42,45])await tag('U_DRV',n,30);
 for(const [ns,n,x] of [[[11,12],'BUS_M',790],[[28,29],'DRV_DVDD',1020],[[33,34],'DRV_VREF',1070],[[30,31],'IPROPI',1100]]){
  for(const pn of ns)await W(n,P('U_DRV',pn),[x,P('U_DRV',pn)[1]]);
  await W(n,[x,P('U_DRV',ns[0])[1]],[x,P('U_DRV',ns[1])[1]]);
 }
 // DVDD mode straps and local decoupling, separate from VREF.
 await tag('U_DRV',24,40);await label('BUS_M',790,P('U_DRV',11)[1],8);await label('DRV_DVDD',1020,P('U_DRV',28)[1]);
 for(const [r,n] of [['C_DRV_DVDD','DRV_DVDD'],['C_DRV_VCC','+5V']]){const p=P(r,1);await W(n,p,[p[0],1430]);await power(n,p[0],1430);await ground(r,2,1550);}
 // IPROPI current mirror: sole 1.5k load; RC branch goes to ADC_I.
 await W('IPROPI',[1100,1240],P('R_ADC_I',1));await W('IPROPI',[1150,1240],P('R_IPROPI',1));
 await W('ADC_I',P('R_ADC_I',2),[1600,1240]);await label('ADC_I',1600,1240);
 await W('ADC_I',[1540,1240],P('C_ADC_I',1));await ground('R_IPROPI',2,1370);await ground('C_ADC_I',2,1370);
 // VREF divider and capacitor directly feed both VREF pins.
 await W('DRV_VREF',[1070,1200],[1900,1200]);
 await W('+3V3',P('R_VREF_TOP',1),[1740,1060]);await power('+3V3',1740,1060);
 await W('DRV_VREF',P('R_VREF_TOP',2),[1740,1200],P('R_VREF_BOTTOM',1));
 await W('DRV_VREF',[1900,1200],P('C_VREF',1));await ground('R_VREF_BOTTOM',2,1370);await ground('C_VREF',2,1370);
 await T(1660,1430,'VREF = 1.65 V',11,true);await T(1660,1460,'IPROPI 1.5k → 名义斩波约 5.19 A',10);
 await T(1010,1600,'MODE1 / MODE2 = DVDD；RSVD 32 / 36 / 38 / 39 悬空。',10);
}
if(stage===4){
 await T(2160,790,'05  母线电压与板温采样',16,true);
 await W('BUS_M',[2160,910],P('R_BUS_TOP1',1));await label('BUS_M',2160,910,8);
 await W('BUS_DIV_MID',P('R_BUS_TOP1',2),P('R_BUS_TOP2',1));
 await W('BUS_SENSE',P('R_BUS_TOP2',2),[2600,910],P('R_ADC_BUS',1));
 await W('BUS_SENSE',[2600,910],P('R_BUS_BOTTOM',1));
 await W('ADC_BUS',P('R_ADC_BUS',2),[2980,910]);await label('ADC_BUS',2980,910);
 await W('ADC_BUS',[2890,910],P('C_ADC_BUS',1));await ground('R_BUS_BOTTOM',2,1100);await ground('C_ADC_BUS',2,1100);
 await T(2160,1140,'两只 110k 串联 / 10k 分压；1k + 100nF ADC 滤波。',10);
 await W('+3V3',[2250,1220],P('R_NTC_PULL',1));await power('+3V3',2250,1220);
 await W('ADC_NTC',P('R_NTC_PULL',2),[2980,1220]);await label('ADC_NTC',2980,1220);
 await W('ADC_NTC',[2600,1220],P('NTC_BOARD',1));await W('ADC_NTC',[2890,1220],P('C_ADC_NTC',1));
 await ground('NTC_BOARD',2,1400);await ground('C_ADC_NTC',2,1400);
 await T(2160,1440,'NTC 10k / B3950；与 10k 上拉构成分压，10nF 就地滤波。',10);
 await T(2160,1490,'06  WeAct STM32G431CBU6 核心板底座',16,true);
 // Core-board pins remain numbered exactly as the external coreboard P1 / P2.
 for(const r of ['J_MCU_A','J_MCU_B']){
  const n=r==='J_MCU_A'?'+5V':'+3V3';
  const a=P(r,1),b=P(r,2),g=P(r,3),h=P(r,4),row=a[1]-35;
  await W(n,a,[a[0]-25,a[1]],[a[0]-25,row],[b[0]+25,row],[b[0]+25,b[1]],b);await power(n,(a[0]+b[0])/2,row);
  // The two ground pins use separate boundary labels to avoid crossing the live signal fan-out.
  await tag(r,3,35);await tag(r,4,35);
  for(const [num,nm] of Object.entries(cp.get(r).pins))if(nm&&!['1','2','3','4'].includes(num))await tag(r,num,55);
 }
 await T(2160,1760,'J_MCU_A ↔ 核心板 P1；J_MCU_B ↔ 核心板 P2，同号连接。',10);
 await T(2160,1790,'PA8/TIM1_CH1 → IN2；PB5 → IN1；PB7 → nSLEEP。',10);
 await T(2160,1820,'PC10 → CAN_STB；PC11 ← nFAULT；PB8/PB9 = CAN_RX/TX。',10);
 await T(2160,1850,'PA0 = ADC_I；PA1 = ADC_BUS；PA2 = ADC_NTC。',10);
 await T(2160,1880,'PB8 / BOOT0：nSWBOOT0=0、nBOOT0=1，烧录后读回验证。',10);
 await T(100,1690,'控制、保护与装配约束',16,true);
 await T(100,1740,'RUN：nSLEEP=1，IN1=1，IN2=反相 PWM（20 kHz）；D = u × min(0.95, 21 / VBUS)。',11);
 await T(100,1775,'BRAKE：nSLEEP=1、IN1=IN2=1；COAST：IN1=IN2=0；复位默认休眠。',11);
 await T(100,1810,'CAN 失联 / 过温 / 故障在本地停机，故障恢复后重新使能；IPROPI 运行斩波不等于制动限流。',11);
 await T(100,1845,'EP45 接地铜及散热过孔。3V3 来自核心板；USB 与母线并用须处理 5V 回灌。',11);
 await T(100,1880,'SMCJ40A 标称脉冲钳位 64.5V：低于驱动 70V 绝对最大，高于 60V 推荐工作上限。',11);
 await T(100,1940,'设计与验证范围',14,true);
 await T(100,1980,'49 个板内器件；核心板、保险及线端配件列入整套 BOM。无阀门驱动。',11);
 await T(100,2015,'本图用于原理图交付；PCB 封装极性 / 高度及核心板机械接口按移交约束核对，样机阶段验证热性能。',11);
 await T(100,2075,'跨功能区：使用同名网络连接；功能区内：用直接连线表达电流路径、反馈和滤波关系。',11,true);
 await T(2160,2075,'水泵驱动模块 | 2026-10-19 | 逻辑接线',11);
}
// Deterministic finish: every netport Name is visible at the API default font size.
const netports=(await eda.sch_PrimitiveComponent.getAll('netport')).filter(p=>typeof p.getState_PrimitiveId()==='string'&&p.getState_PrimitiveId().length>0).sort((a,b)=>{
 const aId=a.getState_PrimitiveId(),bId=b.getState_PrimitiveId();return aId<bId?-1:aId>bId?1:0;
});
for(const p of netports){
 const portId=p.getState_PrimitiveId();
 for(const a of await eda.sch_PrimitiveAttribute.getAll(portId))if(a.getState_ParentPrimitiveId()===portId&&a.getState_Key()==='Name')await eda.sch_PrimitiveAttribute.modify(a.getState_PrimitiveId(),{valueVisible:true,fontSize:null});
}
await eda.sch_Document.save();
return {stage,wireCount,parts:by.size};
