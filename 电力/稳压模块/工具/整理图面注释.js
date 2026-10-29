const texts=await eda.sch_PrimitiveText.getAll();
if(texts.length&&!await eda.sch_PrimitiveText.delete(texts.map(t=>t.getState_PrimitiveId())))throw new Error('清理旧方案注释失败');
const notes=[
 [30,3710,'独立稳压模块 | 9S5P 电池 / 8S 超级电容 | 当前采用接线',20],
 [30,3650,'01  超容串联链：SC1 → SC8；8 × 350F / 2.7V；SC_MINUS 与 GND_SYS 仅经分流器相连',12],
 [30,3450,'02  功率级：BUS_EXT ↔ 主半桥 ↔ 电感 ↔ SC_FUSED；储能侧增加双向实体隔离',12],
 [30,3370,'BUS_SOURCE 为外部电池正极直接接 F_BUS 输入的边界；DRC 单端网络警告对应此处，不增设重复端子。',10],
 [30,2520,'03  辅助供电：海凌科 12V；板外专用 5V 供核心板',12],
 [880,2520,'04  DRV8161：PA8/PB13 PWM，真实故障线 DRV_FAULT_N，硬件 Break 及回读',12],
 [2460,2520,'05  TLV6700：SC / BUS 过压；BUS 硬件 dump 名义 40.32V',12],
 [2460,2130,'06  最终制动：单只板外 0.82Ω ±5%；无温熔、无电阻温度关断',12],
 [30,1820,'07  WeAct Normal QFN48 V1.0 核心板接口；PA9=TIM1_CH2 许可，PA10=故障回读',12],
 [30,1120,'08  CAN：独立本地控制；快速保护不依赖通信',12],
 [1080,1430,'09  采样：BUS / IMON 源侧滤波，TMUX1511 掉电隔离',12],
 [1080,1260,'10  单体抽头分压与慢泄放：120Ω 均衡电阻常接；不再使用 BW6101',12],
 [30,520,'11  NTC：主 MOS / 超容 / 电阻；电阻测温只作状态，不撤销最终耗能支路',12],
 [3850,3300,'12  LTC7001 + 两只 HYG022：背靠背隔离，关断两方向电流',12],
 [3850,2430,'13  开漏许可：PA9 → G07 → DRV_FAULT_N；Break 后保持关闭',12],
 [3850,1990,'14  MCU / 硬件过压通过开漏 NPN 并联请求 dump',12],
 [3850,1300,'15  BUS / IMON 掉电隔离，输出各自下拉；未用开关独立 NC',12],
 [30,-100,'连线表达：短实体导线 + 同名网络端口；同名端口电气相连。NC 单独标记，禁止把 NC 接成公共网络。',10],
 [30,-140,'PCB、固件、制造与实物验证由人类完成；电阻功率及热实现由用户负责。',10]
];
for(const [x,y,s,size]of notes)if(!await eda.sch_PrimitiveText.create(x,y,s,0,null,null,size))throw new Error('注释创建失败');
const wires=new Set((await eda.sch_PrimitiveWire.getAll()).map(w=>w.getState_PrimitiveId()));
let hidden=0;
for(const a of await eda.sch_PrimitiveAttribute.getAll()){
 if(wires.has(a.getState_ParentPrimitiveId())&&a.getState_Key()==='NET'&&a.getState_ValueVisible()){
  await eda.sch_PrimitiveAttribute.modify(a,{valueVisible:false});hidden++;
 }
}
return {status:'NOTES_AND_WIRE_LABELS_UPDATED',notes:notes.length,hiddenDuplicatedWireLabels:hidden,saved:await eda.sch_Document.save()};
