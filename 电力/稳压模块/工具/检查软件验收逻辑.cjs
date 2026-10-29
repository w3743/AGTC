// 离线验证验收器能识别错网、漏脚及NC误连接；不模拟电路或冒充EDA运行。
const fs=require('fs');const path=require('path');
const root=path.resolve(__dirname,'..');
const input=JSON.parse(fs.readFileSync(path.join(root,'原理图准备/软件接线输入.json'),'utf8'));
const base=JSON.parse(fs.readFileSync(path.join(root,'原理图准备/目标连接网表.enet'),'utf8'));
const AsyncFunction=Object.getPrototypeOf(async function(){}).constructor;
const script=fs.readFileSync(path.join(__dirname,'软件后逐针验收.js'),'utf8');
const run=new AsyncFunction('eda','__CLI__',script);
function mock(netlist){
 const prims=input.contract.components.map(c=>({
  getState_Designator:()=>c.ref,getState_PrimitiveId:()=>c.ref,getState_Name:()=>c.manufacturer_part,
  getState_ManufacturerId:()=>c.manufacturer_part,getState_SupplierId:()=>c.supplier_part,
  getState_Component:()=>({uuid:'mock-'+c.ref,libraryUuid:'mock'})}));
 // 网络端口不应被当成没有位号的普通器件导致假失败。
 prims.push({getState_Designator:()=>'',getState_PrimitiveId:()=> 'net-port'});
 return {
  dmt_Project:{getCurrentProjectInfo:async()=>({uuid:'project'})},
  dmt_SelectControl:{getCurrentDocumentInfo:async()=>({uuid:input.target_page_uuid,documentType:1,parentProjectUuid:'project'})},
  dmt_Schematic:{getCurrentSchematicPageInfo:async()=>({uuid:input.target_page_uuid}),getAllSchematicPagesInfo:async()=>[{uuid:input.target_page_uuid}]},
  sch_PrimitiveComponent:{getAll:async()=>prims,getAllPinsByPrimitiveId:async ref=>input.contract.components.find(c=>c.ref===ref).pins.map(p=>({getState_PinNumber:()=>p.number,getState_PinName:()=>p.name,getState_NoConnected:()=>p.connection==='NC'}))},
  sch_ManufactureData:{getNetlistFile:async()=>({name:'mock.enet',type:'application/json',size:1,text:async()=>JSON.stringify(netlist)})}
 };
}
(async()=>{
 const good=await run(mock(base),{args:input});
 if(good.status!=='PASS_COMPONENT_SET_PIN_NUMBERS_AND_PIN_NETS')throw new Error('正确503针样例未通过:'+JSON.stringify(good));
 const nPins=input.contract.components.reduce((n,c)=>n+c.pins.length,0);
 const nNC=input.contract.components.reduce((n,c)=>n+c.pins.filter(p=>p.connection==='NC').length,0);
 const checks=[{name:`correct_${nPins}_pins_and_${nNC}_independent_NC`,pass:true}];
 for(const [name,mutate] of [
  ['wrong_5V_polarity',x=>x.components.PSU5_EXT.pinInfoMap['1'].net='BUS_EXT'],
  ['missing_LTC_EP',x=>delete x.components.U_ISO_DRV.pinInfoMap['11']],
  ['accidental_NC_short',x=>x.components.U_GATE_OD.pinInfoMap['1'].net='GND_SYS']
 ]){
  const wrong=structuredClone(base);mutate(wrong);
  const result=await run(mock(wrong),{args:input});
  if(result.status==='PASS_COMPONENT_SET_PIN_NUMBERS_AND_PIN_NETS')throw new Error(name+' 被错误放行');
  checks.push({name,pass:true,result:result.status});
 }
 fs.writeFileSync(path.join(root,'检查/软件验收器离线核对.json'),JSON.stringify({status:'OFFLINE_LOGIC_CHECKS_PASS_NOT_EDA_EXECUTION',checks},null,2)+'\n');
 console.log(JSON.stringify(checks));
})().catch(e=>{console.error(e);process.exitCode=1});
