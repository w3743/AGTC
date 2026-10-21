const opened=await eda.dmt_EditorControl.openDocument('90aa6c46493624cf');
if(!String(opened).startsWith('90aa6c46493624cf@'))throw new Error('wrong page');
const result={};
for(const key of ['TLV6700DDCR','RC0603FR-071ML','C1608X7R1A104K080AE','CAP_0603']) {
  result[key]=await eda.lib_Device.search(key,undefined,undefined,undefined,5,1);
}
result.capCarrier=await eda.lib_Device.get('4e48d71315ce4d4c9f2ca294a3b1380b','0819f05c4eef4c71ace90d822a990e87');
return result;
