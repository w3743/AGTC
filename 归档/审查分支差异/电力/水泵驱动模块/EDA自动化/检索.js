await eda.dmt_EditorControl.openDocument('bafe8a48e4625018');
const keys=['C22427252','C73013','C132550','C581328','C54803369','C224052','C269443','C50670544','C115324','C269704','C269710','C5123581','C2167399'];
const out=[];for(const key of keys){try{out.push({key,device:await eda.lib_Device.getByLcscIds(key)});}catch(e){out.push({key,error:String(e)})}}
return out;
