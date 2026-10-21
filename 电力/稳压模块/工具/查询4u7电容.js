await eda.dmt_EditorControl.openDocument('9546dca0bc8bce7f');
const doc=await eda.dmt_SelectControl.getCurrentDocumentInfo();if(doc?.uuid!=='9546dca0bc8bce7f')throw new Error('wrong doc');
const lib='0819f05c4eef4c71ace90d822a990e87';
const keys=['GRM32ER72A475KE14L','GRM188R71E105KA12D','GCM219R71E474KA55D','4.7uF','4.7μF','4u7'];
const rows=[];for(const k of keys){const a=await eda.lib_Device.search(k,lib,undefined,undefined,80,1);rows.push({k,items:(a||[]).slice(0,30).map(x=>({uuid:x.uuid,name:x.name,manufacturer:x.property?.manufacturer,manufacturerId:x.property?.manufacturerId,supplierId:x.property?.supplierId,desc:x.description,other:x.property?.otherProperty&&{Value:x.property.otherProperty['Value'],Voltage:x.property.otherProperty['Voltage Rating'],Package:x.property.otherProperty['Supplier Footprint']}}))});}
return {rows};
