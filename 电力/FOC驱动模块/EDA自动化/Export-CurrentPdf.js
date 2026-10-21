await eda.dmt_EditorControl.openDocument('90aa6c46493624cf');
if((await eda.dmt_Schematic.getCurrentSchematicPageInfo())?.uuid!=='90aa6c46493624cf')throw new Error('wrong page');
const f=await eda.sch_ManufactureData.getExportDocumentFile('FOC_G431_单页原理图','PDF',{theme:'Black on White',size:'Original Size'},'Current Schematic Page');
if(!f)throw new Error('no official PDF');
const bytes=new Uint8Array(await f.arrayBuffer());let s='';
for(let i=0;i<bytes.length;i+=8192)s+=String.fromCharCode(...bytes.subarray(i,i+8192));
return {fileName:f.name,bytes:bytes.length,base64:btoa(s)};
