await eda.dmt_EditorControl.openDocument('28c1f3740a9b8e02');
await new Promise(r=>setTimeout(r,1800));
const doc=await eda.dmt_SelectControl.getCurrentDocumentInfo();
if(doc?.uuid!=='28c1f3740a9b8e02')throw Error('wrong page');
const drc=await eda.sch_Drc.check(true,false,true);
const enet=await eda.sch_ManufactureData.getNetlistFile(undefined,'EasyEDA');
const bom=await eda.sch_ManufactureData.getBomFile(undefined,'csv');
const svg=await eda.sch_ManufactureData.getExportDocumentFile(undefined,'SVG',undefined,'Current Schematic Page');
return {doc,drc,enet:await enet.text(),bom:await bom.text(),svg:await svg.text(),status:'ISOLATED_CANDIDATE_NOT_FOR_PRODUCTION'};
