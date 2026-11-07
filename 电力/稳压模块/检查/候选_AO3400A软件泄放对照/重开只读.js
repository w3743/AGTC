const ids=['9546dca0bc8bce7f','28c1f3740a9b8e02','28c1f3740a9b8e03','28c1f3740a9b8e04','de55c77c5dc18e8f'];
const wait=ms=>new Promise(r=>setTimeout(r,ms));
const pages=[];
for(const id of ids){
 await eda.dmt_EditorControl.openDocument(id);
 let doc;for(let i=0;i<40;i++){await wait(200);doc=await eda.dmt_SelectControl.getCurrentDocumentInfo();if(doc?.uuid===id)break;}
 if(doc?.uuid!==id)throw Error('page '+id);
 const drc=await eda.sch_Drc.check(true,false,true);
 let svg=null;
 if(id==='28c1f3740a9b8e02'){const f=await eda.sch_ManufactureData.getExportDocumentFile(undefined,'SVG',undefined,'Current Schematic Page');if(!f)throw Error('no SVG');svg=await f.text();}
 pages.push({id,drc,svg});
}
const net=await eda.sch_ManufactureData.getNetlistFile(undefined,'EasyEDA');
const bom=await eda.sch_ManufactureData.getBomFile(undefined,'csv');
return {project:await eda.dmt_Project.getCurrentProjectInfo(),pages,netlist:await net.text(),bom:await bom.text()};
