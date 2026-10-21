await eda.dmt_EditorControl.openDocument('90aa6c46493624cf');
const page=await eda.dmt_Schematic.getCurrentSchematicPageInfo();
if(page?.uuid!=='90aa6c46493624cf')throw new Error('wrong source page');
const saved=await eda.sch_Document.save();
const enet=await eda.sch_ManufactureData.getNetlistFile('FOC_G431_P1','EasyEDA');
const bom=await eda.sch_ManufactureData.getBomFile('FOC_G431_BOM','csv');
if(!enet||!bom)throw new Error('missing official export');
const drc=await eda.sch_Drc.check(false,false,true);
return {page:page.uuid,saved,enet:await enet.text(),bom:await bom.text(),drc};
