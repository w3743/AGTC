const project = await eda.dmt_Project.getCurrentProjectInfo();
await eda.dmt_EditorControl.openDocument('9546dca0bc8bce7f');
const document = await eda.dmt_SelectControl.getCurrentDocumentInfo();
if(document?.uuid !== '9546dca0bc8bce7f') throw new Error('Wrong schematic');
const all=await eda.sch_PrimitiveComponent.getAll();
const components=all.filter(c=>c.getState_Designator()).map(c=>({
 id:c.getState_PrimitiveId(),ref:c.getState_Designator(),x:c.getState_X(),y:c.getState_Y(),
 value:c.getState_Name(),mpn:c.getState_ManufacturerId(),type:c.getState_ComponentType(),
 device:c.getState_Component(),bom:c.getState_AddIntoBom(),pcb:c.getState_AddIntoPcb()
}));
const wires=(await eda.sch_PrimitiveWire.getAll()).map(w=>({id:w.getState_PrimitiveId(),net:w.getState_Net(),line:w.getState_Line()}));
const ports=all.filter(c=>!c.getState_Designator()).map(c=>({id:c.getState_PrimitiveId(),net:c.getState_Net(),x:c.getState_X(),y:c.getState_Y(),type:c.getState_ComponentType()}));
return {project:project?.name,document,components,wires,ports};
