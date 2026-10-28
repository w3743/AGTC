const project = await eda.dmt_Project.getCurrentProjectInfo();
await eda.dmt_EditorControl.openDocument('9546dca0bc8bce7f');
const document = await eda.dmt_SelectControl.getCurrentDocumentInfo();
if(document?.uuid !== '9546dca0bc8bce7f') throw new Error('Wrong schematic');
const wanted=/^(J_CORE_P[12]|U1|R_BUS_HI|R_BUS_LO|C_BUS_ADC|R_TEST_HI|R_TEST_LO|C_TEST_FB|R_T[1-8]_(HI|LO|TEST)|C_T[1-8]_FILTER|R_DRV_SO_SER|C_DRV_SO|R_PB5_PERMIT|U_SC_OV|R_SC_OV_BOT)$/;
const all=await eda.sch_PrimitiveComponent.getAll();
const components=[];
for(const c of all){const ref=c.getState_Designator(); if(!ref||!wanted.test(ref))continue; const id=c.getState_PrimitiveId(); const pins=await eda.sch_PrimitiveComponent.getAllPinsByPrimitiveId(id); components.push({id,ref,x:c.getState_X(),y:c.getState_Y(),rotation:c.getState_Rotation(),pins:(pins||[]).map(p=>({number:p.getState_PinNumber(),name:p.getState_PinName(),x:p.getState_X(),y:p.getState_Y(),rotation:p.getState_Rotation(),net:p.getState_Net?.(),nc:p.getState_NoConnected?.()}))});}
const allWires=await eda.sch_PrimitiveWire.getAll();
return {project:project?.name,document,components,wires:allWires.map(w=>({id:w.getState_PrimitiveId(),net:w.getState_Net(),line:w.getState_Line()}))};
