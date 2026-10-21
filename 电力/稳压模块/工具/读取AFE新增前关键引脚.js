await eda.dmt_EditorControl.openDocument('9546dca0bc8bce7f');
if((await eda.dmt_SelectControl.getCurrentDocumentInfo())?.uuid!=='9546dca0bc8bce7f') throw new Error('Wrong schematic');
const all=await eda.sch_PrimitiveComponent.getAll();
const result=[];
for(const c of all){const ref=c.getState_Designator();if(!ref||!/^(R_DRV_FAULT_PD|R_PB5_PERMIT|R_DRV_SO_SER|C_DRV_SO|U1|U_SC_OV)$/.test(ref))continue;const id=c.getState_PrimitiveId();const pins=await eda.sch_PrimitiveComponent.getAllPinsByPrimitiveId(id);result.push({id,ref,pins:pins.map(p=>({n:p.getState_PinNumber(),name:p.getState_PinName(),x:p.getState_X(),y:p.getState_Y(),net:p.getState_Net?.(),nc:p.getState_NoConnected?.()}))});}
const ports=all.filter(c=>!c.getState_Designator()).map(c=>({id:c.getState_PrimitiveId(),net:c.getState_Net(),x:c.getState_X(),y:c.getState_Y(),type:c.getState_ComponentType()})).filter(p=>p.x>=850&&p.x<=1400&&p.y>=1700&&p.y<=2100||p.x>2900&&p.y>2200);
return {components:result,ports};
