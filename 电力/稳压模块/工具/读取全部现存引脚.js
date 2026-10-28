await eda.dmt_EditorControl.openDocument('9546dca0bc8bce7f');
if((await eda.dmt_SelectControl.getCurrentDocumentInfo())?.uuid!=='9546dca0bc8bce7f') throw new Error('Wrong schematic');
const all=await eda.sch_PrimitiveComponent.getAll();const rows=[];
for(const c of all){const ref=c.getState_Designator();if(!ref)continue;const id=c.getState_PrimitiveId();const pins=await eda.sch_PrimitiveComponent.getAllPinsByPrimitiveId(id);rows.push({ref,id,pins:pins.map(p=>({n:p.getState_PinNumber(),x:p.getState_X(),y:p.getState_Y(),nc:p.getState_NoConnected?.()}))});}
return rows;
