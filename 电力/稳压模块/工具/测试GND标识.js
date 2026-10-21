await eda.dmt_EditorControl.openDocument('9546dca0bc8bce7f');
const d=await eda.dmt_SelectControl.getCurrentDocumentInfo();if(d?.uuid!=='9546dca0bc8bce7f')throw new Error('wrong doc');
const c=await eda.sch_PrimitiveComponent.createNetFlag('Ground','GND_SYS',5000,5000,0,false);
if(!c) throw new Error('ground create failed');
const id=c.getState_PrimitiveId();const pins=await eda.sch_PrimitiveComponent.getAllPinsByPrimitiveId(id);
const result={id,type:c.getState_ComponentType(),net:c.getState_Net(),x:c.getState_X(),y:c.getState_Y(),pins:(pins||[]).map(p=>({n:p.getState_PinNumber(),name:p.getState_PinName(),x:p.getState_X(),y:p.getState_Y()}))};
const del=await eda.sch_PrimitiveComponent.delete([id]);return {result,del};
