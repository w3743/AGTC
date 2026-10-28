await eda.dmt_EditorControl.openDocument('9546dca0bc8bce7f');
const d=await eda.dmt_SelectControl.getCurrentDocumentInfo(); if(d?.uuid!=='9546dca0bc8bce7f') throw new Error('wrong doc');
const c=await eda.sch_PrimitiveComponent.createNetPort('BI','TMP_NETPORT_TEST',5000,5000,0,false);
if(!c) throw new Error('netport create failed');
const id=c.getState_PrimitiveId();
const pins=await eda.sch_PrimitiveComponent.getAllPinsByPrimitiveId(id);
const result={id,type:c.getState_ComponentType(),net:c.getState_Net(),x:c.getState_X(),y:c.getState_Y(),ref:c.getState_Designator(),pins:(pins||[]).map(p=>({number:p.getState_PinNumber(),name:p.getState_PinName(),x:p.getState_X(),y:p.getState_Y(),net:p.getState_Net?.()}))};
const del=await eda.sch_PrimitiveComponent.delete([id]);
return {result,del};
