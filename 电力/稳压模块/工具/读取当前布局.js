const components=await eda.sch_PrimitiveComponent.getAll(undefined,false);
const rows=[];
for(const c of components){
 const pins=await eda.sch_PrimitiveComponent.getAllPinsByPrimitiveId(c.getState_PrimitiveId());
 rows.push({id:c.getState_PrimitiveId(),ref:c.getState_Designator(),name:c.getState_Name(),
  device:c.getState_Component(),x:c.getState_X(),y:c.getState_Y(),rotation:c.getState_Rotation(),mirror:c.getState_Mirror(),
  uniqueId:c.getState_UniqueId(),subpart:c.getState_SubPartName(),mpn:c.getState_ManufacturerId(),supplier:c.getState_SupplierId(),
  pins:(pins||[]).map(p=>({number:p.getState_PinNumber(),name:p.getState_PinName(),x:p.getState_X(),y:p.getState_Y(),rotation:p.getState_Rotation(),noConnected:p.getState_NoConnected()}))});
}
const wires=await eda.sch_PrimitiveWire.getAll();
return {status:'READ_LAYOUT',components:rows,wireCount:wires.length};
