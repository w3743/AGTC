const page='28c1f3740a9b8e02';
await eda.dmt_EditorControl.openDocument(page);
await new Promise(r=>setTimeout(r,1800));
if((await eda.dmt_SelectControl.getCurrentDocumentInfo())?.uuid!==page)throw Error('wrong page');
const all=await eda.sch_PrimitiveComponent.getAll(undefined,false);
const wanted=['Q_DUMP_SW','R_DUMP_SW_BASE','R_DUMP_IN_PD','R_DUMP_IN_SER'];
const items=[];
for(const c of all.filter(c=>wanted.includes(c.getState_Designator()))) {
 const pins=await eda.sch_PrimitiveComponent.getAllPinsByPrimitiveId(c.getState_PrimitiveId());
 items.push({ref:c.getState_Designator(),id:c.getState_PrimitiveId(),x:c.getState_X(),y:c.getState_Y(),rotation:c.getState_Rotation(),mirror:c.getState_Mirror(),mpn:c.getState_ManufacturerId(),pins:pins.map(p=>({number:p.getState_PinNumber(),name:p.getState_PinName(),x:p.getState_X(),y:p.getState_Y()}))});
}
return {items,mos:await eda.lib_Device.search('C347475'),r100:await eda.lib_Device.search('C105588'),r47:await eda.lib_Device.search('RC0603FR-0747KL')};
