await eda.dmt_EditorControl.openDocument('28c1f3740a9b8e02');
await new Promise(r=>setTimeout(r,1800));
const info=await eda.dmt_SelectControl.getCurrentDocumentInfo();
if(info?.uuid!=='28c1f3740a9b8e02')throw Error('wrong page');
const matches=[];
for(const c of await eda.sch_PrimitiveComponent.getAll(undefined,false)){
 const ref=c.getState_Designator();
 if(!/^(U_SC_OV|Q_PMOS_CTRL|Q_DUMP_SW|R_BUS_OV_TOP1|R_BUS_OV_TOP2|R_BUS_OV_BOT|R_PB6_BASE|R_QP_BE|R_QP_COL|R_DUMP_HW_PU)$/.test(ref||''))continue;
 const pins=await eda.sch_PrimitiveComponent.getAllPinsByPrimitiveId(c.getState_PrimitiveId());
 matches.push({ref,id:c.getState_PrimitiveId(),x:c.getState_X(),y:c.getState_Y(),rotation:c.getState_Rotation(),device:c.getState_Component(),pins:pins.map(p=>({number:p.getState_PinNumber(),x:p.getState_X(),y:p.getState_Y(),net:p.getState_Net?.()}))});
}
let found={};for(const q of ['LTC2965','LTC2965IDD','15MΩ','1N4148']){
 try{const x=await eda.lib_Device.search(q,undefined,undefined,undefined,15,1);found[q]=x;}catch(e){found[q]=String(e)}
}
return {info,matches,found};
