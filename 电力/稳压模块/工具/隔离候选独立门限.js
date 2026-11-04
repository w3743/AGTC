await eda.dmt_EditorControl.openDocument('28c1f3740a9b8e02');
await new Promise(r=>setTimeout(r,1800));
const all=await eda.sch_PrimitiveComponent.getAll(undefined,false);
const maps={
 U_OV_EARLY:{'1':'BUS_EXT','6':'OV_EARLY_REF','7':'OV_EARLY_H','8':'OV_EARLY_L','9':'OV_EARLY_REF','10':'OV_EARLY_REF','11':'GND_SYS','16':'DUMP_OFF'},
 R_OV_EARLY_TOP:{'1':'OV_EARLY_REF','2':'OV_EARLY_H'},
 R_OV_EARLY_MID:{'1':'OV_EARLY_H','2':'OV_EARLY_L'},
 R_OV_EARLY_BOT:{'1':'OV_EARLY_L','2':'GND_SYS'}
};
const wired=[];
for(const [ref,map] of Object.entries(maps)){
 const comp=all.find(c=>c.getState_Designator()===ref);if(!comp)throw Error('missing '+ref);
 const pins=await eda.sch_PrimitiveComponent.getAllPinsByPrimitiveId(comp.getState_PrimitiveId());
 for(const pin of pins){
  const num=String(pin.getState_PinNumber()),net=map[num];
  if(!net){await pin.toAsync().setState_NoConnected(true).done();continue;}
  const x=Math.round(pin.getState_X()),y=Math.round(pin.getState_Y()),angle=pin.getState_Rotation()*Math.PI/180;
  const ex=Math.round(x+60*Math.cos(angle)),ey=Math.round(y+60*Math.sin(angle));
  const wire=await eda.sch_PrimitiveWire.create([x,y,ex,ey],net);if(!wire)throw Error('wire '+ref+':'+num);
  const port=await eda.sch_PrimitiveComponent.createNetPort('BI',net,ex,ey,pin.getState_Rotation(),false);if(!port)throw Error('port '+ref+':'+num);
  wired.push({ref,pin:num,net});
 }
}
await eda.sch_PrimitiveText.create(2720,1190,'ISOLATED REVIEW CANDIDATE - NOT FOR RELEASE',0,null,null,16);
await eda.sch_PrimitiveText.create(2720,1155,'LTC2965 independent thresholds: tolerance and timing NOT PASSED',0,null,null,12);
const save=await eda.sch_Document.save();
return {wired,save,status:'ISOLATED_REVIEW_NOT_RELEASED'};
