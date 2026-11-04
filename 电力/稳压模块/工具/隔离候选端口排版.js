await eda.dmt_EditorControl.openDocument('28c1f3740a9b8e02');await new Promise(r=>setTimeout(r,1700));
const portIds=['2ff5c901dcca4f96','f5cb246a18f3c48b','00e681db130f568c','035d392d69703695','f5daf2caff06bbaf','31952a50cd6d9efe','33456f60eef8981f','e216afa5f598a4e6'];
const wireIds=['d133a7dcec9db539','82e7dbbd96900568','1d6f8e0b4f0384c2','fde7abad5194b714','2614bbe82ec81119','d6b1332bc7f7224e','035960086182196b','2aaedebb4663dea6'];
if(!await eda.sch_PrimitiveComponent.delete(portIds))throw Error('delete ports');
if(!await eda.sch_PrimitiveWire.delete(wireIds))throw Error('delete wires');
const c=(await eda.sch_PrimitiveComponent.getAll(undefined,false)).find(x=>x.getState_Designator()==='U_OV_EARLY');
const pinMap={'1':['BUS_EXT',2890,870,2740], '6':['OV_EARLY_REF',2900,800,2740], '7':['OV_EARLY_H',2920,730,2740], '8':['OV_EARLY_L',2940,660,2740],
 '9':['OV_EARLY_REF',3080,660,3200], '10':['OV_EARLY_REF',3110,730,3200], '11':['GND_SYS',3090,800,3200], '16':['DUMP_OFF',3100,900,3200]};
let wired=[];
for(const p of await eda.sch_PrimitiveComponent.getAllPinsByPrimitiveId(c.getState_PrimitiveId())){
 const n=String(p.getState_PinNumber());if(!pinMap[n])continue;
 const [net,mx,ey,ex]=pinMap[n],x=Math.round(p.getState_X()),y=Math.round(p.getState_Y());
 if(!await eda.sch_PrimitiveWire.create([x,y,mx,y,mx,ey,ex,ey],net))throw Error('wire '+n);
 if(!await eda.sch_PrimitiveComponent.createNetPort('BI',net,ex,ey,p.getState_Rotation(),false))throw Error('port '+n);
 wired.push({pin:n,net});
}
const saved=await eda.sch_Document.save();return {saved,wired,status:'ISOLATED_REVIEW_NOT_RELEASED'};
