const existing=await eda.sch_PrimitiveComponent.getAll();
const byRef=new Map(existing.filter(c=>c.getState_ComponentType()==='part').map(c=>[c.getState_Designator(),c]));
const out=[];
for(const p of pumpParts.slice(globalThis.buildStart,globalThis.buildStart+10)){
 let c=byRef.get(p.ref);
 if(!c)c=await eda.sch_PrimitiveComponent.create({libraryUuid:'0819f05c4eef4c71ace90d822a990e87',uuid:p.deviceUuid,libraryType:'3'},p.x,p.y,undefined,p.rotation,false,true,true);
 if(!c)throw Error('create '+p.ref);
 c=await eda.sch_PrimitiveComponent.modify(c.getState_PrimitiveId(),{designator:p.ref,name:p.value,manufacturer:p.manufacturer,manufacturerId:p.mpn,supplier:p.supplierId?'LCSC':'',supplierId:p.supplierId,otherProperty:{'Manufacturer Part':p.mpn,'Supplier Part':p.supplierId,'CAD Carrier MPN':p.carrierMpn,'CAD Binding':p.binding,'Assembly':p.ref==='R_TERM'?'DNP default; fit only at CAN bus end':'FIT'}});
 const pins=await eda.sch_PrimitiveComponent.getAllPinsByPrimitiveId(c.getState_PrimitiveId());
 const actual=pins.map(v=>v.pinNumber).sort(),expected=Object.keys(p.pins).sort();
 if(JSON.stringify(actual)!==JSON.stringify(expected))throw Error(p.ref+' pin mismatch '+JSON.stringify(actual));
 out.push({ref:p.ref,id:c.getState_PrimitiveId(),pins,footprint:c.getState_Footprint()});
}
await eda.sch_Document.save();return out;
