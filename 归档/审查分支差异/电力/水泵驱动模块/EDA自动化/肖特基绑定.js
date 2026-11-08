const lib='0819f05c4eef4c71ace90d822a990e87';
const d=await eda.lib_Device.get('f6ea2eab8eb84723a354605d95acd217',lib);
d.association.footprintUuid='069b625b09bd496eb2ed424307f3e157';d.association.footprint={uuid:'069b625b09bd496eb2ed424307f3e157',libraryUuid:lib};
d.property.otherProperty['Footprint']='069b625b09bd496eb2ed424307f3e157';
delete d.association.model3D;delete d.property.otherProperty['3D Model'];delete d.property.otherProperty['3D Model Transform'];
const cs=await eda.sch_PrimitiveComponent.getAll();if(cs.some(c=>c.getState_Designator()==='D_5V'))throw Error('already created');
const c=await eda.sch_PrimitiveComponent.create(d,950,650,undefined,0,false,true,true);
await eda.sch_PrimitiveComponent.modify(c.getState_PrimitiveId(),{designator:'D_5V',name:'S310 / 100V',manufacturer:'onsemi',manufacturerId:'S310',supplier:'',supplierId:'',otherProperty:{'Manufacturer Part':'S310','Supplier Part':'','CAD Carrier MPN':'DIYI S310 symbol + official SMC footprint','CAD Binding':'symbol_footprint_composition'}});
await eda.sch_Document.save();return {id:c.getState_PrimitiveId(),footprint:c.getState_Footprint(),pins:await eda.sch_PrimitiveComponent.getAllPinsByPrimitiveId(c.getState_PrimitiveId())};
