let n=0;for(const w of await eda.sch_PrimitiveWire.getAll()){
 const id=w.getState_PrimitiveId();if(!id)continue;
 for(const a of await eda.sch_PrimitiveAttribute.getAll(id))if(a.getState_ParentPrimitiveId()===id&&a.getState_Key()==='Name'){await eda.sch_PrimitiveAttribute.modify(a.getState_PrimitiveId(),{valueVisible:false});n++;}
}await eda.sch_Document.save();return {wireNamesHidden:n};
