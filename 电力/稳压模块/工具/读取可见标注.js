return {texts:(await eda.sch_PrimitiveText.getAll()).map(t=>({id:t.getState_PrimitiveId(),content:t.getState_Content()})),
 attrs:(await eda.sch_PrimitiveAttribute.getAll()).slice(0,10).map(a=>({id:a.getState_PrimitiveId(),parent:a.getState_ParentPrimitiveId(),visible:a.getState_ValueVisible(),key:a.getState_Key()}))};
