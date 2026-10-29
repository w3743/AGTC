const planned=new Map(__CLI__.args.standardization.components.map(c=>[c.ref,c]));
const updated=[];
for(const c of await eda.sch_PrimitiveComponent.getAll(undefined,false)){
 const p=planned.get(c.getState_Designator());if(!p)continue;
 if(c.getState_SupplierId()!==p.supplier_part)throw new Error('供应编号变化 '+p.ref);
 if(!await eda.sch_PrimitiveComponent.modify(c,{name:p.name,manufacturer:p.manufacturer,manufacturerId:p.mpn,supplier:p.supplier}))throw new Error('规范属性失败 '+p.ref);
 updated.push(p.ref);
}
return {updated,status:'STANDARDIZED_SAME_SELECTED_IDENTITIES',saved:await eda.sch_Document.save()};
