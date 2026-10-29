const a=await eda.sch_PrimitiveComponent.getAll(undefined,false);
const refs=a.filter(c=>c.getState_Designator()).map(c=>({ref:c.getState_Designator(),mpn:c.getState_ManufacturerId()}));
return {status:'READ_PROGRESS',count:refs.length,refs};
