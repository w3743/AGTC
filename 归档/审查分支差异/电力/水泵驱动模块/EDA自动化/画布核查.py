from eda import *
r=invoke("const cs=await eda.sch_PrimitiveComponent.getAll();return cs.filter(c=>c.getState_ComponentType()!=='part').slice(0,8).map(c=>({id:c.getState_PrimitiveId(),type:c.getState_ComponentType(),name:c.getState_Name()}));")
print(r)
print(cli('doc','api','--class-name','SCH_Drc'))
