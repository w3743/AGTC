await eda.dmt_Schematic.modifySchematicPageTitleBlock(false);
const cs=await eda.sch_PrimitiveComponent.getAll();if(cs.filter(c=>c.getState_ComponentType()==='part').length)throw Error('Expected blank page');
const parts=[['U_DRV','7e7ea239b19d4650ae10d06c6ce34213',600,1500],['U_5V','1c58e8e86a7942da8bdc5836932b2fc0',600,650],['U_CAN','57d36f806c89417aac9cd75ebb7e0b46',2700,650],['J_MCU_A','40a18e0d50984fa1bb84b43ded96e4bd',2500,1550],['J_MCU_B','40a18e0d50984fa1bb84b43ded96e4bd',3250,1550]];
const out=[];
for(const [ref,uuid,x,y] of parts){const c=await eda.sch_PrimitiveComponent.create({libraryUuid:'0819f05c4eef4c71ace90d822a990e87',uuid,libraryType:'3'},x,y,undefined,0,false,true,true);if(!c)throw Error(ref);await eda.sch_PrimitiveComponent.modify(c.getState_PrimitiveId(),{designator:ref});const pins=await eda.sch_PrimitiveComponent.getAllPinsByPrimitiveId(c.getState_PrimitiveId());out.push({ref,id:c.getState_PrimitiveId(),pins});}
await eda.sch_Document.save();return out;
