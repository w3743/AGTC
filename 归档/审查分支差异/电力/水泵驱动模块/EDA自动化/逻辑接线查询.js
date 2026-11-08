await eda.dmt_EditorControl.openDocument('bafe8a48e4625018');
return {project:await eda.dmt_Project.getCurrentProjectInfo(),parts:(await eda.sch_PrimitiveComponent.getAll()).filter(c=>c.getState_ComponentType()==='part').map(c=>({ref:c.getState_Designator(),x:c.getState_X(),y:c.getState_Y(),rotation:c.getState_Rotation()}))};
