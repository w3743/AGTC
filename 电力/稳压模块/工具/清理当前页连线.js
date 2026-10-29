const wires=await eda.sch_PrimitiveWire.getAll();
if(wires.length&&!await eda.sch_PrimitiveWire.delete(wires.map(w=>w.getState_PrimitiveId())))throw new Error('删除旧线失败');
const ports=(await eda.sch_PrimitiveComponent.getAll(undefined,false)).filter(c=>!c.getState_Designator());
if(ports.length&&!await eda.sch_PrimitiveComponent.delete(ports))throw new Error('删除旧端口失败');
return {status:'CLEARED_FOR_EXPLICIT_PIN_CONNECTIONS',removedWires:wires.length,removedPorts:ports.length,saved:await eda.sch_Document.save()};
