const project = await eda.dmt_Project.getCurrentProjectInfo();
const doc = await eda.dmt_SelectControl.getCurrentDocumentInfo();
const comps = await eda.sch_PrimitiveComponent.getAll();
const pick = (d) => {
  const c = comps.find(x => x.getState_Designator() === d);
  if (!c) return null;
  return {id:c.getState_PrimitiveId(), designator:c.getState_Designator(), uid:c.getState_UniqueId(), x:c.getState_X(), y:c.getState_Y(), rotation:c.getState_Rotation(), component:c.getState_Component(), symbol:c.getState_Symbol(), footprint:c.getState_Footprint(), addBom:c.getState_AddIntoBom(), addPcb:c.getState_AddIntoPcb(), other:c.getState_OtherProperty()};
};
const wiresCell = await eda.sch_PrimitiveWire.getAll('CELL_B9');
const wiresTap = await eda.sch_PrimitiveWire.getAll('TAP_B9');
const wireSummary = (arr) => arr.map(w => ({id:w.getState_PrimitiveId(), net:w.getState_Net(), line:w.getState_Line()}));
const searches = {};
for (const q of ['1x10','Conn_01x10','10pin','connector 10','fuse','Fuse']) {
  try { searches[q] = (await eda.lib_Device.search(q, undefined, undefined, undefined, 30, 1)).slice(0,20); }
  catch(e) { searches[q] = {error:String(e)}; }
}
return JSON.stringify({project,doc,componentCount:comps.length,dbat:pick('D_BAT'),dbreg:pick('D_BREG'),wiresCell:wireSummary(wiresCell),wiresTap:wireSummary(wiresTap),searches});
