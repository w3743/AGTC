// Dry-run only. Requires root review and the new R_BUS_HI2 already placed and pin-verified.
// Replaces only listed labels at their exact baseline IDs/coordinates; does not delete wires.
const EXECUTE = false;
const page = '9546dca0bc8bce7f';
await eda.dmt_EditorControl.openDocument(page);
if ((await eda.dmt_SelectControl.getCurrentDocumentInfo())?.uuid !== page) throw new Error('Wrong schematic');

const replacements = [
  {id:'ac75d06bf61e8782',old:'AD_T8',net:'SC_SUM',x:130,y:1615},
  {id:'1de5b4e533aa4b6f',old:'AD_T8',net:'SC_SUM_RAW',x:1600,y:190},
  {id:'0ef84038c66b9f1b',old:'BUS_ADC',net:'BUS_ADC_RAW',x:1520,y:1270},
  {id:'245e1a11b7d693b9',old:'IMON',net:'IMON_RAW',x:900,y:1780},
  // Only execute after R_BUS_HI2 pin1=BUS_EXT and pin2=BUS_HI_MID has been read back.
  {id:'2be5923f5eeab898',old:'BUS_EXT',net:'BUS_HI_MID',x:1120,y:1330}
];
const newPorts = [
  {net:'BQ_I2C_SCL',x:650,y:1625},
  {net:'BQ_I2C_SDA',x:870,y:1625},
  {net:'NTC_MAIN',x:870,y:1555}
];
const all = await eda.sch_PrimitiveComponent.getAll();
const ports = all.filter(c => !c.getState_Designator());
const byId = new Map(ports.map(p => [p.getState_PrimitiveId(),p]));
const mismatches = [];
for (const e of replacements) {
  const p = byId.get(e.id);
  if (!p || p.getState_Net() !== e.old || p.getState_X() !== e.x || p.getState_Y() !== e.y) {
    mismatches.push({expected:e,actual:p&&{id:p.getState_PrimitiveId(),net:p.getState_Net(),x:p.getState_X(),y:p.getState_Y()}});
  }
}
if (mismatches.length) throw new Error(JSON.stringify({mismatches}));
const componentRefs = new Set(all.map(c => c.getState_Designator()).filter(Boolean));
if (!componentRefs.has('R_BUS_HI2')) {
  if (!EXECUTE) return {status:'DRY_RUN_PENDING_R_BUS_HI2',replacements,newPorts,wireIdsToDelete:[],requiredBeforeExecute:'Place R_BUS_HI2 and verify pin1=BUS_EXT, pin2=BUS_HI_MID'};
  throw new Error('R_BUS_HI2 must exist before changing BUS_EXT midpoint label');
}
const rhi2 = all.find(c => c.getState_Designator() === 'R_BUS_HI2');
const rhiPins = await eda.sch_PrimitiveComponent.getAllPinsByPrimitiveId(rhi2.getState_PrimitiveId());
const rhiNet = Object.fromEntries(rhiPins.map(p => [p.getState_PinNumber(),p.getState_Net?.()]));
if (rhiNet['1'] !== 'BUS_EXT' || rhiNet['2'] !== 'BUS_HI_MID') throw new Error(JSON.stringify({R_BUS_HI2:rhiNet}));
const duplicateTargets = newPorts.filter(e => ports.some(p => p.getState_Net() === e.net && p.getState_X() === e.x && p.getState_Y() === e.y));
if (duplicateTargets.length) throw new Error(JSON.stringify({duplicateTargets}));
const wireIdsToDelete = [];
if (wireIdsToDelete.length) throw new Error('This batch is forbidden from deleting wires');
if (!EXECUTE) return {status:'DRY_RUN_ONLY',replacements,newPorts,wireIdsToDelete,verifiedR_BUS_HI2:rhiNet};

for (const e of replacements) {
  const ok = await eda.sch_PrimitiveComponent.delete([e.id]);
  if (!ok) throw new Error('Could not delete exact old port '+e.id);
  const created = await eda.sch_PrimitiveComponent.createNetPort('BI',e.net,e.x,e.y,0,false);
  if (!created) throw new Error('Could not create replacement port '+e.net);
}
for (const e of newPorts) {
  const created = await eda.sch_PrimitiveComponent.createNetPort('BI',e.net,e.x,e.y,0,false);
  if (!created) throw new Error('Could not create connector port '+e.net);
}
return {status:'EXACT_PORT_LABELS_REPLACED_NO_WIRES',replacements,newPorts,wireIdsToDelete};
