// Dry-run only. Parent review must explicitly change EXECUTE to true before any deletion.
const EXECUTE = false;
const page = '9546dca0bc8bce7f';
await eda.dmt_EditorControl.openDocument(page);
if ((await eda.dmt_SelectControl.getCurrentDocumentInfo())?.uuid !== page) throw new Error('Wrong schematic');

const deleteRefs = [
  'C_T1_FILTER','C_T2_FILTER','C_T3_FILTER','C_T4_FILTER','C_T5_FILTER','C_T6_FILTER','C_T7_FILTER','C_TEST_FB',
  'R_T1_HI','R_T1_LO','R_T1_TEST','R_T2_HI','R_T2_LO','R_T2_TEST','R_T3_HI','R_T3_LO','R_T3_TEST',
  'R_T4_HI','R_T4_LO','R_T4_TEST','R_T5_HI','R_T5_LO','R_T5_TEST','R_T6_HI','R_T6_LO','R_T6_TEST',
  'R_T7_HI','R_T7_LO','R_T7_TEST','R_TEST_HI','R_TEST_LO'
];
const deletePorts = [
  {id:'194619fda6f2a294',net:'AD_T1',x:870,y:1555},{id:'86e6711c6f4dbbad',net:'AD_T1',x:1600,y:790},
  {id:'31053b122f063227',net:'AD_T2',x:650,y:1565},{id:'ac6c34d4dde789a7',net:'AD_T2',x:2200,y:790},
  {id:'c17fcdf0efd3d5ba',net:'AD_T3',x:870,y:1565},{id:'f3af393df7ca7d8a',net:'AD_T3',x:2800,y:790},
  {id:'29f1a1f643954c06',net:'AD_T4',x:650,y:1575},{id:'cf39ef43febc657b',net:'AD_T4',x:3400,y:790},
  {id:'8b0c3602cde1b639',net:'AD_T5',x:870,y:1575},{id:'1272a0430dc865b6',net:'AD_T5',x:3400,y:190},
  {id:'32f827f208f48ad5',net:'AD_T6',x:650,y:1585},{id:'f45622316a36684b',net:'AD_T6',x:2800,y:190},
  {id:'33f3a23f47e0d142',net:'AD_T7',x:870,y:1585},{id:'db50b42e2759270b',net:'AD_T7',x:2200,y:190},
  {id:'9cd3eac57024a010',net:'TEST',x:650,y:1625},{id:'e070fceae6003ffa',net:'TEST',x:1600,y:850},
  {id:'16e305c413a9e698',net:'TEST',x:1670,y:1270},{id:'cee02e643eab290e',net:'TEST',x:2200,y:250},
  {id:'c9dc9cf0bd3735e8',net:'TEST',x:2200,y:850},{id:'027bd79cebeadd99',net:'TEST',x:2800,y:250},
  {id:'686e6fe119e24c15',net:'TEST',x:2800,y:850},{id:'7c612aa36ee95d16',net:'TEST',x:3400,y:250},
  {id:'5b2979fc72435feb',net:'TEST',x:3400,y:850},{id:'4cdce91aa896de8d',net:'TEST_FB',x:350,y:1615},
  {id:'e52781b48a714f50',net:'TEST_FB',x:2070,y:1270},{id:'df217b7f3958eb7e',net:'NTC_MAIN',x:870,y:1625}
];
const protectedRefs = [
  'SC1','SC2','SC3','SC4','SC5','SC6','SC7','SC8',
  'U_BAL1','U_BAL2','U_BAL3','U_BAL4','U_BAL5','U_BAL6','U_BAL7','U_BAL8',
  'C_BAL1','C_BAL2','C_BAL3','C_BAL4','C_BAL5','C_BAL6','C_BAL7','C_BAL8',
  'R_BAL1','R_BAL2','R_BAL3','R_BAL4','R_BAL5','R_BAL6','R_BAL7','R_BAL8','U1'
];
const all = await eda.sch_PrimitiveComponent.getAll();
const components = all.filter(c => c.getState_Designator());
const byRef = new Map();
for (const c of components) {
  const ref = c.getState_Designator();
  if (!byRef.has(ref)) byRef.set(ref, []);
  byRef.get(ref).push(c);
}
const missingRefs = deleteRefs.filter(ref => !byRef.has(ref));
const duplicateRefs = deleteRefs.filter(ref => (byRef.get(ref) || []).length !== 1);
const protectedMissing = protectedRefs.filter(ref => !byRef.has(ref));
if (missingRefs.length || duplicateRefs.length || protectedMissing.length) {
  throw new Error(JSON.stringify({missingRefs,duplicateRefs,protectedMissing}));
}
const portObjects = all.filter(c => !c.getState_Designator());
const portsById = new Map(portObjects.map(c => [c.getState_PrimitiveId(), c]));
const portMismatches = [];
for (const expected of deletePorts) {
  const p = portsById.get(expected.id);
  if (!p || p.getState_Net() !== expected.net || p.getState_X() !== expected.x || p.getState_Y() !== expected.y) {
    portMismatches.push({expected,actual:p&&{id:p.getState_PrimitiveId(),net:p.getState_Net(),x:p.getState_X(),y:p.getState_Y()}});
  }
}
if (portMismatches.length) throw new Error(JSON.stringify({portMismatches}));
const componentIds = deleteRefs.map(ref => byRef.get(ref)[0].getState_PrimitiveId());
const portIds = deletePorts.map(p => p.id);
const wireIdsToDelete = [];
if (wireIdsToDelete.length) throw new Error('This batch is forbidden from deleting wires');
if (!EXECUTE) return {status:'DRY_RUN_ONLY',deleteRefs,componentIds,deletePorts,wireIdsToDelete,protectedRefsVerified:protectedRefs.length};

const ok = await eda.sch_PrimitiveComponent.delete([...componentIds,...portIds]);
if (!ok) throw new Error('Exact ref/port deletion failed');
const after = await eda.sch_PrimitiveComponent.getAll();
const afterRefs = new Set(after.map(c => c.getState_Designator()).filter(Boolean));
const afterIds = new Set(after.map(c => c.getState_PrimitiveId()));
const remainingRefs = deleteRefs.filter(ref => afterRefs.has(ref));
const remainingPortIds = portIds.filter(id => afterIds.has(id));
if (remainingRefs.length || remainingPortIds.length) throw new Error(JSON.stringify({remainingRefs,remainingPortIds}));
return {status:'EXACT_ALLOWLIST_DELETED_NO_WIRES',deletedRefs:deleteRefs,deletedPortIds:portIds,wireIdsToDelete};
