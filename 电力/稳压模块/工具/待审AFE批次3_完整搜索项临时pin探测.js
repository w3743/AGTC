// Disabled probe template. Parent review is required before EXECUTE is changed.
// Each part is looked up with lib_Device.search; the complete returned ILIB_DeviceSearchItem
// is passed to create. The temporary instance is deleted in finally and never saved.
const EXECUTE = false;
const page = '9546dca0bc8bce7f';
const libraryUuid = '0819f05c4eef4c71ace90d822a990e87';
await eda.dmt_EditorControl.openDocument(page);
if ((await eda.dmt_SelectControl.getCurrentDocumentInfo())?.uuid !== page) throw new Error('Wrong schematic');

const probes = [
  {mpn:'BQ7695202PFBR', pins:48},
  {mpn:'SN74LVC1G11DCKR', pins:6},
  {mpn:'TMUX1511PWR', pins:14},
  {mpn:'FCX495TA', pins:3},
  {mpn:'BAT46W_C915627', pins:2}
];
if (!EXECUTE) return {status:'DRY_RUN_ONLY',libraryUuid,probes,note:'No library object is created until parent reviews and explicitly enables this probe.'};

const results = [];
for (const spec of probes) {
  const items = await eda.lib_Device.search(spec.mpn,libraryUuid,undefined,undefined,20,1);
  const exact = (items || []).filter(x => x.name === spec.mpn || x.property?.manufacturerId === spec.mpn);
  if (exact.length !== 1) throw new Error(JSON.stringify({mpn:spec.mpn,exactMatches:exact.length,items}));
  const item = exact[0];
  let c;
  try {
    c = await eda.sch_PrimitiveComponent.create(item,6000,6000,undefined,0,false,false,false);
    if (!c) throw new Error('Create failed for '+spec.mpn);
    const id = c.getState_PrimitiveId();
    const pins = await eda.sch_PrimitiveComponent.getAllPinsByPrimitiveId(id);
    if ((pins || []).length !== spec.pins) throw new Error(JSON.stringify({mpn:spec.mpn,expectedPinCount:spec.pins,actualPinCount:(pins||[]).length}));
    results.push({mpn:spec.mpn,searchItem:item,pins:(pins||[]).map(p=>({number:p.getState_PinNumber(),name:p.getState_PinName(),x:p.getState_X(),y:p.getState_Y(),rotation:p.getState_Rotation()}))});
  } finally {
    if (c) {
      const id = c.getState_PrimitiveId();
      const ok = await eda.sch_PrimitiveComponent.delete([id]);
      if (!ok) throw new Error('Temporary probe cleanup failed for '+spec.mpn+' id='+id);
    }
  }
}
return {status:'TEMPORARY_PIN_PROBES_DELETED',results,noSaveCalled:true};
