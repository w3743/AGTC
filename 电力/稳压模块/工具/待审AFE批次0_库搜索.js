// Read-only. Returns full ILIB_DeviceSearchItem records for parent review.
// Do not replace returned items with hand-built {libraryUuid, uuid} objects.
await eda.dmt_EditorControl.openDocument('9546dca0bc8bce7f');
if ((await eda.dmt_SelectControl.getCurrentDocumentInfo())?.uuid !== '9546dca0bc8bce7f') throw new Error('Wrong schematic');
const libraryUuid = '0819f05c4eef4c71ace90d822a990e87';
const mpns = [
  'BQ7695202PFBR', 'SN74LVC1G11DCKR', 'TMUX1511PWR', 'FCX495TA', 'BAT46W_C915627',
  'ABD03K1000FT', '0603B224K500NT', 'TCC1210X7R225K101MT', 'GRM188R71H223KA01D',
  'GRM32ER72A475KE14L', 'GRM188R71E105KA12D', 'C0805C104K5RAC7800',
  'RC2512FK-7W1KL', 'RS-03L1005FT', 'PTFR0805B10K0N9', 'PTFR0805B47K0N9',
  'RC0603FR-074K7L', 'RC0603FR-07680RL', 'RT1206BRD07150KL',
  'HoAR0603-1/10W-10KR-0.1%-TCR25'
];
const rows = [];
for (const mpn of mpns) {
  const items = await eda.lib_Device.search(mpn, libraryUuid, undefined, undefined, 20, 1);
  rows.push({ mpn, exactNameMatches: (items || []).filter(x => x.name === mpn).length, items: items || [] });
}
return { project: await eda.dmt_Project.getCurrentProjectInfo(), libraryUuid, rows, note: 'LICAP SC0350-270-RSS is project-local/custom and must be resolved separately; no short UUID fallback is allowed.' };
