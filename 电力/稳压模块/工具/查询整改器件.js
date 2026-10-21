const project=await eda.dmt_Project.getCurrentProjectInfo();
await eda.dmt_EditorControl.openDocument('9546dca0bc8bce7f');
const doc=await eda.dmt_SelectControl.getCurrentDocumentInfo();
if(doc?.uuid!=='9546dca0bc8bce7f') throw new Error('wrong doc');
const lib='0819f05c4eef4c71ace90d822a990e87';
const keys=['BQ7695202PFBR','ABD03K1000FT','0603B224K500NT','GRM188R71E105KA12D','GRM188R71E223KA01D','TCC1210X7R225K101MT','FCX495TA','BAT46W','RC2512FK-7W1KL','RS-03L1005FT','SN74LVC1G11DCKR','TMUX1511PWR','RC0603FR-074K7L','RC0603FR-0747KL','RC0603FR-0710KL','RC0603FR-07100RL','RC0603FR-07680RL'];
const rows=[];
for(const key of keys){const a=await eda.lib_Device.search(key,lib,undefined,undefined,20,1);rows.push({key,items:(a||[]).slice(0,8).map(x=>({uuid:x.uuid,libraryUuid:x.libraryUuid,name:x.name,manufacturerId:x.property?.manufacturerId,supplierId:x.property?.supplierId,designator:x.property?.designator,association:x.association}))});}
return {project:project?.name,lib,rows};
