await eda.dmt_EditorControl.openDocument('9546dca0bc8bce7f');
const doc=await eda.dmt_SelectControl.getCurrentDocumentInfo(); if(doc?.uuid!=='9546dca0bc8bce7f')throw new Error('wrong doc');
const lib='0819f05c4eef4c71ace90d822a990e87';
const ids=['11a7ff39fa6a461a89d259faea7aed4a','9d334130c9d0406ca221f499d6bd292d','GRM188R71E475KA12D','GRM188R71E475KE15D','GRM188R71E105KA12D','GCM219R71E105KA12D'];
const out=[];for(const id of ids){if(id.includes('-')||id.startsWith('GRM')||id.startsWith('GCM')){const a=await eda.lib_Device.search(id,lib,undefined,undefined,20,1);out.push({key:id,items:(a||[]).slice(0,5).map(x=>({uuid:x.uuid,libraryUuid:x.libraryUuid,name:x.name}))});}else{const x=await eda.lib_Device.get(id,lib);out.push({key:id,item:x&&{uuid:x.uuid,name:x.name,libraryUuid:x.libraryUuid,property:x.property,association:x.association}})}}return {out};
