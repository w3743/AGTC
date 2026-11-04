const result={};
for(const q of ['RT1206BRD07150KL','RMC06031K1%N','RMC0603100K1%N','RMC060330K1%N']){
 const rows=await eda.lib_Device.search(q,undefined,undefined,undefined,10,1);
 result[q]=rows.map(r=>({uuid:r.uuid,libraryUuid:r.libraryUuid,name:r.name,manufacturerId:r.manufacturerId,supplierId:r.supplierId}));
}
return result;
