const r=await eda.sch_Drc.check(true,true,true);return {result:r,keys:r.map(v=>Object.getOwnPropertyNames(v)),json:JSON.stringify(r)};
