const fs=require('fs'), path=require('path'),crypto=require('crypto');
const dir=__dirname, saved=JSON.parse(fs.readFileSync(path.join(dir,'重开官方完整.json')));
const files={'重开官方.enet':saved.netlist,'重开官方BOM.csv':saved.bom,'重开官方02.svg':saved.pages.find(x=>x.id==='28c1f3740a9b8e02').svg};
for(const [name,text] of Object.entries(files))fs.writeFileSync(path.join(dir,name),text);
const base=path.resolve(dir,'../候选_预充与超容隔离_R1/候选官方.enet'),next=path.join(dir,'重开官方.enet');
const sha=f=>crypto.createHash('sha256').update(fs.readFileSync(f)).digest('hex');
function load(f){let j=JSON.parse(fs.readFileSync(f));let m={};for(const c of Object.values(j.components)){let p=c.props||{},ref=p.Designator;if(!ref||m[ref])throw Error('bad designator '+ref);let pins={};for(const [n,v] of Object.entries(c.pinInfoMap||{}))pins[n]=v.net||'';m[ref]={mpn:p['Manufacturer Part']||'',value:p.Value||'',pins}}return m}
let a=load(base),b=load(next),refs=[...new Set([...Object.keys(a),...Object.keys(b)])].sort();
let diff=refs.filter(k=>JSON.stringify(a[k])!==JSON.stringify(b[k])).map(k=>({ref:k,old:a[k]||null,new:b[k]||null}));
const result={scope:'R1 official saved ENET versus independently reopened candidate, MPN/value/pin net comparison; no operating qualification',before_sha256:sha(base),after_sha256:sha(next),counts:[Object.keys(a).length,Object.keys(b).length],diff,pages:saved.pages.map(p=>({id:p.id,drc:p.drc})),outputs:Object.fromEntries(Object.keys(files).map(f=>[f,sha(path.join(dir,f))]))};
fs.writeFileSync(path.join(dir,'官方重开与逐针差异.json'),JSON.stringify(result,null,2)+'\n');
console.log(JSON.stringify(result));
