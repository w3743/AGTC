"""Persist two whitelisted text positions through the official file API.

Component/text setters did not persist these positions. Only x/y payload bytes
of the named TEXT records may change; every other byte must remain identical.
"""
import hashlib, json
from pathlib import Path
import cad

paths = [cad.M/'sch/Schematic1/P1.esch2',
         cad.M.parent/'归档/原始材料/bms-local-wiring-runtime/sch/Schematic1/P1.esch2']
targets = {'86038ddb7bf60a52': [3592,2561,3650,2445],
           '3a639241df400b87': [3362,2541,3135,2468]}
before = {str(p): p.read_bytes() for p in paths}
for p, data in before.items():
    assert hashlib.sha256(data).hexdigest() == 'caa109e9aa8261a9f6211514a766f0e33824786195dfbd10e05ea17e05e3325b'
print(cad.call('return await eda.dmt_EditorControl.closeDocument("350a4904c4ec87b3@3d3f01bde1a4fab511fcd4bb27a3cd7cb5ed1af5067a969979918c5676205175");'))
code = '''
const paths=PATHS, targets=TARGETS, results=[];
for(const path of paths){
 const f=await eda.sys_FileSystem.readFileFromFileSystem(path);if(!f)throw Error('missing '+path);
 const s=await f.text();let a=s;let edits=0;
 for(const [id,[ox,oy,x,y]] of Object.entries(targets)){
  const ls=a.split(/(?<=\\n)/);let n=0;
  a=ls.map(line=>{
   if(!line.startsWith('{"type":"TEXT",')||!line.includes('"id":"'+id+'"'))return line;
   const key='||{"x":'+ox+',"y":'+oy+',';
   if(!line.includes(key))throw Error('unexpected original '+id);
   n++;return line.replace(key,'||{"x":'+x+',"y":'+y+',');
  }).join('');if(n!==1)throw Error('TEXT match count '+id+':'+n);edits+=n;
 }
 const strip=t=>t.split(/(?<=\\n)/).filter(l=>!l.startsWith('{"type":"TEXT",')).join('');
 if(strip(s)!==strip(a))throw Error('non-TEXT changed');
 if(!await eda.sys_FileSystem.saveFileToFileSystem(path,new Blob([a],{type:'text/plain'}),undefined,true))throw Error('write failed');
 const r=await eda.sys_FileSystem.readFileFromFileSystem(path);
 if(await r.text()!==a)throw Error('readback mismatch');
 results.push({path,edits,nonTextBytesUnchanged:true,readback:true});
}
return results;
'''.replace('PATHS',json.dumps([p.as_posix() for p in paths])).replace('TARGETS',json.dumps(targets))
result=cad.call(code)
for p in paths:
    old=before[str(p)];new=p.read_bytes()
    ob=old.splitlines(keepends=True);nb=new.splitlines(keepends=True)
    assert len(ob)==len(nb)
    diff=[(o,n) for o,n in zip(ob,nb) if o!=n]
    assert len(diff)==2 and all(o.startswith(b'{"type":"TEXT",') for o,n in diff)
assert paths[0].read_bytes()==paths[1].read_bytes()
cad.write(cad.E/'text-persistence.json',{'officialApiResult':result,'whitelist':targets,
    'beforeSHA256':{p:hashlib.sha256(b).hexdigest() for p,b in before.items()},
    'afterSHA256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
    'onlyTwoTextXYPayloadsChanged':True})
print(result)
