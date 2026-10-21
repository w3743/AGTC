"""Persist reviewed redundant NET-attribute removal via official file API."""
from pathlib import Path
import hashlib,json,sys
HERE=Path(__file__).resolve().parent;MODULE=HERE.parents[1]
sys.path.insert(0,str(MODULE/'检查/tmp'));import drc_fix as d
plan=json.loads((HERE/'重复网标清理分析.json').read_text(encoding='utf-8-sig'))
backup=json.loads((HERE/'latest-drc-backup.json').read_text(encoding='utf-8-sig'))
p1=MODULE/'sch/Schematic1/P1.esch2';before=p1.read_bytes()
assert hashlib.sha256(before).hexdigest()==backup['files']['sch/Schematic1/P1.esch2']
groups=[{'net':g['net'],'wire':g['nativeWireId'],'keep':g['keepNetAttributeId'],'remove':[a['id'] for a in g['attributes'] if a['id']!=g['keepNetAttributeId']]} for g in plan['groups']]
assert len(groups)==11 and sum(len(g['remove']) for g in groups)==29
code='const path='+json.dumps(p1.as_posix())+';const groups='+json.dumps(groups)+';'+r'''
const current=await eda.dmt_SelectControl.getCurrentDocumentInfo();
if(current?.uuid)throw Error('Must write with no open document');
const f=await eda.sys_FileSystem.readFileFromFileSystem(path);if(!f)throw Error('Missing native');
const source=await f.text();const ids=new Set(groups.flatMap(g=>g.remove));
const lines=source.split(/(?<=\n)/);const counts={};
const patched=lines.filter(line=>{
 const pos=line.indexOf('||');if(pos<0)return true;
 const h=JSON.parse(line.slice(0,pos));if(h.type!=='ATTR'||!ids.has(h.id))return true;
 const v=JSON.parse(line.slice(pos+2).trim().replace(/\|+$/,''));
 const g=groups.find(g=>g.remove.includes(h.id));
 if(v.key!=='NET'||v.parentId!==g.wire||v.value!==g.net)throw Error('Whitelist mismatch '+h.id);
 counts[h.id]=(counts[h.id]||0)+1;return false;
}).join('');
if(ids.size!==29||[...ids].some(id=>counts[id]!==1))throw Error('Attribute multiplicity mismatch');
for(const g of groups){const l=lines.find(l=>l.startsWith('{"type":"ATTR",')&&l.includes('"id":"'+g.keep+'"'));if(!l)throw Error('Keep missing');const v=JSON.parse(l.slice(l.indexOf('||')+2).trim().replace(/\|+$/,''));if(v.key!=='NET'||v.parentId!==g.wire||v.value!==g.net)throw Error('Keep mismatch');}
if(!await eda.sys_FileSystem.saveFileToFileSystem(path,new Blob([patched],{type:'text/plain'}),undefined,true))throw Error('Write failed');
const readback=await eda.sys_FileSystem.readFileFromFileSystem(path);if(await readback.text()!==patched)throw Error('Readback mismatch');
return {method:'official_FileSystem_read_write',removedNetAttributes:[...ids],groups,removedRecords:29,allOtherBytesUnchanged:true,readback:true};
'''
r=d.h.invoke(code,session=json.loads((HERE/'session.json').read_text())['sessionId'])
ids={a for g in groups for a in g['remove']}
expected=b''.join(l for l in before.splitlines(keepends=True) if not (l.startswith(b'{"type":"ATTR",') and json.loads(l.split(b'||',1)[0])['id'] in ids))
assert p1.read_bytes()==expected
r.update(beforeSHA256=hashlib.sha256(before).hexdigest(),afterSHA256=hashlib.sha256(expected).hexdigest(),independentByteWhitelistVerified=True)
d.out('latest-drc-applied',r)
print(json.dumps(r,ensure_ascii=False))
