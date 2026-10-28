"""官方API保存与导出当前工程；不执行ERC或PCB操作。"""
import json,sys,hashlib,base64
from pathlib import Path
from 原生更新 import invoke
B=Path(__file__).resolve().parents[1];C=B/'检查'
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def write(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2),encoding='utf-8')
pages=read(C/'原理图页面映射.json')
mode=sys.argv[2]
if mode=='remove-legacy':
 evidence=[]
 for name,uid in list(pages.items()):
  if not (name.startswith('AFE_LTC') or name=='AFE_LOCAL_UV'):continue
  ok=invoke('return await eda.dmt_Schematic.deleteSchematicPage('+json.dumps(uid)+');')
  if not ok:raise RuntimeError('delete failed '+name)
  evidence.append({'name':name,'uuid':uid,'deleted':ok});del pages[name]
  write(C/'原理图页面映射.json',pages)
 write(C/'旧均衡页面退出证据.json',evidence)
 print('removed legacy pages',len(evidence))
elif mode=='merge':
 base=read(C/'原理图构建清单.json')
 base=[e for e in base if not e['pageGroup'].startswith('BAL_')]
 new=read(C/'共享电感施工输入.json')
 refs={e['ref']:e for e in base+new}
 if len(refs)!=len(base)+len(new):raise RuntimeError('duplicate ref')
 combined=list(refs.values());write(C/'原理图构建清单.json',combined)
 old=read(C/'Layout器件映射.json')
 oldlist=old['components'] if isinstance(old,dict) else old
 oldrefs={e['ref']:e for e in oldlist}
 mapped=[]
 for e in combined:
  v={**oldrefs.get(e['ref'],{}),**e}
  v.setdefault('xy',{'x':v.get('x',0),'y':v.get('y',0)})
  mapped.append(v)
 if isinstance(old,dict):old['components']=mapped
 else:old=mapped
 write(C/'Layout器件映射.json',old)
 switches=read(C/'开关构建清单.json')
 write(C/'当前施工输入.json',combined+switches)
 print('merged manifest',len(combined),'switch symbol units',len(switches))
elif mode=='read-current':
 out=[]
 for name,uid in pages.items():
  data=invoke('await eda.dmt_EditorControl.openDocument('+json.dumps(uid)+');const cs=await eda.sch_PrimitiveComponent.getAll();return cs.filter(c=>c.getState_ComponentType()==="part").map(c=>({ref:c.getState_Designator(),id:c.getState_PrimitiveId(),name:c.getState_Name(),mpn:c.getState_ManufacturerId(),footprint:c.getState_Footprint(),pcb:c.getState_AddIntoPcb(),bom:c.getState_AddIntoBom()}));')
  out.append({'page':name,'uuid':uid,'parts':data})
 write(C/'当前原生器件清单.json',out)
 print('pages',len(out),'parts',sum(len(p['parts']) for p in out))
elif mode=='export':
 project='f3b5f17f6b76f52ea94c3da976e24511a61863492a292d84aa98bcb199920378'
 # 导出官方工程格式；它与工作数据库.eprj2分开，不猜测文件内容。
 target=B/'eda工程文件'/'BMS-9S4P.epro2'
 # 客户端BETA文件系统API拒绝此工程路径。使用官方导出File的数据，
 # 经CLI传回已授权的共享工作目录；不修改客户端路径权限。
 code='const f=await eda.sys_FileManager.getProjectFileByProjectUuid('+json.dumps(project)+',"BMS-9S4P",undefined,"epro2");if(!f)throw Error("export missing");const b=new Uint8Array(await f.arrayBuffer());let s="";for(let i=0;i<b.length;i+=32768)s+=String.fromCharCode(...b.subarray(i,i+32768));return {name:f.name,size:f.size,type:f.type,base64:btoa(s)};'
 out=invoke(code)
 exported=base64.b64decode(out.pop('base64'),validate=True)
 if len(exported)!=out['size']:raise RuntimeError('official export payload size mismatch')
 target.write_bytes(exported)
 out['saved']=True
 data=target.read_bytes()
 if len(data)!=out['size']:raise RuntimeError('export size mismatch')
 out.update(path=str(target),sha256=hashlib.sha256(data).hexdigest(),magic=data[:16].hex(),date='2026-10-13',method='official getProjectFileByProjectUuid File payload transferred through CLI and written in shared workspace',clientFilesystemLimitation='saveFileToFileSystem returned 路径不允许; client path permissions unchanged')
 local=B/'eda工程文件'/'BMS-9S4P.eprj2'
 if local.exists():
  ld=local.read_bytes();out['workingProject']={'path':str(local),'size':len(ld),'sha256':hashlib.sha256(ld).hexdigest(),'note':'Local engineering database retained separately from official portable export.'}
 write(C/'当前官方工程导出.json',out)
 print(json.dumps(out,ensure_ascii=False))
elif mode=='retire-stale-local':
 current=read(C/'当前官方工程导出.json');target=Path(current['path'])
 if hashlib.sha256(target.read_bytes()).hexdigest()!=current['sha256']:raise RuntimeError('current export mismatch')
 old=B/'eda工程文件'/'BMS-9S4P.eprj2'
 if old.exists():
  import zipfile
  original=old.read_bytes();sha=hashlib.sha256(original).hexdigest()
  with zipfile.ZipFile(B/'归档/历史与调试.zip') as z:
   archived=z.read('共享电感替换前/BMS-9S4P.eprj2')
  if hashlib.sha256(archived).hexdigest()!=sha:raise RuntimeError('stale local has changed; preserve separately')
  dest=B/'归档/原始材料/BMS-9S4P.eprj2'
  if not dest.resolve().is_relative_to(B.resolve()):raise RuntimeError('target outside module')
  if dest.exists():raise RuntimeError('target exists')
  dest.parent.mkdir(parents=True,exist_ok=True)
  old.rename(dest)
  if hashlib.sha256(dest.read_bytes()).hexdigest()!=sha:raise RuntimeError('move readback mismatch')
  write(C/'旧本地工程退出.json',{'old':str(old),'recoverable':str(dest),'sha256':sha,'current':str(target),'date':'2026-10-12'})
  print('stale local moved recoverably',str(dest))
 else:print('no stale local entry')
else:raise ValueError(mode)
