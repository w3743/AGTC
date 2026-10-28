"""Publish official saved NC change after verified recoverable archive."""
from pathlib import Path
import datetime, hashlib, json, shutil, zipfile
P=Path(__file__).resolve().parent; M=P.parent; O=P/'剩余验证'; W=P/'tmp/remaining_validation_work'; N=M/'eda工程文件'
load=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
sha=lambda b:hashlib.sha256(b).hexdigest()
initial=load(O/'初始原生SHA256.json')
for rel,digest in initial.items():assert sha((N/rel).read_bytes())==digest,rel
assert load(O/'保存重开器件核对.json')['drc']==[{'type':'warn','count':8}]
assert (O/'BMS-9S4P.epro2').is_file() and (O/'官方导出SHA256.json').is_file()
old=load(P/'BMS-9S4P.enet'); new=load(O/'final_reopened.enet')
def pinmap(net):return {(v['props']['Designator'],p):d.get('net','') for v in net['components'].values() for p,d in v['pinInfoMap'].items()}
a,b=pinmap(old),pinmap(new);assert a.keys()==b.keys()
diff=[(k,a[k],b[k]) for k in a if a[k]!=b[k]]
assert set((k,old,new) for k,old,new in diff)=={(('U_SEQ','6'),'MAIN_HOLD_N',''),(('U_SEQ','8'),'PRE_USED_N','')},diff
paths=list(N.rglob('*'))+[P/'当前施工输入.json',M/'原理图准备/功率元件连接.json',P/'Layout器件映射.json',P/'BMS-9S4P.enet',P/'当前原理图导航审阅.pdf',P/'当前原理图导航审阅.json',P/'本轮落实与验证.json',P/'本轮落实与验证.md',P/'DRC逐项核对.json',P/'DRC逐项核对.md',M/'README.md',M/'TODO.md',M/'项目要求.md',P/'当前设计状态.md']
files=[p for p in paths if p.is_file()];key=datetime.datetime.now().strftime('%Y%m%d_%H%M%S')+'_未用输出NC发布前'
records=[]; archive=M/'归档/历史与调试.zip'
with zipfile.ZipFile(archive,'a',compression=zipfile.ZIP_DEFLATED) as z:
 for p in files:
  raw=p.read_bytes();entry=key+'/'+p.relative_to(M).as_posix();z.writestr(entry,raw)
  records.append({'path':p.relative_to(M).as_posix(),'entry':entry,'sha256':sha(raw)})
with zipfile.ZipFile(archive) as z:
 for r,p in zip(records,files):assert sha(z.read(r['entry']))==r['sha256']==sha(p.read_bytes())
for rel in initial:
 if rel.endswith('P1.esch2'):shutil.copyfile(W/rel,N/rel)
shutil.copyfile(O/'BMS-9S4P.epro2',N/'BMS-9S4P.epro2')
for path,container in [(P/'当前施工输入.json',None),(M/'原理图准备/功率元件连接.json',None),(P/'Layout器件映射.json','components')]:
 data=load(path);rows=data if container is None else data[container];row=next(x for x in rows if x['ref']=='U_SEQ')
 assert row['pins']['6']=='MAIN_HOLD_N' and row['pins']['8']=='PRE_USED_N'
 row['pins']['6']=None;row['pins']['8']=None
 if 'note' in row:row['note']+=' 未使用反相输出6/8脚移除有名短线和端口，官方API设置NoConnected=true；不影响Q1/Q2和逻辑功能。'
 if container:data['inputs']={'independentExpected':'当前施工输入.json','officialReopenedEnet':'剩余验证/final_reopened.enet'}
 path.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
published=[{'file':p.relative_to(M).as_posix(),'sha256':sha(p.read_bytes())} for p in N.rglob('*') if p.is_file()]
(O/'原生发布校验.json').write_text(json.dumps(published,ensure_ascii=False,indent=2),encoding='utf8')
(O/'发布归档记录.json').write_text(json.dumps({'archive':'归档/历史与调试.zip','readbackSHA256Verified':True,'before':records,'pinChanges':diff,'published':published,'scope':'仅U_SEQ未用反相输出6/8针NC、相应源连接与官方再导出；BOM和料号不变'},ensure_ascii=False,indent=2),encoding='utf8')
print(json.dumps({'pinChanges':diff,'archiveReadback':'PASS','nativeFiles':len(published)},ensure_ascii=False))
