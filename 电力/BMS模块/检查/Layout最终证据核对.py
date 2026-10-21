"""Read-only delivery gates, then refresh only report evidence hashes."""
from pathlib import Path
import hashlib,json,re,zipfile
P=Path(__file__).resolve().parent;M=P.parent;R=M.parent
load=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
fan=load(M/'仿真/结果/fan_direct_supply_result.json')
binding=fan['topology_binding']
assert binding['mode']=='FORMAL_RELEASE' and binding['topology_status']=='PASS_FORMAL_RELEASE'
assert binding['enet_sha256'].lower()==sha(P/'BMS-9S4P.enet')
assert binding['bom_binding']['sha256'].lower()==sha(M/'原理图准备/BOM.csv')
assert fan['screening_decision']['status']=='PASS_FOR_LAYOUT_WITH_EXPLICIT_BOUNDARIES'
for rel,h in fan['source_sha256'].items():
 assert sha(R/rel)==h.lower(), 'fan model source drift: '+rel
for rel in ['检查/Layout前置审查.md','检查/风扇直供验证.md','原理图准备/Layout交接.md','README.md','项目要求.md','检查/本轮落实与验证.md']:
 p=M/rel;assert p.is_file(),rel
 for target in re.findall(r'\]\(([^)]+)\)',p.read_text(encoding='utf8')):
  if not re.match(r'^(https?:|#)',target):assert (p.parent/target.split('#')[0]).exists(),(rel,target)
count=0
with zipfile.ZipFile(M/'归档/历史与调试.zip') as z:
 for name,key in [('发布归档记录.json','archiveEntries'),('交接文档归档记录.json','records')]:
  for record in load(P/'Layout收尾'/name)[key]:
   assert hashlib.sha256(z.read(record['entry'])).hexdigest()==record['sha256'];count+=1
p=P/'本轮落实与验证.json';d=load(p)
for e in d['evidence']:e['sha256']=sha(M/e['path'])
for rel in ['检查/Layout前置审查.md','检查/风扇直供验证.md','原理图准备/Layout交接.md']:
 d['evidence'].append({'path':rel,'sha256':sha(M/rel)})
d['finalReadOnlyGate']={'formalModelBinding':'PASS','currentFanSourceHashes':'PASS','handoffLinks':'PASS','archiveEntriesReadbackSHA256':count,'stoppedAt':'human Layout handoff; no PCB created'}
p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
print(json.dumps(d['finalReadOnlyGate'],ensure_ascii=False))
