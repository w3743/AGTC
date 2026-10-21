"""同步现行连接源与既有页面/封装映射；不执行ERC或PCB操作。"""
import copy,json,pathlib
B=pathlib.Path(__file__).resolve().parents[1]; C=B/'检查'; S=B/'原理图准备'
def read(p): return json.loads(p.read_text(encoding='utf-8-sig'))
def write(p,j): p.write_text(json.dumps(j,ensure_ascii=False,indent=2),encoding='utf-8')
names=['AFE元件连接.json','功率元件连接.json','接口元件连接.json','共享电感矩阵连接.json','共享电感功率级连接.json','共享电感控制保护连接.json','共享电感泄放连接.json','风扇元件连接.json']
sources={}
for n in names:
 for e in read(S/n):
  if e['ref'] in sources: raise ValueError('duplicate '+e['ref'])
  sources[e['ref']]={**e,'sourceFile':n}
old=read(C/'原理图构建清单.json'); originals={e['ref']:e for e in old}
mapping=read(C/'Layout器件映射.json'); maps={e['ref']:e for e in mapping['components']}
# Existing logical service anchors represent the external loom, not mounted connectors.
for e in read(S/'维护线束元件连接.json'):
 if e['ref'] in ('J_CELL_SERVICE','J_BAL_SERVICE'): sources[e['ref']]={**e,'sourceFile':'维护线束元件连接.json'}
changed=set(); entries=[]; outmap=[]
for ref,s in sources.items():
 if ref=='SW_RUN_STORE': continue
 base=originals.get(ref)
 if base is None and ref=='R_MAIN_REQ_PD': base=originals['R_PRE_READY_PD']
 if base is None and ref=='R_DDSG_PD':
  base={**originals['R_PRE_REQ_PD'],'x':2250,'y':1800,'pageGroup':'PWR_PRECHARGE_LATCH'}
 if base is None: raise ValueError('missing template '+ref)
 e=copy.deepcopy(base); e['ref']=ref
 special=e.get('libType')=='2' or ref in ('J_CORE_P1','J_CORE_P2')
 for k in ('mpn','pins','note','sourceFile','value','addIntoBom','addIntoPcb'):
  if k in s: e[k]=s[k]
 if 'value' not in s and s.get('mpn')!=base.get('mpn'):e['value']=s.get('mpn') or ref
 if not special:
  for k in ('libUuid','libraryUuid','libType'):
   if s.get(k): e[k]=s[k]
  e['libType']='3'
 if ref=='RSHUNT':
  p=s['pins'];e['pins']={'1':p['I_BAT'],'2':p['I_LOAD'],'3':p['V_BAT'],'4':p['V_LOAD']}
 if ref in ('J_BAT_HV','J_PACK_HV'):e['pins']['3']=None
 diff=any(e.get(k)!=base.get(k) for k in ('ref','mpn','pins','note','value','libUuid','addIntoBom','addIntoPcb'))
 e['forceRecreate']=not special and (base.get('forceRecreate',False) or ref not in originals or e.get('libUuid')!=base.get('libUuid'))
 if diff: changed.add(e['pageGroup'])
 m=copy.deepcopy(maps.get(ref,maps.get(base['ref'],{}))); m.update(e)
 m['xy']={'x':e['x'],'y':e['y']};m.setdefault('unresolved',[]);m.setdefault('carrierReason','Exact source device; manufacturer package still subject to human pad-map review.')
 if e['forceRecreate']:
  for k in ('candidate','footprintCarrier','nativeFootprint','nativeLibUuid'):m.pop(k,None)
  e.pop('nativeFootprint',None)
  m['carrierReason']='现行源精确器件库；重新创建并读取关联封装。'
 entries.append(e);outmap.append(m)
for e in old:
 if e['ref'] not in sources:changed.add(e['pageGroup'])
mapping['components']=outmap;mapping['generatedBy']='大量精简同步施工.py';mapping['inputs']['sourceConnectionJson']=names
write(C/'原理图构建清单.json',entries);write(C/'Layout器件映射.json',mapping)
write(C/'大量精简施工变化.json',{'changedGroups':sorted(changed),'manifestParts':len(entries),'removedRefs':[e['ref'] for e in old if e['ref'] not in sources and e['ref']!='R_PRE_READY_PD'],'renamedRefs':{'R_PRE_READY_PD':'R_MAIN_REQ_PD'}})
print(json.dumps({'parts':len(entries),'changedGroups':sorted(changed)},ensure_ascii=False))
