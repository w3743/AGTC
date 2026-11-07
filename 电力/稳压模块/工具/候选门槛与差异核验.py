"""Read official candidate export and compare with the existing saved netlist."""
import hashlib
import itertools
import json
from pathlib import Path

root=Path(__file__).resolve().parents[1]
p=root/'检查/候选_独立门槛_LTC2965'
j=json.loads((p/'候选磁盘官方读回.json').read_text(encoding='utf-8'))
for key,name in [('enet','候选.enet'),('bom','候选BOM.csv'),('svg','候选02.svg')]:
    (p/name).write_bytes(j[key].encode('utf-8'))
candidate=json.loads(j['enet'])
base=json.loads((root/'检查/本轮磁盘读回.enet').read_text(encoding='utf-8'))
def normalized(x):
    return {c['props']['Designator']:{'mpn':c['props'].get('Manufacturer Part',''),
      'manufacturer':c['props'].get('Manufacturer',''),
      'supplier':c['props'].get('Supplier',''),
      'supplier_part':c['props'].get('Supplier Part',''),
      'bom':c['props'].get('Add into BOM',''), 'pcb':c['props'].get('Convert to PCB',''),
      'footprint':c['props'].get('FootprintName',''), 'value':c['props'].get('Value',''),
      'pins':{str(v.get('number')):v.get('net') for v in c['pinInfoMap'].values()}}
      for c in x['components'].values()}
a,b=normalized(candidate),normalized(base)
differences={k:{'new':a.get(k),'old':b.get(k)} for k in sorted(a.keys()|b.keys()) if a.get(k)!=b.get(k)}
assert set(differences)=={'R_QP_COL','U_OV_EARLY','R_OV_EARLY_TOP','R_OV_EARLY_MID','R_OV_EARLY_BOT'}, 'unexpected netlist diff'
expected={'1':'BUS_EXT','6':'OV_EARLY_REF','7':'OV_EARLY_H','8':'OV_EARLY_L','9':'OV_EARLY_REF','10':'OV_EARLY_REF','11':'GND_SYS','16':'DUMP_OFF'}
assert all(a['U_OV_EARLY']['pins'].get(k)==v for k,v in expected.items())
assert a['R_QP_COL']['pins']==b['R_QP_COL']['pins']
assert a['R_OV_EARLY_TOP']['pins']=={'1':'OV_EARLY_REF','2':'OV_EARLY_H'}
assert a['R_OV_EARLY_MID']['pins']=={'1':'OV_EARLY_H','2':'OV_EARLY_L'}
assert a['R_OV_EARLY_BOT']['pins']=={'1':'OV_EARLY_L','2':'GND_SYS'}
corners=[]
for rt,rm,rb,vr in itertools.product([150000*.999,150000*1.001],[1000*.99,1000*1.01],[100000*.99,100000*1.01],[2.378,2.426]):
    corners.append({'rise':40*vr*(rb+rm)/(rt+rm+rb),'fall':40*vr*rb/(rt+rm+rb)})
ideal={'nominal_rise_V':40*2.402*101/251,'nominal_fall_V':40*2.402*100/251,
       'rise_min_V':min(x['rise'] for x in corners),'rise_max_V':max(x['rise'] for x in corners),
       'fall_min_V':min(x['fall'] for x in corners),'fall_max_V':max(x['fall'] for x in corners),
       'limitations':'only stated reference and initial resistor extremes; ideal 40x gain, zero comparator offset, pin leakage, TCR, board leakage and dynamics; not guaranteed threshold intervals'}
out={'status':'FAIL_STATIC_AND_UNQUALIFIED_DYNAMIC','count_before':len(b),'count_after':len(a),
     'candidate_drc':j['drc'],'differences':differences,'threshold_screen':ideal,
     'candidate_source_sha256':{x.relative_to(p/'sc-regulator').as_posix():hashlib.sha256(x.read_bytes()).hexdigest() for x in sorted((p/'sc-regulator').rglob('*')) if x.is_file()},
     'outputs_sha256':{name:hashlib.sha256((p/name).read_bytes()).hexdigest() for name in ['候选.enet','候选BOM.csv','候选02.svg','候选02.png','候选重点.png'] if (p/name).exists()}}
(p/'针网差异与哈希.json').write_bytes((json.dumps(out,ensure_ascii=False,indent=2)+'\n').encode('utf-8'))
print(json.dumps({'count_before':len(b),'count_after':len(a),'drc':j['drc'],'threshold_screen':ideal,'diff_refs':list(differences)},ensure_ascii=False))
