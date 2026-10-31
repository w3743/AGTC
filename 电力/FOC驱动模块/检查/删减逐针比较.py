"""Compare retained FOC pins and source identities against pre-delete official ENET."""
import sys,json,collections
from pathlib import Path
before,after=map(lambda x:json.loads(Path(x).read_text(encoding='utf-8-sig')),sys.argv[1:3])
def refs(d):
 x=list(d['components'].values()); out={c['props']['Designator']:c for c in x}
 if len(out)!=len(x):raise ValueError('duplicate designators')
 return out
old,new=refs(before),refs(after)
removed=set('R_DUMP_A R_DUMP_B F_BRAKE J_BRAKE_R J_DUMP_SW Q7 R_DUMP_GATE R_DUMP_GATE_PD U4 U4_C1 U4_C2 J_TEMP R_DUMP_INP_PD R_THERM_TOP R_THERM_BOT R_THERM_SER'.split())
errors=[]
if set(new)!=set(old)-removed:errors.append({'reference_set_missing':sorted((set(old)-removed)-set(new)),'unexpected':sorted(set(new)-(set(old)-removed))})
count=0
for ref in sorted(set(new)&(set(old)-removed)):
 a,b=old[ref],new[ref]
 for key in ['Manufacturer Part','Add into BOM','Convert to PCB','Device','Footprint']:
  if a['props'].get(key)!=b['props'].get(key):errors.append({'ref':ref,'property':key,'old':a['props'].get(key),'new':b['props'].get(key)})
 if set(a['pinInfoMap'])!=set(b['pinInfoMap']):errors.append({'ref':ref,'error':'pin set changed'})
 for pin,info in a['pinInfoMap'].items():
  original=info.get('net') or ''; expected='' if (ref,pin)==('J_MCU_L','24') else original
  got=b['pinInfoMap'].get(pin,{}).get('net') or ''
  count+=1
  if got!=expected:errors.append({'ref':ref,'pin':pin,'expected':expected,'actual':got})
print(json.dumps({'status':'PASS' if not errors else 'FAIL','old_count':len(old),'new_count':len(new),'checked_pins':count,'allowed_change':'J_MCU_L.24 disconnected from removed thermal sense; verify NC marker separately','failures':errors},ensure_ascii=False,indent=2))
sys.exit(bool(errors))
