"""Verify an officially exported FOC candidate ENET against the live delivery ENET.

Read-only on both ENETs and project/PCB files; writes a JSON evidence report only.
It does not open, edit, generate, or save an EasyEDA native project.
"""
import argparse, hashlib, json
from pathlib import Path

HERE=Path(__file__).resolve().parent
FOC=HERE.parent
BASE=FOC/'eda工程文件/导出/当前交付/FOC_G431_P1.enet'
CONTRACT=HERE/'FOC保持与MOS通知合同_未落图.json'
OUT=HERE/'FOC候选交付针网差异.json'

def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def load(p): return json.loads(p.read_text(encoding='utf-8-sig'))
def mapping(path):
 data=load(path); refs={}
 for c in data['components'].values():
  p=c['props']; ref=p['Designator']
  if ref in refs: raise ValueError('duplicate reference '+ref)
  refs[ref]={'pins':{str(k):v.get('net') or '' for k,v in c['pinInfoMap'].items()},
             'mpn':p.get('Manufacturer Part',''), 'supplier':p.get('Supplier Part',''),
             'manufacturer':p.get('Manufacturer','')}
 return refs

def main():
 ap=argparse.ArgumentParser(); ap.add_argument('candidate_enet',type=Path)
 ap.add_argument('--source-pcb',type=Path); ap.add_argument('--candidate-pcb',type=Path)
 ap.add_argument('--report',type=Path,default=OUT)
 a=ap.parse_args(); contract=load(CONTRACT)
 if sha(BASE)!=contract['source_sha256']['foc_enet']: raise ValueError('baseline ENET drift; review before re-baselining')
 old,new=mapping(BASE),mapping(a.candidate_enet)
 expected={x['ref']:x for x in contract['parts_per_board']}
 issues=[]
 if set(new)-set(old)!=set(expected): issues.append({'added_mismatch':sorted(set(new)-set(old)),'expected':sorted(expected)})
 if set(old)-set(new): issues.append({'removed':sorted(set(old)-set(new))})
 for ref in sorted(set(old)&set(new)):
  before,after=old[ref],new[ref]
  if (before['mpn'],before['supplier'],before['manufacturer'])!=(after['mpn'],after['supplier'],after['manufacturer']):
   issues.append({'identity_changed':ref,'before':before,'after':after})
  p=dict(before['pins'])
  if ref=='J_MCU_L': p['9']='FOC_HOLD_GPIO'; p['16']='LPV_FAIL_MCU'
  if p!=after['pins']: issues.append({'pin_difference':ref,'expected':p,'actual':after['pins']})
 for ref,want in expected.items():
  if ref not in new: continue
  got=new[ref]
  if got['pins']!=want['pins']: issues.append({'new_pins':ref,'expected':want['pins'],'actual':got['pins']})
  if got['supplier']!=want['supplier']: issues.append({'new_supplier':ref,'expected':want['supplier'],'actual':got['supplier']})
  if ref=='Q_LPV_MOS' and ('UMW' not in got['manufacturer'] or got['mpn']!='AO3400A'):
   issues.append({'wrong_mos_identity':got})
 pcb=None
 if (a.source_pcb is None)!=(a.candidate_pcb is None): raise ValueError('supply both PCB paths or neither')
 if a.source_pcb:
  pcb={'source_sha256':sha(a.source_pcb),'candidate_sha256':sha(a.candidate_pcb)}
  if pcb['source_sha256']!=pcb['candidate_sha256']: issues.append({'pcb_changed':pcb})
 result={'status':'PASS_STATIC_NETLIST_ONLY' if not issues else 'FAIL',
         'baseline_sha256':sha(BASE),'candidate_sha256':sha(a.candidate_enet),
         'baseline_components':len(old),'candidate_components':len(new),'pcb_comparison':pcb,
         'issues':issues,'scope':'Official ENET pin/identity difference and optional PCB hash only. Does not prove source-save/reopen, explicit NC removal, DRC, PDF readability, physical/vehicle acceptance, or population markings.'}
 a.report.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
 print(result['status'],len(issues),'issues')
 if issues: raise SystemExit(1)

if __name__=='__main__': main()
