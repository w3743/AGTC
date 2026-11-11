"""Read-only source checks and reproducible unimplemented FOC candidate contract."""
import hashlib,json
from pathlib import Path
HERE=Path(__file__).resolve().parent
POWER=HERE.parents[1]
F=POWER/'FOC驱动模块/eda工程文件/导出/当前交付/FOC_G431_P1.enet'
B=POWER/'BMS模块/检查/当前源/跨板并料_当前.enet'
DEVICE=POWER/'元件封装库/共享仓库/库/原生片段/DEVICE/a604a113e9894ae7ab14b02ff5f80e3bbba6993c6d2883d91aae555cfbf8b802.jsonl'
OUT=HERE/'FOC保持与MOS通知合同_未落图.json'
sha=lambda x:hashlib.sha256(x.read_bytes()).hexdigest()
def load(p): return json.loads(p.read_text(encoding='utf-8-sig'))
def byref(p): return {x['props']['Designator']:x for x in load(p)['components'].values()}
f,b=byref(F),byref(B)
for pin in ('9','16'): assert f['J_MCU_L']['pinInfoMap'][pin]['net']==''
expect={'Q_BUCK_RELEASE':('BC847C','C9900010419'),'R_BUCK_VALID_BASE':('RC0603FR-074K7L','C99782'),'R_BUCK_VALID_PD':('RC0603FR-07100KL','C14675'),'R_HA_5V_SER':('RC0603FR-0710KL','C98220')}
for ref,(mpn,sku) in expect.items():
 p=f[ref]['props']; assert p['Manufacturer Part']==mpn and p['Supplier Part']==sku
q=b['Q_DFETOFF']; assert q['props']['Manufacturer']=='UMW(友台半导体)' and q['props']['Manufacturer Part']=='AO3400A' and q['props']['Supplier Part']=='C347475'
assert q['props']['Device']=='8ae7cb4a29cf630a' and [(q['pinInfoMap'][str(i)]['name']) for i in (1,2,3)]==['G','S','D']
rows=DEVICE.read_text(encoding='utf-8').splitlines()
head=json.loads(rows[0].split('||')[1].removesuffix('|')); meta=json.loads(rows[1].split('||')[1].removesuffix('|'))
assert head['uuid']=='8ae7cb4a29cf630a' and meta['attributes']['Supplier Part']=='C347475'
assert meta['attributes']['Symbol']=='cc29dd40af12c275' and meta['attributes']['Footprint']=='63a89eb4f6592f79'
parts=[
 {'ref':'Q_HOLD','mpn':'Nexperia BC847C','supplier':'C9900010419','pins':{'1':'FOC_HOLD_BASE','2':'GND_SYS','3':'BUCK_CLAMP_BASE'}},
 {'ref':'R_HOLD_BASE','mpn':'FH RS-03K4701FT','supplier':'C116693','pins':{'1':'FOC_HOLD_GPIO','2':'FOC_HOLD_BASE'}},
 {'ref':'R_HOLD_PD','mpn':'Tyohm RMC0603100K1%N','supplier':'C269699','pins':{'1':'FOC_HOLD_BASE','2':'GND_SYS'}},
 {'ref':'Q_LPV_MOS','mpn':'UMW AO3400A','supplier':'C347475','device':'8ae7cb4a29cf630a','symbol':'cc29dd40af12c275','footprint':'63a89eb4f6592f79','pins':{'1':'LOAD_POWER_VALID','2':'GND_SYS','3':'LPV_FAIL_MCU'}},
 {'ref':'R_LPV_GATE_PD','mpn':'Tyohm RMC0603100K1%N','supplier':'C269699','pins':{'1':'LOAD_POWER_VALID','2':'GND_SYS'}},
 {'ref':'R_LPV_PULLUP','mpn':'FH RS-03K1002FT','supplier':'C115324','pins':{'1':'MCU_3V3','2':'LPV_FAIL_MCU'}}]
result={'status':'NOT_IMPLEMENTED_CANDIDATE_ONLY','source_sha256':{'foc_enet':sha(F),'bms_reference_enet':sha(B),'shared_device':sha(DEVICE)},'parts_per_board':parts,'per_board_other_connections':{'J_MCU_L.9':'FOC_HOLD_GPIO / selected WeAct Normal V1.0 QFN48 P1-9 PC6; remove NC','J_MCU_L.16':'LPV_FAIL_MCU / selected WeAct Normal V1.0 QFN48 P1-16 PC10; remove NC'},'population':'A and B each fit all six added parts; A only existing R_LOAD_VALID_BIAS=1k, B DNP; confirm other board contracts and physical orientation.',
 'cost':'Two boards need 2 BC847C + 2 AO3400A + 2 FH 4.7k + 4 Tyohm 100k + 2 FH 10k. Historical price subtotal for UMW AO3400A, 100k and 10k only = 2*0.2054 + 4*0.0241 + 2*0.0125 = CNY 0.5322. BC847C, 4.7k exact comparable quote, MOQ, shipping and assembly are excluded. Not vehicle delivered cost or a saving claim.',
 'constraints':['Existing LPV optical/output population and protective circuits retained.','LPV healthy=PC10 low; failure=high; read current level before ARM and detect rising edge.','User selected WeAct Normal V1.0 QFN48; verify physical top-view orientation and supply.','AO3400A library META RDS 17mOhm is not 48mOhm max guarantee; use UMW datasheet conditions.','LPV optical VCE/CTR, physical-main-to-LPV timing, actual gate, input and power transition and shared wiring require system and bench validation.','Candidate only; no PCB, firmware or active native schematic modified.']}
OUT.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print('Source identities match; unimplemented candidate contract written')
