"""生成采用方案清单；不覆盖原生导出BOM，不伪造原理图/DRC。"""
import csv, hashlib, json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
current_contract = ROOT / '原理图准备/最终逐针连接合同.json'
if current_contract.exists() and json.loads(current_contract.read_text(encoding='utf-8-sig')).get('interface_only_revision'):
    raise SystemExit('当前已采用板外仅接口合同；本脚本仅重建2026-10-29的171条历史选型，不能覆盖当前选型BOM。请维护原理图准备/选型BOM.csv。')
BASE = ROOT / '原理图准备/BOM.csv'
rows = list(csv.DictReader(BASE.open(encoding='utf-8-sig')))
original = {r['Reference']: r.copy() for r in rows}
assert len(original) == len(rows) == 169
retired = [f'U_BAL{i}' for i in range(1,9)] + ['Q_BUS_P','D_QP_GS_ZENER','R_12V_PRELOAD'] + [f'C_BAL{i}' for i in range(6,9)] + [f'C_T{i}_FILTER' for i in range(4,9)] + ['C_TEST_FB']
adopted = {k: v.copy() for k,v in original.items() if k not in retired}
for r in adopted.values():
    r['Action'] = 'RETAIN'
    r['Selection status'] = 'ADOPTED_SELECTION_NOT_NATIVE_EXPORT'

def change(ref, nets=None, note='', mpn=None, supplier=None, value=None, footprint=None):
    r = adopted[ref]
    r['Action'] = 'REWIRE' if mpn is None else 'REPLACE_SAME_QUANTITY'
    if nets is not None: r['Pin nets'] = nets
    if mpn is not None: r['Manufacturer part'] = mpn
    if supplier is not None: r['Supplier part'] = supplier
    if value is not None: r['Value or candidate'] = value
    if footprint is not None: r['Footprint'] = footprint
    r['Requirements and notes'] = note

def add(ref, value, mpn, code, footprint, nets, reason, manufacturer=''):
    assert ref not in original and ref not in adopted
    r = {k: '' for k in rows[0]}
    r.update({'Scope':'ON_BOARD','Reference':ref,'Quantity':'1','Unit':'pcs',
              'Value or candidate':value,'Manufacturer':manufacturer,'Manufacturer part':mpn,
              'Supplier':'LCSC','Supplier part':code,'Footprint':footprint,'Convert to PCB':'Yes',
              'Procurement status':'SELECTED_PRICE_NOT_CONFIRMED','Requirements and notes':reason,
              'Pin nets':nets,'Source':'2026-10-29 adopted selection; not native export',
              'Action':'ADD','Selection status':'ADOPTED_SELECTION_NOT_NATIVE_EXPORT'})
    adopted[ref] = r

for i in range(1,9):
    hi = f'T{i}' if i < 8 else 'SC_PLUS'
    lo = f'T{i-1}' if i > 1 else 'SC_MINUS'
    change(f'R_BAL{i}',f'1={hi}; 2={lo}','120Ω常接慢泄放；不是20A主充电流均衡器')
for i in range(1,8): change(f'R_T{i}_TEST',f'1=MUX_SEL{i}; 2=GND_SYS','移用47k为选择脚下拉')
change('R_QP_GS','1=MUX_SEL8; 2=GND_SYS','移用47k；不得同时给PA9下拉')
change('R_TEST_HI','1=BUS_ADC; 2=GND_SYS','移用MCU侧下拉；BUS实际比例约61')
change('R_TEST_LO','1=IMON; 2=GND_SYS','移用MCU侧下拉')
change('C_T1_FILTER','1=SC_ADC_A; 2=GND_SYS','移用至MUX公共输出')
change('C_T2_FILTER','1=SC_ADC_B; 2=GND_SYS','移用至MUX公共输出')
change('C_T3_FILTER','1=BUS_OV_DIV; 2=SC_MINUS','复用为硬件BUS过压滤波')
for i in range(1,6):
    rail = '3V3_CORE' if i < 5 else '12V_DRV'
    change(f'C_BAL{i}',f'1={rail}; 2=GND_SYS','依次用于MUX1/MUX2/SIGNAL_ISO/G07/LTC7001本地去耦',
           'CGA3E2X7R1H104KT0Y0S','C342998','100nF 50V X7R','C0603')
for ref in ['C_CAN1','C_NTC_DUMP','C_NTC_MAIN','C_NTC_SC']:
    change(ref,note='修正旧MPN/商城代码身份不一致，统一现有TDK 100nF',
           mpn='CGA3E2X7R1H104KT0Y0S',supplier='C342998',value='100nF 50V X7R',footprint='C0603')
for ref in ['C_BUS_FILM1','C_BUS_FILM2','C_BUS_HF','C_DRV_VDRAIN','Q_MAIN_H','U1']:
    change(ref,original[ref]['Pin nets'].replace('BUS_BR','BUS_EXT'),'删除旧P管后BUS_BR与BUS_EXT直连')
change('C_BUS_FILM3','1=SC_FUSED; 2=GND_SYS','移用至隔离桥侧，双向夹位非极性电容')
for ref in ['C_SC_BULK','C_SC_BULK2']:
    change(ref,'1=SC_ISO_STORAGE; 2=SC_MINUS','35V极性电容移至隔离储能侧')
change('C_SC_HF','1=SC_FUSED; 2=GND_SYS','隔离桥侧高耐压非极性电容')
change('F_SC','1=SC_PLUS; 2=SC_ISO_STORAGE','AMG-80必须为70VDC/2.5kA版本，非旧32V库存')
change('F_BUS',note='AMG-80必须为70VDC/2.5kA版本，非旧32V库存')
change('J_SC_POWER','1=SC_ISO_STORAGE; 2=SC_MINUS','M5螺栓功率接口，几何/紧固由人类PCB及结构设计落实')
for ref in ['J_BUS_POWER','J_DUMP']:
    change(ref,note='采用M5螺栓功率接口；这是板级接口要求，非另加未定电子件')
change('Q_PMOS_CTRL','1=DUMP_HW_BASE; 2=GND_SYS; 3=DUMP_OFF','旧NPN复用作硬OV开启dump')
change('R_PB6_BASE','1=BUS_OV_OUT; 2=DUMP_HW_BASE','旧10k复用硬OV NPN基极')
change('R_QP_BE','1=DUMP_HW_BASE; 2=GND_SYS','旧100k复用硬OV NPN下拉')
change('R_QP_COL','1=12V_DRV; 2=DUMP_OFF','旧10k移用UCC IN−上拉；不能再作ISO TGUP')
change('R_DUMP_IN_SER','1=DUMP_PWM; 2=DUMP_PWM_LIMIT','保留100Ω串接，后接新增10k基极电阻')
change('R_DUMP_IN_PD','1=DUMP_SW_BASE; 2=GND_SYS','旧10k移为软件NPN基极下拉')
change('U_DUMP_DRV','1=12V_DRV; 2=UCC_OUT_H; 3=UCC_OUT_L; 4=GND_SYS; 5=DUMP_OFF; 6=12V_DRV','IN+固定12V，IN−低有效；硬件/软件NPN并联开启')
change('U_SC_OV','1=BUS_OV_OUT; 2=SC_MINUS; 3=BUS_OV_DIV; 4=SC_OV_DIV; 5=12V_DRV; 6=DRV_FAULT_N','TLV6700 A=BUS硬OV，B=SC总压OV，原供电与地不改')
change('R_PB5_PERMIT','1=3V3_CORE; 2=DRV_FAULT_N','1k改作实际许可线拉高；PB5释放给MUX')
change('R_BUS_HI','1=BUS_HI_MID; 2=BUS_ADC_SOURCE','BUS原分压在TMUX源侧')
change('R_BUS_LO','1=BUS_ADC_SOURCE; 2=GND_SYS','BUS原分压在TMUX源侧')
change('C_BUS_ADC','1=BUS_ADC_SOURCE; 2=GND_SYS','保留源侧1nF')
change('R_DRV_SO_SER','1=DRV_SO; 2=IMON_SOURCE','CSA输出22Ω后进入TMUX')
change('C_DRV_SO','1=IMON_SOURCE; 2=GND_SYS','保留源侧100pF')
change('R_DUMP_EXT','1=BUS_EXT; 2=DUMP_SW','用户承担电阻功率/热实现；本设计仅冻结阻值、容差、数量；无温熔',value='0.82R ±5%, single resistor; thermal rating user scope')
change('COREBOARD_WEACT',note='采用WeAct Normal QFN48 STM32G431CBU6 V1.0；5V输入；USB CC跳桥不得占PA9/PA10/PB4；CAN期间不接USB数据',value='WeAct Normal QFN48 STM32G431CBU6 V1.0')
change('U_AUX1',note='固定海凌科Hi-Link厂家URB4812YMD-10WR3，18–75V输入，12V/833mA；不同厂家同名不可混用')
change('PSU5_EXT','1=GND_SYS; 2=BUS_EXT; 3=5V_EXT; 4=NC; 5=GND_SYS; 6=NC','固定海凌科Hi-Link URB4805YMD-10WR3；按官方单路引脚表1=−Vin/2=+Vin，纠正旧合同输入反接；4=NP/6=CTRL悬空开启')
adopted['PSU5_EXT']['Scope']='OFF_BOARD_EXTERNAL'
adopted['PSU5_EXT'].update({'Manufacturer':'HI-LINK(海凌科)','Supplier part':'C5365326','Convert to PCB':'No'})
adopted['R_DUMP_EXT']['Convert to PCB']='No'

for bank in (1,2):
    n = (bank-1)*4
    sig = '; '.join(f'S{j}=AD_T{n+j}; D{j}=SC_ADC_{"A" if bank==1 else "B"}; SEL{j}=MUX_SEL{n+j}' for j in range(1,5))
    add(f'U_CELL_MUX{bank}','4xSPST powered-off protection','TMUX1511PWR','C2866750','TSSOP-14',sig+'; VDD=3V3_CORE; GND=GND_SYS','四通道共用ADC，断电不从累计抽头反灌MCU','Texas Instruments')
add('U_SIGNAL_ISO','4xSPST powered-off protection','TMUX1511PWR','C2866750','TSSOP-14','S1=BUS_ADC_SOURCE; D1=BUS_ADC; S2=IMON_SOURCE; D2=IMON; SEL1/2=3V3_CORE; SEL3/4=GND_SYS; S3/4=GND_SYS; D3/4=NC; VDD=3V3_CORE; GND=GND_SYS','BUS及CSA掉电输入隔离','Texas Instruments')
add('U_GATE_OD','single non-inverting open drain','SN74LVC1G07DCKR','C7830','SC70-5','A=PA9_TIM1_CH2; Y=DRV_FAULT_N; VCC=3V3_CORE; GND=GND_SYS','TIM1 Break清MOE后许可保持低；无需外置FF','Texas Instruments')
add('U_ISO_DRV','static high-side NMOS driver','LTC7001IMSE#PBF','C581062','MSOP-10-EP','1=12V_DRV; 2=GND_SYS; 3=GND_SYS; 4=DRV_FAULT_N; 5=GND_SYS; 6=ISO_G; 7=ISO_TGUP; 8=ISO_MID; 9=ISO_BST; 10=NC; 11/EP=GND_SYS','双向隔离管驱动；VCCUV pin2接地，OVLO pin5接地；12V合格后至少等1秒首次ARM','Analog Devices')
for i,drain in [(1,'SC_ISO_STORAGE'),(2,'SC_FUSED')]:
    add(f'Q_SC_ISO{i}','100V N-MOS','HYG022N10NS1TA','C2925050','TOLL','1=ISO_G; 2-8=ISO_MID; 9/tab='+drain,'共源背靠背，阻断两方向电流；复用BMS精确型号；TA封装为TOLL','Huayi')
for ref,a,k in [('D_SC_LO','GND_SYS','SC_FUSED'),('D_SC_HI','SC_FUSED','BUS_EXT'),('D_DUMP_FW','DUMP_SW','BUS_EXT')]:
    add(ref,'100V 30A single Schottky','STPS30100ST','C172485','TO-220AB',f'1/3={a}; 2/tab={k}','主桥双向续流或制动电阻/线束电感续流；单二极管器件','STMicroelectronics')
add('Q_DUMP_SW','MMBT5551 NPN','MMBT5551','C2972766','SOT-23','1=DUMP_SW_BASE; 2=GND_SYS; 3=DUMP_OFF','MCU独立请求dump；避免12V接MCU','JKSEMI(金开盛)')
add('C_ISO_BST','1uF 100V X7R','C3225X7R2A105K200AA','C2167578','C1210','1=ISO_BST; 2=ISO_MID','LTC7001浮动驱动储能；复用BUS/SC既有型号，12V偏压及温度后有效C≥0.5µF','TDK')
resistors = [
('R_ADC_A_PD','10k 0.1%','PTFR0805B10K0N9','C328436','SC_ADC_A','GND_SYS'),
('R_ADC_B_PD','10k 0.1%','PTFR0805B10K0N9','C328436','SC_ADC_B','GND_SYS'),
('R_ARM_PD','100k 1%','RMC0603100K1%N','C269699','PA9_TIM1_CH2','GND_SYS'),
('R_ISO_GS','1M 1%','RC0603FR-071ML','C105578','ISO_G','ISO_MID'),
('R_ISO_TGUP','10k 0.1%','HoAR0603-1/10W-10KR-0.1%-TCR25','C5123581','ISO_TGUP','ISO_G'),
('R_DUMP_HW_PU','10k 0.1%','HoAR0603-1/10W-10KR-0.1%-TCR25','C5123581','12V_DRV','BUS_OV_OUT'),
('R_DUMP_SW_BASE','10k 0.1%','HoAR0603-1/10W-10KR-0.1%-TCR25','C5123581','DUMP_PWM_LIMIT','DUMP_SW_BASE'),
('R_BUS_OV_TOP1','255k 0.1%','PTFR0805B255KP9','C2849126','BUS_EXT','BUS_OV_MID'),
('R_BUS_OV_TOP2','255k 0.1%','PTFR0805B255KP9','C2849126','BUS_OV_MID','BUS_OV_DIV'),
('R_BUS_OV_BOT','5.11k 0.1%','RT0603BRD075K11L','C705788','BUS_OV_DIV','SC_MINUS')]
for ref,value,mpn,code,a,b in resistors:
    add(ref,value,mpn,code,'R0805' if mpn.startswith('PTFR0805') else 'R0603',f'1={a}; 2={b}','ADC负载/复位默认态/隔离驱动/制动硬OV必要外围')

# 核心板接口按采用的Normal QFN48版本；物理排母针号在落图时绑定厂家图。
gpio = {'PA8':'TIM1_CH1 PWM_H','PB13':'TIM1_CH1N PWM_L','PA9':'TIM1_CH2 ARM',
        'PB12':'TIM1_BKIN FAULT','PA10':'FAULT input','PA6':'COMP1_OUT OD','PA2':'COMP2_OUT OD',
        'PA0':'COMP3 IMON','PB1':'COMP1/ADC IMON','PA7':'COMP2 IMON','PB0':'BUS_ADC',
        'PB14':'SC_ADC_A ADC1_IN5','PB15':'SC_ADC_B ADC2_IN15','PB11':'NTC_MAIN',
        'PC4':'NTC_SC','PA3':'NTC_DUMP','PB7':'DUMP_PWM',
        'PC6':'MUX_SEL1','PA15':'MUX_SEL2','PC10':'MUX_SEL3','PC11':'MUX_SEL4',
        'PB3':'MUX_SEL5','PB4':'MUX_SEL6','PB5':'MUX_SEL7','PB9':'MUX_SEL8',
        'PA11':'CAN_RX','PA12':'CAN_TX','PA13':'SWDIO','PA14':'SWCLK'}
for ref in ['J_CORE_P1','J_CORE_P2']:
    change(ref,note='原ENET针网作废，落图须按最终GPIO合同与WeAct Normal V1.0官方排针号重连',nets='SEE 最终选型清单.json gpio_contract; DO NOT USE OLD NATIVE PIN NETS')

# 最新用户授权外围国产替代；只应用已核对的精确替代表，不改主器件或网络。
domestic_path=ROOT/'检查/国产外围替代核对.json'
if domestic_path.exists():
    domestic=json.loads(domestic_path.read_text(encoding='utf-8-sig'))
    for item in domestic.get('adopted_substitutions',[]):
        for ref in item['references']:
            r=adopted[ref]
            old_action=r['Action']
            change(ref,note=item['reason'],mpn=item['new_mpn'],supplier=item['new_supplier_part'],
                   value=item['new_value'],footprint=item['new_footprint'])
            r['Manufacturer']=item['new_manufacturer']
            if old_action=='ADD': r['Action']='ADD'
contract_path=ROOT/'原理图准备/最终逐针连接合同.json'
standard_path=ROOT/'检查/器件属性标准化计划.json'
if standard_path.exists():
    for item in json.loads(standard_path.read_text(encoding='utf-8'))['components']:
        r=adopted[item['ref']]
        assert r['Supplier part']==item['supplier_part'] and ''.join(r['Manufacturer part'].split())==''.join(item['mpn'].split())
        r['Manufacturer part']=item['mpn'];r['Manufacturer']=item['manufacturer']
if contract_path.exists():
    contract=json.loads(contract_path.read_text(encoding='utf-8-sig'))
    for c in contract['components']:
        adopted[c['ref']]['Pin nets']='; '.join(str(p['number'])+'='+(p['net'] if p['connection']=='NET' else 'NC') for p in c['pins'])

new = [r for r in adopted.values() if r['Action']=='ADD']
assert len(new)==22 and len(retired)==20 and len(adopted)==171
old_mpns={''.join(r['Manufacturer part'].split()) for r in rows if r['Manufacturer part']}
new_mpns={r['Manufacturer part'] for r in adopted.values() if r['Manufacturer part'] and ''.join(r['Manufacturer part'].split()) not in old_mpns}
assert len(new_mpns)>=7
fields = list(rows[0])+['Action','Selection status']
for r in adopted.values():
    if r['Action'] != 'RETAIN':
        r['ENET properties JSON'] = ''
        r['Source'] = '2026-10-29 selected contract; baseline reference '+r['Reference']+'; NOT official ENET'
    if r['Action'] == 'REPLACE_SAME_QUANTITY':
        r['Device'] = ''
        r['Procurement status'] = 'SELECTED_IDENTITY_PRICE_NOT_CONFIRMED'
    if r['Manufacturer part'] == 'CGA3E2X7R1H104KT0Y0S': r['Manufacturer'] = 'TDK'
    if r['Supplier part'] == 'C2972766': r['Manufacturer'] = 'JKSEMI(金开盛)'
with (ROOT/'原理图准备/选型BOM.csv').open('w',encoding='utf-8-sig',newline='') as f:
    w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(adopted.values())
delta=[]
for r in new: delta.append({k:r[k] for k in ['Reference','Quantity','Manufacturer part','Supplier part','Requirements and notes']})
manifest={'status':'ADOPTED_SELECTION_NOT_YET_DRAWN','date':'2026-10-29',
 'scope':'independent SC regulator, 8 cells; not BMS or battery upgrade',
 'baseline_sha256':hashlib.sha256(BASE.read_bytes()).hexdigest(),
 'native_sha256':hashlib.sha256((ROOT/'eda工程文件/sc-regulator/sch/sc-regulator/01 稳压模块全部电路.esch2').read_bytes()).hexdigest(),
 'counts':{'baseline_instances':169,'added':22,'retired':20,'net_added':2,'selected_instances':171,
           'new_exact_MPNs_vs_regulator_baseline':len(new_mpns),'added_instance_MPNs':len({r['Manufacturer part'] for r in new}),
           'additional_mandatory_external_supply_if_not_already_fitted':1,
           'physical_added_including_previously_optional_supply':23,'physical_net_added_including_previously_optional_supply':3},
 'conditional_to_mandatory':[{'ref':'PSU5_EXT','mpn':'URB4805YMD-10WR3','qty':1,'reason':'原169条目已含此备选身份，现明确必配；此前未安装时实物/费用另加1个，不能漏算'}],
 'new_mpn_list':sorted(new_mpns),'added':delta,'retired':retired,
 'domestic_peripheral_review':'国产外围替代核对.json' if domestic_path.exists() else None,
 'gpio_contract':gpio,'working_window':{'highest_measured_cell_standby_V':2.0,'warning_derate_V':2.15,'regen_charge_stop_V':2.2,'scan_period_ms':20,'IWDG_prescaler':4,'IWDG_RLR':700,'first_arm_wait_after_12V_valid_s':1},
 'reused_or_replaced_refs':[r['Reference'] for r in adopted.values() if r['Action'] not in ['ADD','RETAIN']],
 'resistor':{'quantity':1,'ohm':0.82,'tolerance_percent':5,'power_rating':None,'thermal_design':'user scope','temperature_fuse':False},
 'price':{'currency':'CNY','known_added_partial':14.88,'known_refs':['Q_SC_ISO1','Q_SC_ISO2'],
          'unknown_new_refs':[r['Reference'] for r in new if not r['Reference'].startswith('Q_SC_ISO')],
          'complete_total':None,'net_material_delta':None,'mandatory_external_supply_price':None,'tax_freight_moq_not_totalled':True,
          'known_source':'BMS模块/原理图准备/9S5P增量报价.json; HYG 1+ ¥7.44, 2026-10-29',
          'formula':'14.88 + Σ(other 20 added board positions × verified unit price) + mandatory PSU5_EXT if previously not fitted + Σ(same-quantity replacement new-old) − Σ(20 retired positions × old unit price)',
          'battery_separate':{'added_cells':9,'unit_user_CNY':13,'increment_CNY':117}},
 'not_claimed':['native schematic updated','new official ENET or DRC','PCB or firmware implemented','physical validation','complete RMB total'],
 'checks':{'unique_references':True,'count_reconciliation':True,'new_MPN_count':True}}
manifest['count_semantics']={'selected_instances':'历史字段名：171个BOM条目，含COREBOARD_WEACT装配项；不是网表实例数。','selected_BOM_rows':171,'target_netlist_instances':170}
(ROOT/'检查/最终选型清单.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps(manifest['counts'],ensure_ascii=False))
