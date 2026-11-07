"""Conditional precision-comparator candidate; not a released schematic."""
import hashlib
import json
import math
from pathlib import Path

root=Path(__file__).resolve().parents[1]
net=root/'检查/本轮磁盘读回.enet'
outdir=root/'检查/候选_精密比较器接管'
outdir.mkdir(exist_ok=True)
vref=4.096
initial=.0005
dt=35  # 0..60 C as a bounded comparison, not an agreed requirement
tempco=10e-6
offset=.002  # LMP7300 SOIC full-temperature offset at specified common mode
rtol=.001+25e-6*dt
top=825000; bottom=100000
reflo=vref*(1-initial-tempco*dt); refhi=vref*(1+initial+tempco*dt)
kmin=1+top*(1-rtol)/(bottom*(1+rtol))
kmax=1+top*(1+rtol)/(bottom*(1-rtol))
triggerlo=(reflo-offset)*kmin
triggerhi=(refhi+offset)*kmax
guarantee_at_10mv=triggerhi+.010*kmax
c=.05286; vlimit=39.209-.2; power=915; max_time=.002
trigger_boundary=math.sqrt(vlimit*vlimit-2*power*max_time/c)
result={
 'status':'ISOLATED_CONCEPT_NOT_QUALIFIED',
 'basis_netlist_sha256':hashlib.sha256(net.read_bytes()).hexdigest(),
 'purpose':'review minimum concept with additional reference and isolated 5V, preserve original TLV6700 12V hardware backup',
 'components':[
  {'ref':'U_MON5','candidate':'TLV76050DBZR','function':'12V_DRV -> locally regulated MON_5V; stability and transient qualification pending'},
  {'ref':'U_REF','candidate':'SGM4024-4.096XN5G/TR','function':'MON_5V -> 4.096V reference; initial accuracy and drift per manufacturer'},
  {'ref':'U_CMP','candidate':'LMP7300MA/NOPB','function':'MON_5V comparator, external reference, nominal zero hysteresis only for optimistic screening'},
  {'ref':'Q_REQUEST','candidate':'BSS138','function':'open-drain pull-down of existing DUMP_OFF when comparator indicates overvoltage; precise supplier and safe startup not qualified'},
  {'ref':'R_SENSE_TOP','candidate':'PTFR0805B825KP9','value_ohm':825000,'accuracy':'.1%, 25ppm/C; single-short input stress not closed'},
  {'ref':'R_SENSE_BOT','candidate':'PTFR0805B100KP9','value_ohm':100000,'accuracy':'.1%, 25ppm/C'},
  {'ref':'R_OUT_PULLUP','candidate':'existing 10k nominal','function':'MON_5V pull-up to comparator OUT and MOS gate'},
  {'ref':'R_GATE_PD','candidate':'existing 100k nominal','function':'MOS gate to GND_SYS'},
  {'ref':'C_LDO_OUT','candidate':'existing 1uF nominal','function':'LDO output; effective minimum pending'},
  {'ref':'C_REF_OUT','candidate':'existing 1uF nominal','function':'reference output; effective minimum pending'}],
 'net_contract':{
  'U_MON5':'1 OUT=MON_5V; 2 IN=12V_DRV; 3 GND=GND_SYS',
  'U_REF':'1 NC; 2 GND=GND_SYS; 3 EN=MON_5V; 4 IN=MON_5V; 5 OUT=MON_REF_4096',
  'U_CMP':'1 INP=BUS_DIV; 2 INN=MON_REF_4096; 3 GND=GND_SYS; 4 OUT=OV_HIGH; 5 HYSTN=internal_REF; 6 HYSTP=internal_REF; 7 REF=internal_REF; 8 V+=MON_5V',
  'R_SENSE_TOP':'BUS_EXT -> BUS_DIV', 'R_SENSE_BOT':'BUS_DIV -> GND_SYS',
  'R_OUT_PULLUP':'MON_5V -> OV_HIGH','R_GATE_PD':'OV_HIGH -> GND_SYS',
  'Q_REQUEST':'G=OV_HIGH; S=GND_SYS; D=DUMP_OFF',
  'backup':'existing TLV6700, Q_PMOS_CTRL, 12V_DRV and 10k DUMP_OFF pullup unchanged'},
 'static_screen':{'assumptions':'0..60C component temperature, no self-heating, 12V-derived 5V valid, ideal zero hysteresis, no input bias/CMRR/line/load regulation/board leakage, independent resistor and reference endpoints',
  'ref_min_V':reflo,'ref_max_V':refhi,'offset_V':offset,'resistor_fraction_extreme':rtol,
  'nominal_bus_threshold_V':vref*(1+top/bottom),'static_min_V':triggerlo,'static_max_V':triggerhi,
  'bus_at_10mV_overdrive_max_V':guarantee_at_10mv,
  'release_requirement_V':37.8,'effective_switch_budget_V':trigger_boundary,
  'release_pass':triggerlo>37.8,'specified_overdrive_before_budget_pass':guarantee_at_10mv<trigger_boundary,
  'reference_ratio_feasible_k_min':37.8/(reflo-offset),
  'reference_ratio_feasible_k_max':trigger_boundary/(refhi+offset+.010),
  'comparator_delay_condition':'15us maximum only at 25C, 5V, 10mV overdrive, 10pF specified load; not a 0..60C or actual gate-load guarantee'},
 'existing_mcu_path':{'ratio_bus_to_ADC_approx':61,'Vref_example_V':3.3,'ADC_12bit_bus_LSB_V':61*3.3/4096,
  'max_positive_detection_error_at_38V_V':trigger_boundary-38,
  'max_positive_error_ADC_LSB':(trigger_boundary-38)/(61*3.3/4096),
  'status':'CONDITIONAL_ONLY: actual sample age, total ADC/divider/reference error, firmware, 12V and 0.82ohm qualified current ramp unknown'},
 'thermal_scope':'0.82 ohm physical resistor, switch, diode and wiring remain independently UNQUALIFIED',
 'procurement_scope':'LMP7300 exact CNY in-stock not established; SGM4024 exact product JLC listing indicated zero; precise total cost not established',
 'not_covered':['FOC threshold 39.209V and reserve 0.2V are conditional inputs',
                'zero hysteresis is for optimistic arithmetic; not an approved noise or latch design',
                'reference startup, power-loss reverse current and transistor reset behavior are unqualified',
                'sense top-resistor single-short can overstress the comparator input',
                'switch, diode, wiring current ramp and resistor pulse/repeat heat unqualified'],
 'sources':['https://www.ti.com/lit/ds/symlink/lmp7300.pdf','https://www.sg-micro.com/rect/assets/1e09b082-efe2-48eb-a053-fdd06b611465/SGM4024.pdf','https://www.ti.com/lit/ds/symlink/tlv760.pdf']}
(outdir/'概念针网和边界.json').write_bytes((json.dumps(result,ensure_ascii=False,indent=2)+'\n').encode('utf-8'))
print(json.dumps(result['static_screen'],ensure_ascii=False,indent=2))
