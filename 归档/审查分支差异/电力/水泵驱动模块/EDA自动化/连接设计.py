"""Independent electrical connection contract and functional placement for the pump."""
from eda import ROOT,save
import csv,json
BOM=list(csv.DictReader((ROOT/'原理图准备/BOM.csv').open(encoding='utf-8-sig',newline='')))
byref={ref.strip():r for r in BOM for ref in r['Reference'].split(',')}
parts=[]
def add(ref,pins,x,y,value,rotation=0):
 r=byref[ref];parts.append(dict(ref=ref,mpn=r['Manufacturer part'],manufacturer=r['Manufacturer'],supplierId=r['Supplier part'],pins={str(k):v for k,v in pins.items()},x=x,y=y,value=value,rotation=rotation))
def two(ref,a,b,x,y,val,rotation=0):add(ref,{1:a,2:b},x,y,val,rotation)
drv={i:None for i in range(1,46)}
for pin in [2,11,12,21]:drv[pin]='BUS_M'
for pin in [3,10,13,20,22,23,27,37,45]:drv[pin]='GND_SYS'
for pin in [4,5,6,17,18,19]:drv[pin]='MOTOR_A'
for pin in [7,8,9,14,15,16]:drv[pin]='MOTOR_B'
for pin in [24,28,29,35]:drv[pin]='DRV_DVDD'
for pin in [30,31]:drv[pin]='IPROPI'
for pin in [33,34]:drv[pin]='DRV_VREF'
drv.update({1:'DRV_VCP',25:'+5V',26:'DRV_nFAULT',40:'DRV_IN2',41:'DRV_IN1',42:'DRV_SLEEP',43:'DRV_CPL',44:'DRV_CPH'})
add('U_DRV',drv,600,1500,'DRV8262DDWR')
add('U_5V',{1:'BUS_M',2:'XL_SW',3:'GND_SYS',4:'XL_FB',5:'GND_SYS',6:'GND_SYS'},600,650,'XL7015E1')
add('U_CAN',{1:'CAN_TX',2:'GND_SYS',3:'+5V',4:'CAN_RX',5:'+3V3',6:'CANL',7:'CANH',8:'CAN_STB'},2700,650,'TCAN1042HGVDRQ1')
ma={i:None for i in range(1,25)};mb=ma.copy()
ma.update({1:'+5V',2:'+5V',3:'GND_SYS',4:'GND_SYS',10:'DRV_IN2',16:'CAN_STB',17:'DRV_nFAULT',20:'DRV_IN1',22:'DRV_SLEEP',23:'CAN_RX',24:'CAN_TX'})
mb.update({1:'+3V3',2:'+3V3',3:'GND_SYS',4:'GND_SYS',17:'ADC_BUS',18:'ADC_NTC',20:'ADC_I'})
add('J_MCU_A',ma,2500,1550,'U_MCU P1 / Normal QFN48')
add('J_MCU_B',mb,3250,1550,'U_MCU P2 / Normal QFN48')
two('J_BUS','BUS_M','GND_SYS',250,1900,'BUS IN / FUSED')
two('J_MOTOR','MOTOR_A','MOTOR_B',250,1250,'PUMP / 21V PWM')
two('C_VM_BULK','BUS_M','GND_SYS',600,1900,'470uF / 100V')
two('D_BUS','BUS_M','GND_SYS',950,1900,'SMCJ40A') # library pin 1=K checked on placement
two('C_DRV_VM1','BUS_M','GND_SYS',1300,1900,'10nF / 250V')
two('C_DRV_VM2','BUS_M','GND_SYS',1650,1900,'10nF / 250V')
two('C_DRV_VCP','DRV_VCP','BUS_M',1100,1630,'1uF / 16V')
two('C_DRV_CP','DRV_CPH','DRV_CPL',1450,1630,'100nF / 100V')
two('C_DRV_DVDD','DRV_DVDD','GND_SYS',1800,1630,'1uF / 16V')
two('C_DRV_VCC','+5V','GND_SYS',1100,1430,'100nF / 50V')
two('R_SLEEP','DRV_SLEEP','GND_SYS',1450,1430,'10k')
two('R_IN1_PD','DRV_IN1','GND_SYS',1800,1430,'10k')
two('R_IN2_PD','DRV_IN2','GND_SYS',1100,1230,'10k')
two('R_FAULT_PU','+3V3','DRV_nFAULT',1450,1230,'10k')
two('R_IPROPI','IPROPI','GND_SYS',1800,1230,'1.5k / 1%')
two('R_VREF_TOP','+3V3','DRV_VREF',450,1000,'10k / 1%')
two('R_VREF_BOTTOM','DRV_VREF','GND_SYS',800,1000,'10k / 1%')
two('C_VREF','DRV_VREF','GND_SYS',1150,1000,'100nF / 50V')
two('R_ADC_I','IPROPI','ADC_I',2450,1220,'1k')
two('C_ADC_I','ADC_I','GND_SYS',2800,1220,'10nF / 50V')
two('R_BUS_TOP1','BUS_M','BUS_DIV_MID',2450,1020,'110k / 0.1%')
two('R_BUS_TOP2','BUS_DIV_MID','BUS_SENSE',2800,1020,'110k / 0.1%')
two('R_BUS_BOTTOM','BUS_SENSE','GND_SYS',3150,1020,'10k / 0.1%')
two('R_ADC_BUS','BUS_SENSE','ADC_BUS',3500,1020,'1k')
two('C_ADC_BUS','ADC_BUS','GND_SYS',3850,1020,'100nF / 50V')
two('R_NTC_PULL','+3V3','ADC_NTC',3150,1220,'10k / 0.1%')
two('NTC_BOARD','ADC_NTC','GND_SYS',3500,1220,'10k / B3950')
two('C_ADC_NTC','ADC_NTC','GND_SYS',3850,1220,'10nF / 50V')
two('C_5V_IN','BUS_M','GND_SYS',250,820,'33uF / 100V')
two('C_5V_IN_HF','BUS_M','GND_SYS',600,820,'100nF / 100V')
two('D_5V','XL_SW','GND_SYS',950,650,'S310 / 100V')
two('L_5V','XL_SW','+5V',1300,650,'100uH')
two('C_5V_OUT','+5V','GND_SYS',1650,650,'330uF / 16V')
two('C_5V_OUT_HF','+5V','GND_SYS',2000,650,'100nF / 50V')
two('R_FB_TOP','+5V','XL_FB',600,410,'10k / 1%')
two('R_FB_BOTTOM','XL_FB','GND_SYS',950,410,'3.3k / 1%')
two('C_FB_FF','+5V','XL_FB',1300,410,'33nF / 50V')
two('C_CAN_VCC','+5V','GND_SYS',2450,820,'100nF / 50V')
two('C_CAN_VIO','+3V3','GND_SYS',2800,820,'100nF / 50V')
two('R_CAN_STB_PU','+3V3','CAN_STB',2450,410,'10k')
two('R_CAN_TXD_PU','+3V3','CAN_TX',2800,410,'10k')
add('D_CAN',{1:'CANH',2:'CANL',3:'GND_SYS'},3150,650,'PESD2CANFD24V-TR')
add('J_CAN',{1:'GND_SYS',2:'CANH',3:'CANL'},3500,650,'CAN / 1:GND 2:H 3:L')
two('R_TERM','CANH','CANL',3150,410,'120R / DNP default')
assert len(parts)==len({p['ref'] for p in parts})
expected={ref for ref,r in byref.items() if r['Scope']=='ON_BOARD'}
assert {p['ref'] for p in parts}==expected,expected-{p['ref'] for p in parts}
save('../原理图准备/原理图连接契约.json',{'parts':parts,'NC_policy':'All explicitly null pins are not connected; no hidden pin omission.','module':'WeAct Normal QFN48 STM32G431CBU6 module mounted through J_MCU_A/P1 and J_MCU_B/P2.'})
print(len(parts))
