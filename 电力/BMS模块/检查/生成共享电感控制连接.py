"""共享电感控制与保护连接；仅生成设计输入，不运行测试或创建PCB。"""
import json
from pathlib import Path
B = Path(__file__).resolve().parents[1]
items = []
group = ''
def part(ref, mpn, pins, value=None, note='', key=None):
    e = dict(ref=ref,mpn=mpn,pins={str(k):v for k,v in pins.items()},pageGroup=group,note=note)
    if value:e['value']=value
    if key:e['libraryKey']=key
    items.append(e)
def r(ref,a,b,v,note='',precision='1%',key='R0603'):
    part('R_BAL_'+ref,None,{1:a,2:b},v+' '+precision,note,key)
def c(ref,a,b,v,note=''):
    part('C_BAL_'+ref,None,{1:a,2:b},v,note,'C0603')
def logic(ref,key,pins):
    part('U_BAL_'+ref,key,pins,key,'未用输入固定电平；本页B0或BAL_N电压域不得混接。',key)
def supplycap(ref,v,g):c(ref,v,g,'100nF 16V X7R')
def opto(ref,lednet,outnet,gnd,vcc,driver_rail,driver_gnd):
    r(ref+'_BASE',lednet,ref+'_B','4.7kΩ','74HCT只驱动MMBT3904基极；约0.9mA基极电流。')
    r(ref+'_BASE_PD',ref+'_B',driver_gnd,'100kΩ','无驱动时确保NPN关断。')
    part('Q_BAL_'+ref+'_DRV','onsemi MMBT3904LT1G',{1:ref+'_B',2:driver_gnd,3:ref+'_K'},key='MMBT3904LT1G',note='SOT-23物理脚1=B、2=E、3=C；低侧开关驱动光耦LED，不让74HCT直接承担LED电流。')
    r(ref+'_LED',driver_rail,ref+'_A','750Ω','5V侧串联电阻；按LED Vf约1.2V、NPN饱和压降约0.2V，IF约4.8mA。')
    part('U_BAL_'+ref,'Everlight EL357N(C)(TA)-G',{1:ref+'_A',2:ref+'_K',3:gnd,4:outnet},key='EL357N(C)(TA)-G',note='控制输入高时LED约4.8mA、隔离输出_RAW_N拉低；后续浮地反相器保持EN/DIR语义，输出与输入域隔离。')
    r(ref+'_PU',vcc,outnet,'10kΩ')
group='BAL_AUX'
for ref,neg,pos in [('GATE','BAL_B9','BAL_GATE_BIAS'),('FLOAT','BAL_N','BAL_AUX12')]:
    if ref=='GATE':
        module='Mornsun B0512S-1WR3'; module_key='B0512S-1WR3'
        pins={1:'B0',2:'VOUT_5V',3:neg,4:pos}
        module_note='原厂B0512S-1WR3 SIP4:1 GND=B0,2 VIN=VOUT_5V,3 0V=BAL_B9,4 +VO=BAL_GATE_BIAS；输出不稳压。非自制均衡变压器；只供隔离偏置。'
        load='1.1kΩ'; load_key='RC2512FK-071K1L'
        note='MORNSUN B_S-1WR3原厂DS：最大/最小负载电流84/9mA。1.1kΩ、1%按1111Ω最坏值，在10.8V时约9.72mA，高于9mA最低负载；12V约10.91mA、0.131W。Yageo RC2512FK-071K1L，2512、1W。'
    else:
        module='Mornsun B0512S-2WR3'; module_key='B0512S-2WR3'
        pins={1:'VOUT_5V',2:'B0',4:neg,6:pos}
        module_note='原厂B0512S-2WR3 SIP6:1 VIN=VOUT_5V,2 GND=B0,4 0V=BAL_N,6 +VO=BAL_AUX12；输出不稳压。仅供浮地逻辑，不是定制均衡变压器。'
        load='620Ω'; load_key='RC2512FK-07620RL'
        note='MORNSUN B_S-2WR3原厂DS：最大/最小负载电流167/17mA。620Ω、1%按626.2Ω最坏值，在10.8V时约17.25mA，略高于17mA最低负载，仍须核对实际输出下限/容差；12V约19.35mA、0.232W；13.2V电阻低限时约0.284W。Yageo RC2512FK-07620RL，2512、1W。'
    part('U_BAL_AUX_'+ref,module,pins,key=module_key,note=module_note)
    r('AUX_'+ref+'_LOAD',pos,neg,load,note,key=load_key)
    c('AUX_'+ref+'_IN','VOUT_5V','B0','4.7µF 10V X7R')
    c('AUX_'+ref+'_OUT',pos,neg,'4.7µF 25V X7R')
part('U_BAL_LDO','TI TLV76050DBZR',{1:'BAL_FLOAT_5V',2:'BAL_AUX12',3:'BAL_N'},key='TLV76050DBZR',note='保护逻辑独立于LM51772 EN和VCC供电，避免启动互锁死锁。需按实际输入/输出电流核算LDO耗散。')
c('LDO_IN','BAL_AUX12','BAL_N','1µF 25V X7R')
c('LDO_OUT','BAL_FLOAT_5V','BAL_N','1µF 10V X7R')
part('U_BAL_REF','TI REF3125AIDBZT',{1:'BAL_FLOAT_5V',2:'BAL_REF2V5',3:'BAL_N'},key='REF3125',note='2.5V基准；保护阈值电阻用0.1%，计入基准初始误差、温漂和比较器偏置。')
supplycap('REF_IN','BAL_FLOAT_5V','BAL_N');c('REF_OUT','BAL_REF2V5','BAL_N','1µF 10V X7R')
group='BAL_ISOLATED_I2C'
part('U_BAL_I2C_ISO','TI ISO1640BDR',{1:'CORE_3V3',2:'BQ_I2C_SDA',3:'BQ_I2C_SCL',4:'B0',5:'BAL_N',6:'BAL_I2C_SCL_FLOAT',7:'BAL_I2C_SDA_FLOAT',8:'BAL_FLOAT_5V'},key='ISO1640BDR',note='按原厂SOIC-8脚位；两侧电源各自就近100nF，浮地SCL/SDA各以4.7kΩ上拉到BAL_FLOAT_5V。')
supplycap('I2C_ISO_CORE','CORE_3V3','B0');supplycap('I2C_ISO_FLOAT','BAL_FLOAT_5V','BAL_N')
r('I2C_SCL_FLOAT_PU','BAL_FLOAT_5V','BAL_I2C_SCL_FLOAT','4.7kΩ')
r('I2C_SDA_FLOAT_PU','BAL_FLOAT_5V','BAL_I2C_SDA_FLOAT','4.7kΩ')
for idx,(v,g,net) in enumerate([('VOUT_5V','B0','BAL_PG5'),('BAL_FLOAT_5V','BAL_N','BAL_FLOAT_PG')],1):
    part('U_BAL_SUP'+str(idx),'TI TLV803EB46DCKR',{1:g,2:net,3:v},key='TLV803EB46DCKR',note='4.63V复位监控；电源不足时清零寄存器/故障锁存器。')
    r('SUP'+str(idx)+'_PU',v,net,'10kΩ');supplycap('SUP'+str(idx),v,g)
group='BAL_LOGIC'
logic('SHIFT','74HCT595',{1:'BAL_A1',2:'BAL_A2',3:'BAL_A3',4:'BAL_DIR_CMD',5:'BAL_STAGE_REQ',6:None,7:None,8:'B0',9:None,10:'BAL_PG5',11:'BAL_SCK',12:'BAL_LATCH_DELAY',13:'B0',14:'BAL_MOSI',15:'BAL_A0',16:'VOUT_5V'})
decoder={**{i+1:'BAL_SEL'+str(i)+'_N' if i<9 else None for i in range(11)},12:'B0',13:None,14:None,15:None,16:None,17:None,18:'BAL_PERMIT_N',19:'BAL_BLANK',20:'BAL_A3',21:'BAL_A2',22:'BAL_A1',23:'BAL_A0',24:'VOUT_5V'}
logic('DECODER','74HCT154',decoder)
logic('TIMER','CD74HCT123M',{1:'B0',2:'BAL_CS_N',3:'BAL_PG5',4:'BAL_NOT_BLANK',5:'BAL_WD_OK',6:'BAL_WD_CX',7:'BAL_WD_RX',8:'B0',9:'B0',10:'BAL_RUN',11:'BAL_PG5',12:None,13:'BAL_BLANK',14:'BAL_BLANK_CX',15:'BAL_BLANK_RX',16:'VOUT_5V'})
r('BLANK_RT','VOUT_5V','BAL_BLANK_RX','100kΩ');c('BLANK_CT','BAL_BLANK_RX','BAL_BLANK_CX','680nF 10V X7R','单稳态约0.45RC=30.6ms，系数/容差须计入，不当作关断实测证明。')
r('WD_RT','VOUT_5V','BAL_WD_RX','1MΩ');c('WD_CT','BAL_WD_RX','BAL_WD_CX','1µF 10V X7R','心跳每50ms上升沿；约450ms失效关断，静态高/低不能续命。')
r('LATCH_RC','BAL_CS_N','BAL_LATCH_RC','1kΩ');c('LATCH_RC','BAL_LATCH_RC','B0','100nF 16V X7R')
logic('SCHMITT','SN74HCT14D',{1:'BAL_LATCH_RC',2:'BAL_LATCH_INV',3:'BAL_LATCH_INV',4:'BAL_LATCH_DELAY',5:'BAL_PERMIT',6:'BAL_PERMIT_N',7:'B0',8:'BAL_SEL_SETTLED',9:'BAL_SETTLE_INV',10:'BAL_SETTLE_INV',11:'BAL_SETTLE_RC',12:None,13:'B0',14:'VOUT_5V'})
logic('AND1','SN74HCT08D',{1:'BAL_PG5',2:'BMS_LOCAL_OK',3:'BAL_PG_LOCAL',4:'BAL_PG_LOCAL',5:'BAL_OE',6:'BAL_PERMIT0',7:'B0',8:'BAL_PERMIT',9:'BAL_PERMIT0',10:'BAL_WD_OK',11:'BAL_MATRIX_ALLOWED',12:'BAL_PERMIT',13:'BAL_NOT_BLANK',14:'VOUT_5V'})
logic('AND2','SN74HCT08D',{1:'BAL_MATRIX_ALLOWED',2:'BAL_SEL_SETTLED',3:'BAL_STAGE_ALLOWED',4:'BAL_STAGE_ALLOWED',5:'BAL_STAGE_REQ',6:'BAL_EN_CMD',7:'B0',8:None,9:'B0',10:'B0',11:None,12:'B0',13:'B0',14:'VOUT_5V'})
r('SETTLE_RT','BAL_MATRIX_ALLOWED','BAL_SETTLE_RC','47kΩ');c('SETTLE_CT','BAL_SETTLE_RC','B0','330nF 16V X7R')
part('D_BAL_SETTLE','Nexperia 1N4148W',{1:'BAL_MATRIX_ALLOWED',2:'BAL_SETTLE_RC'},key='1N4148W',note='物理脚1=K、脚2=A；阳极接RC、阴极接允许信号，RC高于允许时快速放电；最终AND仍受即时允许门控制。')
for ref,net in [('OE_PD','BAL_OE'),('HB_PD','BAL_RUN'),('LATCH_PD','BAL_CS_N'),('CLK_PD','BAL_SCK'),('DATA_PD','BAL_MOSI')]:r(ref,net,'B0','100kΩ')
for ref in ['SHIFT','DECODER','TIMER','SCHMITT','AND1','AND2']:supplycap(ref,'VOUT_5V','B0')
opto('EN_OPTO','BAL_EN_CMD','BAL_EN_RAW_N','BAL_N','BAL_FLOAT_5V','VOUT_5V','B0')
opto('DIR_OPTO','BAL_DIR_CMD','BAL_DIR_RAW_N','BAL_N','BAL_FLOAT_5V','VOUT_5V','B0')
group='BAL_FLOAT_CONTROL'
logic('FLOAT_INV','SN74HCT14D',{1:'BAL_EN_RAW_N',2:'BAL_EN_REQ_FLOAT',3:'BAL_DIR_RAW_N',4:'BAL_DIR_FLOAT',5:'BAL_DIR_FLOAT',6:'BAL_TARGET_FLOAT',7:'BAL_N',8:None,9:'BAL_N',10:None,11:'BAL_N',12:None,13:'BAL_N',14:'BAL_FLOAT_5V'})
logic('FAULT_LATCH','SN74HCT74D',{1:'BAL_FAULTS_OK',2:'BAL_FLOAT_5V',3:'BAL_EN_REQ_FLOAT',4:'BAL_FLOAT_5V',5:'BAL_FAULT_ARMED',6:None,7:'BAL_N',8:None,9:None,10:'BAL_FLOAT_5V',11:'BAL_N',12:'BAL_N',13:'BAL_N',14:'BAL_FLOAT_5V'})
logic('FLOAT_AND','SN74HCT08D',{1:'BAL_EN_REQ_FLOAT',2:'BAL_FAULT_ARMED',3:'BAL_EN_FLOAT',4:'BAL_FLOAT_PG',5:'BAL_FAULT_ARMED',6:'BAL_FLOAT_READY',7:'BAL_N',8:None,9:'BAL_N',10:'BAL_N',11:None,12:'BAL_N',13:'BAL_N',14:'BAL_FLOAT_5V'})
r('EN_PD','BAL_EN_FLOAT','BAL_N','10kΩ');r('MODE_PU','BAL_FLOAT_5V','BAL_MODE_SS_DONE','10kΩ');r('FAULTS_PU','BAL_FLOAT_5V','BAL_FAULTS_OK','10kΩ')
part('D_BAL_FLOAT_RESET','Nexperia 1N4148W',{1:'BAL_FLOAT_PG',2:'BAL_FAULTS_OK'},key='1N4148W',note='物理脚1=K接BAL_FLOAT_PG，脚2=A接BAL_FAULTS_OK/锁存CLR；floatPG低时阳极侧故障网可经二极管拉低CLR，统一使用电源主管脚名BAL_FLOAT_PG。')
for ref in ['FLOAT_INV','FAULT_LATCH','FLOAT_AND']:supplycap(ref,'BAL_FLOAT_5V','BAL_N')
part('Q_BAL_FB','Nexperia 2N7002',{1:'BAL_TARGET_FLOAT',2:'BAL_FB',3:'BAL_FB_TGT'},key='2N7002',note='标准SOT-23物理脚1=G、2=S、3=D；target时将额外反馈上臂接入，关断default只保留高buffer目标。')
r('FB_SEL_PD','BAL_TARGET_FLOAT','BAL_N','100kΩ')
r('FB_EXTRA','BAL_BUF_P','BAL_FB_TGT','2.42MΩ','与523kΩ并联约430.06kΩ；1V基准、100kΩ下臂时低目标约5.3006V。0.1%值更利于目标一致。')
r('FB_TOP','BAL_BUF_P','BAL_FB','523kΩ','1V基准、100kΩ下臂时默认目标6.23V；cell电压上限由独立OV比较器控制。')
# 状态回传：失电或锁存故障时高电平；新鲜enable上升沿且无故障才恢复。
r('READY_BASE','BAL_FLOAT_READY','BAL_READY_B','4.7kΩ','浮地HCT只驱动MMBT3904基极；READY有效时LED约4.8mA。')
r('READY_BASE_PD','BAL_READY_B','BAL_N','100kΩ','无驱动时确保NPN关断。')
part('Q_BAL_READY_DRV','onsemi MMBT3904LT1G',{1:'BAL_READY_B',2:'BAL_N',3:'BAL_READY_K'},key='MMBT3904LT1G',note='SOT-23物理脚1=B、2=E、3=C；在BAL_N域低侧驱动READY光耦LED。')
r('READY_LED','BAL_FLOAT_5V','BAL_READY_A','750Ω','5V侧串联电阻；按LED Vf约1.2V、NPN饱和压降约0.2V，IF约4.8mA。')
part('U_BAL_READY_OPTO','Everlight EL357N(C)(TA)-G',{1:'BAL_READY_A',2:'BAL_READY_K',3:'B0',4:'BAL_READY_N'},key='EL357N(C)(TA)-G',note='BAL_READY_N作为READY_N接现有MCU PB2/P2-7空脚；不得接VREF/P2-8，接口文件不在本次修改范围。')
r('READY_PU','CORE_3V3','BAL_READY_N','10kΩ')
group='BAL_PROTECTION'
comparisons=[('BAL_CELL_UV_DIV','BAL_REF2V5','BAL_FAULTS_OK'),('BAL_REF2V5','BAL_CELL_OV_DIV','BAL_FAULTS_OK'),('BAL_REF2V5','BAL_BUF_OV_DIV','BAL_FAULTS_OK'),('BAL_TEMP_DIV','BAL_TEMP_HOT_REF','BAL_FAULTS_OK'),('BAL_TEMP_OPEN_REF','BAL_TEMP_HALF','BAL_FAULTS_OK'),('BAL_SS_HALF','BAL_SS_REF','BAL_MODE_SS_DONE'),('BAL_REF2V5','BAL_CAP1_OV_DIV','BAL_FAULTS_OK'),('BAL_CAP2_OV_REF','BAL_CAP2_OV_DIFF_DIV','BAL_FAULTS_OK'),('BAL_CAP3_OV_REF','BAL_CAP3_OV_DIFF_DIV','BAL_FAULTS_OK'),('BAL_BUF_OV_REF','BAL_BUF_OV_DIFF_DIV','BAL_FAULTS_OK'),('BAL_TEMP_COLD_REF','BAL_TEMP_DIV','BAL_FAULTS_OK'),('BAL_REF2V5','BAL_N',None)]
for n in range(3):
    pins={3:'BAL_FLOAT_5V',12:'BAL_N'}
    for j,(ip,im,o) in enumerate(comparisons[4*n:4*n+4]):
        pp,pm,po=[(5,4,2),(7,6,1),(9,8,14),(11,10,13)][j];pins[pp]=ip;pins[pm]=im;pins[po]=o
    part('U_BAL_CMP'+str(n+1),'TI TLV1704AIPWR',pins,key='TLV1704',note='输出开漏；所有保护good为开路，fault拉低BAL_FAULTS_OK锁存停止；SS comparator仅控制MODE。')
    supplycap('CMP'+str(n+1),'BAL_FLOAT_5V','BAL_N')
def div(ref,top,bottom,net,rt,rb,note=''):
    r(ref+'_T',top,net,rt,note,precision='0.1%');r(ref+'_B',net,bottom,rb,note,precision='0.1%')
div('UV','BAL_P','BAL_N','BAL_CELL_UV_DIV','18.7kΩ','100kΩ')
div('OV','BAL_P','BAL_N','BAL_CELL_OV_DIV','16.5kΩ','24.9kΩ')
div('BUF','BAL_BUF_P','BAL_N','BAL_BUF_OV_DIV','36kΩ','20kΩ',note='2.5V比较基准对应约7.0V缓冲母线过压备用阈值。')
div('TEMP_HOT_REF','BAL_FLOAT_5V','BAL_N','BAL_TEMP_HOT_REF','33.5kΩ','10kΩ',note='同源BAL_FLOAT_5V比率参考；对10kΩ/B3435 NTC约60°C标称阈值。')
div('TEMP_COLD_REF','BAL_FLOAT_5V','BAL_N','BAL_TEMP_COLD_REF','54.9kΩ','100kΩ',note='同源浮地5V比率参考约3.23V；与10kΩ/B3435 NTC比较，约10°C触发硬件冷停。')
div('TEMP_OPEN_REF','BAL_FLOAT_5V','BAL_N','BAL_TEMP_OPEN_REF','110kΩ','90kΩ',note='浮地5V经110kΩ/90kΩ分压得到2.25V开路阈值。')
div('TEMP_HALF','BAL_TEMP_DIV','BAL_N','BAL_TEMP_HALF','100kΩ','100kΩ')
r('TEMP_PULL','BAL_FLOAT_5V','BAL_TEMP_DIV','10kΩ')
part('NTC_BAL_FLOAT','Murata NCP18XH103F03RB',{1:'BAL_TEMP_DIV',2:'BAL_N'},'10kΩ B3435 1%','贴近缓冲电容组热点测温；按B3435估算R60≈2.98kΩ、10kΩ上拉到5V时传感节点约1.15V，33.5kΩ/10kΩ同源阈值约60°C硬停；R10≈18.4kΩ、节点约3.24V，54.9kΩ/100kΩ同源阈值约10°C冷停。软件默认15–55°C才允许；B0域BOARD NTC分别测温，不可直连浮地NTC。','NCP18XH103F03RB')
div('SS','BAL_SS','BAL_N','BAL_SS_HALF','1MΩ','1MΩ')
div('SS_REF','BAL_REF2V5','BAL_N','BAL_SS_REF','354kΩ','100kΩ',note='2.5V经354kΩ/100kΩ得到约0.5507V；SS_HALF比较对应BAL_SS约1.101V。')
# 四节电容各自独立差分OV；CAP1直接分压，CAP2/CAP3/BUF沿用电阻求和法，阈值约2.55V。
div('CAP1_OV','BAL_CAP1','BAL_N','BAL_CAP1_OV_DIV','2kΩ','100kΩ')
for name,top,low in [('CAP2','BAL_CAP2','BAL_CAP1'),('CAP3','BAL_CAP3','BAL_CAP2'),('BUF','BAL_BUF_P','BAL_CAP3')]:
    div(name+'_OV',top,'BAL_N','BAL_'+name+'_OV_DIFF_DIV','200kΩ','100kΩ')
    r(name+'_OV_LOW',low,'BAL_'+name+'_OV_REF','100kΩ',precision='0.1%')
    r(name+'_OV_REF','BAL_REF2V5','BAL_'+name+'_OV_REF','98kΩ',precision='0.1%')
    r(name+'_OV_GND','BAL_'+name+'_OV_REF','BAL_N','102kΩ',precision='0.1%')
group='BAL_BUFFER_CURRENT'
for idx,(pos,neg) in enumerate([('BAL_CAP1','BAL_N'),('BAL_CAP2','BAL_CAP1'),('BAL_CAP3','BAL_CAP2'),('BAL_BUF_P','BAL_CAP3')],1):
    for suffix in ('A','B'):
        part('C_BAL_BUFFER'+str(idx)+suffix,'Fenghua FH2R7L106M-N1025',{1:pos,2:neg},'10F 2.7V ±20%','原厂10×25mm径向电容，四串两并共8只；极性1正2负需与实体端子标识再核对。每节独立2.55V差分硬件OV并联监测，四串约5F。','FH2R7L106M-N1025')
    r('BUFFER'+str(idx)+'_BLEED',pos,neg,'330Ω','仅作缓慢静态均压；每只硬件OV才是动态过压保护。使用0.125W以上。')
part('U_BAL_CURRENT','Allegro ACS724LLCTR-05AB-T',{1:'BAL_P_RAW',2:'BAL_P_RAW',3:'BAL_P',4:'BAL_P',5:'B0',6:'BAL_CURRENT_FILTER',7:'BAL_CURRENT_5V',8:'VOUT_5V'},key='ACS724LLCTR-05AB-T',note='±5A，400mV/A，5V隔离Hall；初级导体串选中电芯正端，二次以B0参考。量程误差/带宽与零偏需校准，软件零流不是强制关断保证。BUF低于5V时只能先做峰值受控充电，不能宣称平均限流规格已成立；1.85A窗口估计需堆ESR≤0.20Ω，0.25Ω只是无余量绝对边界，须实测确认。')
c('CURRENT_FILTER','BAL_CURRENT_FILTER','B0','1nF 16V C0G');supplycap('CURRENT_SUP','VOUT_5V','B0')
div('CURRENT_ADC','BAL_CURRENT_5V','B0','BAL_CURRENT_ADC','10kΩ','20kΩ')
c('CURRENT_ADC','BAL_CURRENT_ADC','B0','1nF 16V C0G')
out=B/'原理图准备'/'共享电感控制保护连接.json'
out.write_text(json.dumps(items,ensure_ascii=False,indent=2),encoding='utf-8')
print('design entries',len(items))
