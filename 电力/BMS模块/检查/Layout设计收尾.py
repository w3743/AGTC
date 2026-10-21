"""落实 layout 入口的原理图变更；不创建 PCB，不运行测试。"""
import json, copy, pathlib, sys
B=pathlib.Path(__file__).resolve().parents[1]; C=B/'检查'; S=B/'原理图准备'
def read(p): return json.loads(p.read_text('utf-8-sig'))
def write(p,d): p.write_text(json.dumps(d,ensure_ascii=False,indent=2),encoding='utf-8')
cache=read(C/'Layout收尾库检索.json')
base={e['ref']:e for e in read(C/'原理图构建清单.json')}
def exact(e,key):
 d=next(d for d in cache[key] if d.get('name')==key or d.get('manufacturerId')==key)
 e.update(mpn=key,libraryKey=key,libUuid=d['uuid'],libraryUuid=d['libraryUuid'],libType='3',keepExisting=False,unresolved=[],nativeFootprint={'uuid':d['footprintUuid'],'libraryUuid':d['libraryUuid'],'name':d['footprintName']})
 return e
def new(ref,template,pins,note):
 e=copy.deepcopy(base[template]);e.update(ref=ref,pins=pins,note=note,pageGroup='BAL_OC_B0',sourceFile='共享电感控制保护连接.json',keepExisting=False,unresolved=[])
 return e
note='B0 域独立双向过流硬停，复用 ACS724 / TLV1704 / HCT74 型号；只关转换器 EN，保留当前矩阵供续流。故障恢复须 OE 先低、确认零流，再产生新 OE 上升沿；禁止以看门狗心跳自动重置。'
adds=[
 new('U_BAL_OC_CMP','U_BAL_CMP1',{'3':'VOUT_5V','12':'B0','5':'BAL_OC_HI_REF','4':'BAL_CURRENT_5V','2':'BAL_CURRENT_TRIP_N','7':'BAL_CURRENT_5V','6':'BAL_OC_LO_REF','1':'BAL_CURRENT_TRIP_N','9':'B0','8':'B0','14':None,'11':'B0','10':'B0','13':None},note+' 通道1/2开漏线与，通道3/4两输入接地、输出NC。'),
 new('U_BAL_OC_LATCH','U_BAL_FAULT_LATCH',{'1':'BAL_CURRENT_CLR_N','2':'VOUT_5V','3':'BAL_OE','4':'VOUT_5V','5':'BAL_CURRENT_ARMED','6':None,'7':'B0','8':None,'9':None,'10':'VOUT_5V','11':'B0','12':'B0','13':'B0','14':'VOUT_5V'},note),
]
for ref,template,pins in [
 ('R_BAL_OC_HI_T','R_BAL_SS_REF_B',{'1':'VOUT_5V','2':'BAL_OC_HI_REF'}),
 ('R_BAL_OC_HI_B','R_BAL_SS_REF_B',{'1':'BAL_OC_HI_REF','2':'B0'}),
 ('R_BAL_OC_LO_T','R_BAL_SS_REF_B',{'1':'VOUT_5V','2':'BAL_OC_LO_REF'}),
 ('R_BAL_OC_LO_B','R_BAL_SS_REF_B',{'1':'BAL_OC_LO_REF','2':'B0'}),
 ('R_BAL_OC_PULL','R_BAL_CURRENT_ADC_T',{'1':'VOUT_5V','2':'BAL_CURRENT_TRIP_N'}),
 ('C_BAL_OC_CMP','C_BAL_CURRENT_SUP',{'1':'VOUT_5V','2':'B0'}),
 ('C_BAL_OC_LATCH','C_BAL_CURRENT_SUP',{'1':'VOUT_5V','2':'B0'}),
]:
 e=new(ref,template,pins,note)
 if ref in ('R_BAL_OC_HI_T','R_BAL_OC_LO_B'):exact(e,'RT0603BRD0744K2L');e['value']='44.2kΩ ±0.1% 0603'
 adds.append(e)
for i,e in enumerate(adds):
 e.update(x=250+(i%4)*700,y=250+(i//4)*650);e['xy']={'x':e['x'],'y':e['y']}
paths=[S/'共享电感功率级连接.json',S/'共享电感控制保护连接.json',C/'原理图构建清单.json',C/'Layout器件映射.json',C/'当前施工输入.json',C/'共享电感施工输入.json']
for p in paths:
 j=read(p);es=j if isinstance(j,list) else j['components']
 for e in es:
  e['pins']={k:('BAL_P' if v=='BAL_P_RAW' else v) for k,v in e.get('pins',{}).items()}
  r=e['ref']
  if r=='R_BAL_AVG1':e['pins']['2']='BAL_BUF_SENSE';e['note']='27mΩ输出平均检测；ISNSP/N Kelvin只跨本电阻，Hall串在其后。名义1.85A软设定，独立±2.4185A Hall比较器锁存仅作安全上限。'
  if r=='U_BAL_CTRL1':e['pins']['19']='BAL_BUF_SENSE'
  if r=='U_BAL_CURRENT':
   e['pins'].update({'1':'BAL_BUF_SENSE','2':'BAL_BUF_SENSE','3':'BAL_BUF_P','4':'BAL_BUF_P'})
   e['note']='±5A、400mV/A、零流VCC/2，同源5V比率输出；初级串缓冲端27mΩ分流器之后，二次参考B0。正向流入buffer，负向流出；ADC和独立双向比较器共用测量。1nF FILTER保持较快硬停响应，实测噪声后可调整；勿将5V输出网名误认为固定电源。'
  if r=='U_BAL_AND2':
   e['pins'].update({'6':'BAL_STAGE_EN_PRE_OC','9':'BAL_STAGE_EN_PRE_OC','10':'BAL_CURRENT_ARMED','8':'BAL_EN_CMD','12':'BAL_PG5','13':'BAL_CURRENT_TRIP_N','11':'BAL_CURRENT_CLR_N'})
   e['note']=note+' gate3复用串联EN许可，gate4复用PG5 AND TRIP_N复位许可；不增加AND芯片。'
  if r=='R_BAL_FB_EXTRA':
   exact(e,'RC0603FR-072M7L');e['value']='2.7MΩ ±1% 0603';e['note']='523kΩ并联2.7MΩ，下臂100kΩ，目标档名义5.38132V；提高ISNSN>=5V静态余量。FB取Hall后BAL_BUF_P，ISNSN取Hall前BAL_BUF_SENSE。基准、漏电、纹波及动态误差另见当前计算。'
  if r=='D_BAL_TVS1':
   exact(e,'SMBJ12CA-13-F');e['value']='SMBJ12CA-13-F 12V bidirectional';e['note']='12V VRWM双向TVS跨L节点，替换VRWM不足的6V候选；正常差分最高约6.46V。故障主要经四MOS体二极管回灌本地25V MLCC，TVS提供差分过冲钳位。单次理想L能量0.422mJ未含延迟/寄生；脉冲和重复温升留首板，非已验收。'
 if p.name!='共享电感功率级连接.json':
  refs={e['ref'] for e in es}
  es.extend(copy.deepcopy(e) for e in adds if e['ref'] not in refs)
 write(p,j)
# 防止历史阻容脚本重新写回2.43M。
p=C/'通用阻容官方库检索.json';j=read(p);j['RC0603FR-072M7L']=cache['RC0603FR-072M7L'];write(p,j)
p=C/'通用阻容原位落实.py';t=p.read_text('utf-8').replace('RC0603FR-072M43L','RC0603FR-072M7L').replace('2.43MΩ','2.7MΩ').replace('5.3037V','5.38132V').replace('430.37kΩ','438.132kΩ');p.write_text(t,encoding='utf-8')
p=C/'通用阻容落实记录.json';j=read(p)
for e in j['parts']:
 if e['ref']=='R_BAL_FB_EXTRA':e.update(mpn='RC0603FR-072M7L',value='2.7MΩ ±1%',device=cache['RC0603FR-072M7L'][0])
write(p,j)
p=C/'原生更新.py';t=p.read_text('utf-8').replace("('L_BAL_PWR1','R_BREG_FEED')","('L_BAL_PWR1','R_BREG_FEED','D_BAL_TVS1')");p.write_text(t,encoding='utf-8')
write(C/'Layout设计收尾变更.json',{'date':'2026-10-13','addedRefs':[e['ref'] for e in adds],'changed':['U_BAL_CURRENT','R_BAL_AVG1','U_BAL_CTRL1','U_BAL_AND2','R_BAL_FB_EXTRA','D_BAL_TVS1'],'scope':'Schematic source changes; native save/export recorded separately; no PCB or tests.'})
print('source changes saved; added',len(adds))
