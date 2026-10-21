"""根据真实官方库查询落实阻容；不运行电气测试或创建PCB。"""
import json, re
from pathlib import Path
B=Path(__file__).resolve().parents[1]; C=B/'检查'; S=B/'原理图准备'
def read(p): return json.loads(p.read_text(encoding='utf-8-sig'))
def write(p,v): p.write_text(json.dumps(v,ensure_ascii=False,indent=2),encoding='utf-8')
proposal=read(C/'通用阻容选型建议.json')['parts']
cache=read(C/'通用阻容官方库检索.json')
if 'results' in cache: cache=cache['results']
e96=read(C/'E96候选库检索.json')
cache={**e96,**cache}
choices={p['ref']:dict(p) for p in proposal}
changes={
 'R_BAL_RT1':('RC0603FR-07158KL','158kΩ','±1%'),
 'R_BAL_FB_BOT1':('RT0603BRD07100KL','100kΩ','±0.1%'),
 'R_BAL_FB_TOP':('RT0603BRD07523KL','523kΩ','±0.1%'),
 'R_BAL_FB_EXTRA':('RC0603FR-072M7L','2.7MΩ','±1%'),
 'R_BAL_TEMP_HOT_REF_T':('RT0603BRD0733K2L','33.2kΩ','±0.1%'),
 'R_BAL_TEMP_COLD_REF_T':('RT0603BRD0760K4L','60.4kΩ','±0.1%'),
 'R_BAL_SS_REF_T':('RT0603BRD07348KL','348kΩ','±0.1%')}
choices['C_BAL_VCC1_1'].update(mpn='CL21B225KAFNNNE',manufacturer='Samsung',value='2.2µF',voltage='25V X7R',package='0805',tolerance='±10%',sourceUrl='https://www.lcsc.com/product-detail/C19110.html')
choices['C_BAL_COMP1'].update(mpn='C2012X7R1E335KT000N',manufacturer='TDK',value='3.3µF',voltage='25V X7R',package='0805',tolerance='±10%',sourceUrl='https://www.lcsc.com/product-detail/C342789.html')
for prefix in ('R_BAL_CAP2_OV','R_BAL_CAP3_OV','R_BAL_BUF_OV'):
 for suffix in ('REF','GND'):
  changes[prefix+'_'+suffix]=('RT0603BRD07100KL','100kΩ','±0.1%')
for ref,(mpn,value,tol) in changes.items(): choices[ref].update(mpn=mpn,value=value,tolerance=tol)
selected={}; pending=[]
for ref,p in choices.items():
 items=cache.get(p['mpn'],[])
 if isinstance(items,dict): items=items.get('items',items.get('results',[]))
 d=next((d for d in items if d.get('manufacturerId')==p['mpn'] or d.get('name')==p['mpn']),None)
 if not d or not d.get('footprintUuid'): pending.append({'ref':ref,'mpn':p['mpn']}); continue
 selected[ref]=(p,d)
note_updates={
 'R_BAL_RT1':'TI RRT≈1/(32pF·fSW)；158kΩ对应197.78kHz标称，仅电阻±1%时195.82–199.78kHz。RT不可并联电容；振荡器自身容差、开关损耗及最小导通时间另核。',
 'R_BAL_FB_BOT1':'FB下臂100kΩ改用已有0.1%薄膜器件，复用型号；523kΩ上臂也为0.1%，目标阶段并联2.7MΩ/1%。标称BUF目标6.23/5.38132V。须计入基准±3%、电阻、FB最大60nA、档位MOS漏电、纹波和反向shunt压降；共模余量见阻值归并计算，尚未动态验收。',
 'R_BAL_FB_TOP':'默认上臂523kΩ改为0.1%薄膜，替换唯一523kΩ/1%型号，不增加类型。与100kΩ/0.1%下臂形成6.23V标称反馈；目标档并联2.7MΩ/1%，约5.38132V。基准/FB漏电/开关漏电/温漂/动态误差仍须核。',
 'R_BAL_FB_EXTRA':'2.7MΩ与523kΩ并联约438.132kΩ，100kΩ下臂、1V基准使目标阶段BUF反馈标称5.38132V；反馈在27mΩ shunt后。电阻±1%及反馈基准±3%均须计入。',
 'R_BAL_SS_REF_T':'2.5V经348kΩ/100kΩ分压约0.5580V，对应BAL_SS约1.1161V。门限较旧值略高，补偿/SS完成与比较器误差须在样机核定。',
 'R_BAL_SS_REF_B':'2.5V经348kΩ/100kΩ分压约0.5580V，对应BAL_SS约1.1161V。',
 'R_BAL_TEMP_HOT_REF_T':'同源5V经33.2kΩ/10kΩ产生HOT参考；10k/B3435 NTC需计入TEMP_HALF的200kΩ并联负载，标称停止温度见阻值归并计算。',
 'R_BAL_TEMP_HOT_REF_B':'同源5V经33.2kΩ/10kΩ产生HOT参考，标称停止温度含200kΩ传感分压负载。',
 'R_BAL_TEMP_COLD_REF_T':'同源浮地5V经60.4kΩ/100kΩ分压，计入TEMP_HALF的200kΩ并联负载后，10kΩ/B3435 NTC标称约10.46°C冷停；旧54.9kΩ仅约8.03°C，已修正。误差、热滞后及比较器回差另核。',
 'R_BAL_TEMP_COLD_REF_B':'同源浮地5V经60.4kΩ/100kΩ分压，含200kΩ传感负载，标称约10.46°C冷停。',
 'R_BAL_TEMP_OPEN_REF_T':'浮地5V经110kΩ/90kΩ分压得到2.25V开路阈值；TEMP_HALF分压对传感节点有200kΩ负载，开路检出仍须计入全部误差。',
 'R_BAL_TEMP_OPEN_REF_B':'浮地5V经110kΩ/90kΩ分压得到2.25V开路阈值。',
 'NTC_BAL_FLOAT':'贴近缓冲电容组热点，仅浮地硬件温度状态，无MCU温度数值；HOT参考33.2kΩ/10kΩ、COLD参考60.4kΩ/100kΩ。计算须包含TEMP_HALF两只100kΩ的200kΩ负载，标称阈值见阻值归并计算。全部9组电芯软件默认15–40°C允许均衡；B0域板温NTC独立，不可直连浮地NTC。'}
for prefix in ('R_BAL_CAP2_OV','R_BAL_CAP3_OV','R_BAL_BUF_OV'):
 for suffix in ('REF','GND','LOW','T','B'):
  note_updates[prefix+'_'+suffix]='逐bank差分OV检测：200kΩ/100kΩ检测Sk/3，三支100kΩ参考(Sprev+2.5V)/3，标称该bank过压2.500V。REF和GND必须成对从98kΩ/102kΩ改为100kΩ，减少料号并消除标称共模项；误差、延时及残余电感能量待核。'
paths=[S/'共享电感功率级连接.json',S/'共享电感控制保护连接.json',C/'原理图构建清单.json',C/'Layout器件映射.json',C/'当前施工输入.json',C/'共享电感施工输入.json']
evidence=[]
for path in paths:
 j=read(path); es=j if isinstance(j,list) else j['components']
 for e in es:
  ref=e['ref']
  if ref in selected:
   p,d=selected[ref]; note=e.get('note','')
   note=re.sub(r'\s*该libraryKey仅为.*?(?:询价。|不是采购MPN。)', '', note)
   note=re.sub(r'\s*选定MPN .*$', '', note)
   note=note_updates.get(ref,note)
   note+=' 选定MPN '+p['mpn']+'；'+p['voltage']+'；'+p['tolerance']+'；'+p['package']+'。官方库匹配不代替焊盘审核、偏压容量/功率降额、报价或电气验收。'
   e.update(mpn=p['mpn'],value=p['value']+' '+p['tolerance']+' '+p['voltage'],note=note,libraryKey=d['name'],libUuid=d['uuid'],libraryUuid=d['libraryUuid'],libType='3')
   if 'xy' in e or 'x' in e:
    e.update(keepExisting=False,unresolved=[],carrierReason='Selected exact passive device; human pad and DC-bias/derating review required.',nativeFootprint={'uuid':d['footprintUuid'],'libraryUuid':d['libraryUuid'],'name':d['footprintName']})
   if path.name.startswith('共享电感') and path.parent==S:
    evidence.append({'ref':ref,'mpn':p['mpn'],'value':e['value'],'previousValue':p['sourceValue'],'pageGroup':e['pageGroup'],'device':d,'source':p['sourceUrl'],'engineeringFlag':p.get('flag')})
  elif ref in note_updates: e['note']=note_updates[ref]
 write(path,j)
write(C/'通用阻容落实记录.json',{'date':'2026-10-13','selectedCount':len(selected),'distinctMpns':len({p['mpn'] for p,d in selected.values()}),'pending':pending,'parts':evidence,'scope':'Source and construction input changes; native save recorded separately; no electrical tests.'})
print('selected',len(selected),'distinct',len({p['mpn'] for p,d in selected.values()}),'pending',pending)
