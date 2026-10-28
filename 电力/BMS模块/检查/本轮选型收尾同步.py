"""原位同步电感与REG1前级阻值规格；不执行电气测试。"""
import json
from pathlib import Path
B=Path(__file__).resolve().parents[1]
C=B/'检查'
def read(p): return json.loads(p.read_text(encoding='utf-8-sig'))
def write(p,j): p.write_text(json.dumps(j,ensure_ascii=False,indent=2),encoding='utf-8')
selected={
 'L_BAL_PWR1':('SRP1265A-220M','电感收尾库检索.json',
  '22uH +/-20%; Irms 9A; Isat 12A',
  'Bourns ordinary shielded inductor; official DCR typ32.5/max37mOhm, Irms9A at40K rise, Isat12A at20% L reduction. Source: https://www.bourns.com/docs/Product-Datasheets/SRP1265A.pdf . Previous YSPI DCR5.8mOhm note was incorrect (Irms5.8A, DCRmax44mOhm). Peak limit approximately5.7A before delay/spikes; final RMS/core loss, thermal and physical pads remain human review.'),
 'R_BREG_FEED':('RC2512FK-7W300RL','BREG2W库检索.json',
  '300Ohm 1% 2W 2512',
  'REG0 collector feed, retain300Ohm and upgrade0603 to2512/2W. REG1 aggregate design budget<=30mA; at30mA resistor loss<=0.273W including1% and at45mA<=0.614W;80mA short-circuit current corner gives1.94W before temperature derating. Fan LED is a REG1 load; do not add MCU/fan power here. Q_BREG thermal/SOA, REGIN>=4.1V and startup remain human hardware acceptance. Source: https://www.yageogroup.com/component-documentation/download/specsheet/RC2512FK-7W300RL .')}
library={}
for ref,(mpn,cache,value,note) in selected.items():
 d=next(x for x in read(C/cache) if x['name']==mpn)
 library[ref]=(d,value,note)
paths=[B/'原理图准备'/'共享电感功率级连接.json',B/'原理图准备'/'AFE元件连接.json',C/'原理图构建清单.json',C/'Layout器件映射.json',C/'当前施工输入.json']
for p in paths:
 j=read(p);es=j if isinstance(j,list) else j['components']
 for e in es:
  if e['ref'] not in library: continue
  d,value,note=library[e['ref']]
  e.update(mpn=d['manufacturerId'],value=value,note=note,libraryKey=d['name'],libUuid=d['uuid'],libraryUuid=d['libraryUuid'],libType='3')
  if 'xy' in e or 'x' in e:
   e.update(keepExisting=False,unresolved=[],carrierReason='Selected exact library device; human pad review required.',nativeFootprint={'uuid':d['footprintUuid'],'libraryUuid':d['libraryUuid'],'name':d['footprintName']})
 write(p,j)
print('updated inductor and BREG feed in current source/manifest/layout inputs')
