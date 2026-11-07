from eda import *
b=json.loads((ROOT/'检查/CAD施工绑定.json').read_text(encoding='utf8'))
c=json.loads((ROOT/'原理图准备/原理图连接契约.json').read_text(encoding='utf8'))
carriers={'C_VM_BULK':('464a39e10b2f40069f6ec6971ccb1c57','420HXW82MEFR12.5X40','09d14ec2708540d59d6cbd7768c49d80'),'C_5V_OUT':('b51ff2cc375e4639953f603e51ed432f','470uF/16V-C8X12DIP','5160339178af4a8bb8df557483bc12da'),'C_5V_IN':('5c7773e542614272ae97f01857028948','EEEFK2A330P','2bf37764b06b487d8936ca28c6957164'),'D_5V':('3b940d90ffaa464aaa06f653c1ac2744','SS310SMC','069b625b09bd496eb2ed424307f3e157')}
for p in c['parts']:
 if p['ref'] in carriers:
  d,m,f=carriers[p['ref']];b[p['ref']]={**p,'deviceUuid':d,'carrierMpn':m,'footprintUuid':f,'binding':'geometry_carrier_not_procurement_replacement'}
save('CAD施工绑定.json',b)
put('pumpParts',[b[r] for r in carriers])
print(invoke("const cs=await eda.sch_PrimitiveComponent.getAll();const d=cs.find(c=>c.getState_Designator()==='D_5V');if(d)await eda.sch_PrimitiveComponent.delete(d.getState_PrimitiveId());globalThis.buildStart=0;return true;"))
r=invoke((ROOT/'EDA自动化/放置外围.js').read_text(encoding='utf-8-sig'));save('缺失外围实建引脚.json',r)
print(json.dumps(r,ensure_ascii=True)[:5000])
