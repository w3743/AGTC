from eda import *
a=json.loads((ROOT/'检查/官方精确器件绑定.json').read_text(encoding='utf8'))
c=json.loads((ROOT/'原理图准备/原理图连接契约.json').read_text(encoding='utf8'))
rows=a['lcscLookup']+a['exactMpnSearch']
norm=lambda x:x.replace(' ','').lower()
bindings={}
for p in c['parts']:
 candidates=[v for r in rows for v in r['candidates'] if norm(v.get('manufacturerId',''))==norm(p['mpn'])]
 if p['ref']=='D_5V':candidates=[]
 if candidates:
  d=next((x for x in candidates if x['uuid']=='76bcd52e6dd845eb950d4f76382feb0e'),candidates[0]);bindings[p['ref']]={**p,'deviceUuid':d['uuid'],'carrierMpn':d['manufacturerId'],'footprintUuid':d['footprintUuid'],'binding':'exact_mpn' if d['manufacturerId']==p['mpn'] else 'mpn_spacing_alias'}
# Identical nonpolar 0603 CAD carrier; preserve procurement MPN in instance.
for p in c['parts']:
 if p['mpn']=='C1608X7R1H104K080AA':bindings[p['ref']]={**p,'deviceUuid':'f859353c7fbb498ca2652dedd5c200cd','carrierMpn':'C1608X7R1H103K080AA','footprintUuid':'76d6c29a47804db487f8a92b2621751b','binding':'geometry_carrier_0603_nonpolar'}
save('CAD施工绑定.json',bindings)
put('pumpParts',list(bindings.values()))
print('ready',len(bindings),'missing',[p['ref'] for p in c['parts'] if p['ref'] not in bindings])
