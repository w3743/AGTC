from eda import *
for name in ['CAP-TH_BD12.5-P5.00','CAP-TH_BD8.0-P3.50','EEEFK2A330P','SS310 SMC']:
 r=invoke('return await eda.lib_Device.search('+json.dumps(name)+",'0819f05c4eef4c71ace90d822a990e87',undefined,undefined,20,1);")
 save('载体搜索-'+name.replace('/','_')+'.json',r)
 print(name,[(x.get('uuid'),x.get('name'),x.get('footprintName'),x.get('footprintUuid')) for x in r][:15],flush=True)
