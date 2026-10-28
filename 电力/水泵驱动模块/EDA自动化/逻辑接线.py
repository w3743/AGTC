"""Explicit current-page re-layout runner. Requires a verified archive baseline."""
from eda import *
contract=json.loads((ROOT/'原理图准备/原理图连接契约.json').read_text(encoding='utf8'))
put('pumpLayoutContract',contract)
code=(ROOT/'EDA自动化/逻辑接线.js').read_text(encoding='utf8')
logs=[]
for stage in range(5):
 invoke(f'globalThis.pumpLayoutStage={stage};return true;')
 r=invoke(code);logs.append(r)
 if stage==0:save('逻辑接线坐标.json',r)
 print('stage',stage,'parts',r.get('parts'),'wires',r.get('wireCount'),flush=True)
save('逻辑接线施工.json',logs)
