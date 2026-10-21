from eda import *
contract=json.loads((ROOT/'原理图准备/原理图连接契约.json').read_text(encoding='utf8'))
put('pumpContract',contract)
code=(ROOT/'EDA自动化/连接.js').read_text(encoding='utf-8-sig')
for i in range(0,len(contract['parts']),10):
 invoke(f'globalThis.wireStart={i};return true;')
 r=invoke(code);save('网络施工-'+str(i)+'.json',r);print('wired',i,r,flush=True)
