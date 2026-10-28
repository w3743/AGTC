from eda import *
code=(ROOT/'EDA自动化/放置外围.js').read_text(encoding='utf-8-sig')
for i in range(0,len(json.loads((ROOT/'检查/CAD施工绑定.json').read_text(encoding='utf8'))),10):
 invoke(f'globalThis.buildStart={i};return true;')
 r=invoke(code);save('外围引脚-'+str(i)+'.json',r);print('placed',i,len(r),flush=True)
