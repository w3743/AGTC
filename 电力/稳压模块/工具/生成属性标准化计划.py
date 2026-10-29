"""按当前原生缓存的确切商城编号规范显示名、厂家与MPN拼写，不改变选型。"""
from pathlib import Path
import importlib.util,collections,json
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('shared',ROOT.parent/'元件封装库/工具/更新元件库.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
s=m.parse_source(ROOT/'eda工程文件/sc-regulator/sch/sc-regulator/01 稳压模块全部电路.esch2','稳压模块','native')
dev={d['local_uuid']:d['meta'] for d in s['sections'] if d['doc_type']=='DEVICE'}
records={}
for rec in s['pages'][0]['records']:
 h,v=rec['header'],rec['value'];i=h.get('id')
 if h['type']=='DELETE':records.pop(i,None)
 elif i:
  if i not in records:records[i]={'h':h,'v':{}}
  records[i]['v'].update(v)
attrs=collections.defaultdict(dict)
for rec in records.values():
 h,v=rec['h'],rec['v']
 if h['type']=='ATTR' and v.get('key'):attrs[v.get('parentId')][v['key']]=v.get('value')
rows=[];issues=[]
for rec in records.values():
 h,v=rec['h'],rec['v']
 if h['type']!='COMPONENT':continue
 a=attrs[h['id']];d=dev.get(a.get('Device'),{}).get('attributes',{});ref=a.get('Designator')
 if not ref or not a.get('Supplier Part'):continue
 if a['Supplier Part']!=d.get('Supplier Part'):
  issues.append({'ref':ref,'reason':'SUPPLIER_CODE_DIFFERS_FROM_CARRIER','instance':a['Supplier Part'],'carrier':d.get('Supplier Part')});continue
 if ''.join(str(a.get('Manufacturer Part','')).split())!=''.join(str(d.get('Manufacturer Part','')).split()):
  issues.append({'ref':ref,'reason':'REAL_MPN_DIFFERENCE','instance':a.get('Manufacturer Part'),'carrier':d.get('Manufacturer Part')});continue
 rows.append({'ref':ref,'supplier_part':a['Supplier Part'],'mpn':d['Manufacturer Part'],
              'manufacturer':d.get('Manufacturer',''),'supplier':d.get('Supplier','LCSC'),
              'name':d.get('Name') or d.get('Value') or d['Manufacturer Part'],
              'before_name':a.get('Name'),'before_mpn':a.get('Manufacturer Part')})
out={'status':'CACHE_EXACT_CODE_AND_MPN_VERIFIED','note':'只规范同一商城编号/同一MPN（忽略空格）的显示及厂家拼写；不改绑定封装，不取消DRC规则。',
     'components':rows,'issues_not_blindly_changed':issues}
(ROOT/'检查/器件属性标准化计划.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps({'count':len(rows),'issues':issues},ensure_ascii=False))
