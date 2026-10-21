"""仅通过官方 API 同步九个差分电容的设计备注，不修改连接或库关联。"""
import json
from pathlib import Path
from 原生更新 import invoke

root = Path(__file__).resolve().parent
entries = json.loads((root / '当前施工输入.json').read_text(encoding='utf-8-sig'))
pages = json.loads((root / '原理图页面映射.json').read_text(encoding='utf-8-sig'))
targets = [e for e in entries if e.get('ref', '').startswith(('C_LTC1_C', 'C_LTC2_C'))]
assert len(targets) == 9
assert all('≥30µF' in e['note'] and '26.667µF' in e['note'] for e in targets)
results = []
for group in dict.fromkeys(e['pageGroup'] for e in targets):
    notes = {e['ref']: '100uF+10uF per adjacent cell pair; effective total >=30uF at bias/temp/tolerance/aging corners. 55mV/24.75mOhm=2.222A; Cmin=(Ichg+Idsg)*6us/1V=26.667uF. See BOM.csv and tap-connection review; hardware proof pending.' for e in targets if e['pageGroup'] == group}
    code = 'const notes=' + json.dumps(notes, ensure_ascii=True) + ';'
    code += 'await eda.dmt_EditorControl.openDocument(' + json.dumps(pages[group]) + ');'
    code += r'''
const changed=[];
for(const c of await eda.sch_PrimitiveComponent.getAll()){
 const ref=c.getState_Designator();if(!Object.hasOwn(notes,ref))continue;
 c.setState_OtherProperty({...c.getState_OtherProperty(),'Design note':notes[ref]});
 await c.done();
 changed.push(ref);
}
if(changed.length!==Object.keys(notes).length)throw Error('missing note target');
const saved=await eda.sch_Document.save();
const readback=(await eda.sch_PrimitiveComponent.getAll())
 .filter(c=>Object.hasOwn(notes,c.getState_Designator()))
 .map(c=>({ref:c.getState_Designator(),note:c.getState_OtherProperty()['Design note']}));
return {saved,changed,readback};
'''
    result = invoke(code)
    (root / '差分电容原生注释最近调用.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    assert result['saved'] and all(row['note'] == notes[row['ref']] for row in result['readback'])
    results.append({'pageGroup': group, **result})
    print(group, len(result['changed']), flush=True)
(root / '差分电容原生注释同步.json').write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding='utf-8')
