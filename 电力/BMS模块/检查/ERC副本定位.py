"""仅在记录的隔离临时副本中通过官方API缩小ERC范围。"""
import json,pathlib,sys
from 原生更新 import invoke
root=pathlib.Path(__file__).resolve().parent
pages=json.loads((root/'原理图页面映射.json').read_text(encoding='utf8'))
invoke('await eda.dmt_EditorControl.openDocument('+json.dumps(next(iter(pages.values())))+');return true;')
results=[{'stage':'baseline','result':invoke('return await eda.sch_Drc.check(true,false,true);')}]
for group,pg in pages.items():
    if not group.startswith('AFE_LTC_BAL_CH'):continue
    code='await eda.dmt_EditorControl.openDocument('+json.dumps(pg)+');'
    code+=r'''
const removed=[];
for(const c of (await eda.sch_PrimitiveComponent.getAll()).filter(Boolean)){
 const ref=c.getState_Designator();
 if(String(ref).startsWith('T_BAL')){removed.push(ref);await eda.sch_PrimitiveComponent.delete(c.getState_PrimitiveId());}
}
return {removed,saved:await eda.sch_Document.save(),result:await eda.sch_Drc.check(true,false,true)};
'''
    result={'stage':group,**invoke(code)};results.append(result);print(group,result['result'],flush=True)
(root/'ERC副本移除磁件结果.json').write_text(json.dumps(results,ensure_ascii=False,indent=2),encoding='utf8')
