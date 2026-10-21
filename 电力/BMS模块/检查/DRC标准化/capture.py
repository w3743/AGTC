"""Capture a full official API readback without exporting CAD formats."""
import ast,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tmp'))
import drc_fix as d
E=Path(__file__).resolve().parent
sid=json.loads((E/sys.argv[1]).read_text(encoding='utf-8'))['sessionId']
tree=ast.parse((E.parent/'DRC修复/capture_current.py').read_text(encoding='utf-8'))
code=next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='code' for t in n.targets))
# CLI stdout can split a multibyte Chinese character at a transport chunk.
# Return ASCII JSON from inside the engine, then decode once in Python.
code='const captured=await(async()=>{'+code+'})();return JSON.stringify(captured).replace(/[^\\x00-\\x7f]/g,c=>"\\\\u"+c.charCodeAt(0).toString(16).padStart(4,"0"));'
v=json.loads(d.h.invoke(code,session=sid));(E/(sys.argv[2]+'.json')).write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps({'parts':len(v['rows']),'pins':sum(len(r['pins']) for r in v['rows']),'drc':v['drc']},ensure_ascii=False))
