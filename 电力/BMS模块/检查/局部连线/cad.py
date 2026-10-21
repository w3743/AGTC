"""Official API helpers for local schematic wiring; no alternate CAD export."""
from pathlib import Path
import ast, base64, json, sys
E = Path(__file__).resolve().parent
M = E.parents[1]
sys.path.insert(0, str(M/'检查/tmp'))
import drc_fix as d

def read(p): return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def write(p,v): Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def call(code): return d.h.invoke(code,session=read(E/'session.json')['sessionId'])
def capture(name):
    tree=ast.parse((M/'检查/DRC修复/capture_current.py').read_text(encoding='utf-8'))
    code=next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='code' for t in n.targets))
    wrapped='const captured=await(async()=>{'+code+'})();return JSON.stringify(captured).replace(/[^\\x00-\\x7f]/g,c=>"\\\\u"+c.charCodeAt(0).toString(16).padStart(4,"0"));'
    v=json.loads(call(wrapped));write(E/(name+'.json'),v)
    print({'parts':len(v['rows']),'ports':len(v['ports']),'wires':len(v['wires']),'drc':v['drc']})
    return v

if __name__=='__main__':
    mode=sys.argv[1]
    if mode=='capture': capture(sys.argv[2])
    elif mode=='call': print(call(sys.stdin.read()))
    elif mode=='view':
        box=list(map(float,sys.argv[3:7]))
        if box: print(call('return await eda.dmt_EditorControl.zoomToRegion('+','.join(map(str,box))+');'))
        v=call('const b=await eda.dmt_EditorControl.getCurrentRenderedAreaImage();if(!b)return null;const a=new Uint8Array(await b.arrayBuffer());let x="";for(const n of a)x+=String.fromCharCode(n);return {type:b.type,data:btoa(x)};')
        if v: (E/(sys.argv[2]+'.png')).write_bytes(base64.b64decode(v['data']))
        print({'image':sys.argv[2],'available':bool(v)})
