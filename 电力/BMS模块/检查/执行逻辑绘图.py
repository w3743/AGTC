"""Apply explicit offline drawing plans through the official editor API only."""
from pathlib import Path
import importlib.util,json,re,sys,time
sys.path.insert(0,str(Path(__file__).resolve().parent))
from 原理图逻辑整理 import invoke,write,O,M

def module(filename):
    spec=importlib.util.spec_from_file_location(filename,M/'检查/tmp'/filename)
    obj=importlib.util.module_from_spec(spec);spec.loader.exec_module(obj);return obj

def load(name):return json.loads((O/name).read_text(encoding='utf8'))
def batches(a,n=25):
    for i in range(0,len(a),n):yield i,a[i:i+n]
def js(value):return json.dumps(value,ensure_ascii=False,separators=(',',':'))
def plans():
    a=module('主电路绘图规划.py');b=module('均衡电路路由.py')
    return a,b

if __name__=='__main__':
    mode=sys.argv[1]
    if mode=='move':
        mods=plans();pos={}
        for mod in mods:
            p=mod.placements() if callable(mod.placements) else mod.placements
            assert not (pos.keys() & p.keys()),pos.keys() & p.keys()
            pos.update(p)
        old={r['ref']:r for r in load('修改前几何.json')['rows']}
        assert set(old)==set(pos),(sorted(old.keys()-pos.keys()),sorted(pos.keys()-old.keys()))
        write('目标器件位置.json',pos)
        rows=[dict(ref=r,id=old[r]['id'],**p) for r,p in pos.items()]
        for i,chunk in batches(rows):
            r=invoke('const rows='+js(chunk)+';for(const p of rows){const c=await eda.sch_PrimitiveComponent.get(p.id);if(!c)throw Error(p.ref);const a=c.toAsync();a.setState_X(p.x);a.setState_Y(p.y);a.setState_Rotation(p.rotation??0);a.setState_Mirror(p.mirror??false);await a.done();}return {moved:rows.length};')
            write('移动批次_'+str(i)+'.json',r);print(i,r,flush=True)
    elif mode=='plan':
        pinmap=load('移动后几何.json');manifest=json.loads((M/'检查/当前施工输入.json').read_text(encoding='utf8'))
        merged={key:[] for key in ['wires','labels','texts','boxes']}
        for mod in plans():
            p=mod.build_routes(pinmap,manifest)
            for key in merged:merged[key].extend(p.get(key,[]))
        for w in merged['wires']:
            line=w['line'];assert len(line)>=4 and len(line)%2==0,w
            pts=list(zip(line[::2],line[1::2]))
            assert all(a[0]==b[0] or a[1]==b[1] for a,b in zip(pts,pts[1:])),w
        write('绘图计划.json',merged);print({k:len(v) for k,v in merged.items()})
    elif mode=='draw':
        p=load('绘图计划.json')
        for key in ['wires','labels','texts','boxes']:
            for i,chunk in batches(p[key],35):
                if key=='wires':
                    call='await eda.sch_PrimitiveWire.create(p.line,p.net,null,p.width??1)'
                elif key=='labels':
                    call="await eda.sch_PrimitiveComponent.createNetPort(p.direction??'BI',p.net,p.x,p.y,p.rotation??0,false)"
                elif key=='texts':
                    call="await eda.sch_PrimitiveText.create(p.x,p.y,p.content??p.text,0,p.color??'#17365D',null,p.size??p.fontSize??9,p.bold??false)"
                else:
                    call="await eda.sch_PrimitiveRectangle.create(p.x??p.topLeftX,p.y??p.topLeftY,p.width,p.height,0,0,'#7190AB',null,1)"
                r=invoke('const rows='+js(chunk)+';const failed=[];for(let i=0;i<rows.length;i++){const p=rows[i];try{const x='+call+';if(!x)failed.push({i,p,reason:"undefined"});}catch(e){failed.push({i,p,reason:String(e)});}}return {count:rows.length,failed};')
                write(f'绘制_{key}_{i}.json',r);print(key,i,r,flush=True)
                assert not r['failed'],r
    elif mode=='attributes':
        # Attribute coordinates follow the moved parent automatically. Format only
        # the displayed identity/value; do not touch procurement/library fields.
        rows=load('移动后几何.json')['rows']
        for i,chunk in batches(rows):
            r=invoke('const rows='+js([{'ref':r['ref'],'id':r['id']} for r in chunk])+';let count=0;for(const p of rows){for(const v of await eda.sch_PrimitiveAttribute.getAll(p.id)){if(["Name","Designator"].includes(v.getState_Key())){const a=v.toAsync();a.setState_Rotation(0);await a.done();count++;}}}return {count};')
            write('属性批次_'+str(i)+'.json',r);print(i,r,flush=True)
    elif mode=='format':
        # Preserve procurement identity and native Name values. Show compact
        # reading values as separate text; hide only the long display Name.
        rows=load('移动后几何.json')['rows']
        displays=[]
        for row in rows:
            ref=row['ref'];name=row.get('name','');mpn=row.get('mpn','')
            if ref.startswith(('R_','RSHUNT','C_','NTC_')):
                compact=name.split(' ')[0]
                if compact.replace('.','').isdigit():compact+='Ω'
            else:compact=mpn or name
            if len(compact)>28:compact=compact[:28]
            xs=[p['x'] for p in row['pins']];ys=[p['y'] for p in row['pins']]
            small=ref.startswith(('R_','RSHUNT','C_','NTC_','L_'))
            xx=row['x']-18 if small else min(xs)
            top=row['y']+18 if small else max(ys)+16
            bottom=row['y']-19 if small else min(ys)-20
            displays.append(dict(ref=ref,id=row['id'],name=name,display=compact,x=xx,top=top,bottom=bottom))
        write('显示标注计划.json',displays)
        for i,chunk in batches(displays):
            r=invoke('const rows='+js(chunk)+';let count=0;for(const p of rows){for(const v of await eda.sch_PrimitiveAttribute.getAll(p.id)){const k=v.getState_Key();if(k==="Name"){const a=v.toAsync();a.setState_ValueVisible(false);await a.done();}else if(k==="Designator"){const a=v.toAsync();a.setState_Rotation(0);a.setState_X(p.x);a.setState_Y(p.top);a.setState_FontSize(6.75);a.setState_AlignMode(3);await a.done();}}await eda.sch_PrimitiveText.create(p.x,p.bottom,p.display,0,"#28548A",null,6.75,false);count++;}return {count};')
            write('标注格式批次_'+str(i)+'.json',r);print(i,r,flush=True)
    else:raise ValueError(mode)
