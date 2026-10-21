"""Read-only geometry checks before any official CAD drawing mutation."""
from pathlib import Path
import json

P=Path(__file__).resolve().parent
O=P/'原理图逻辑重排'
def on_segment(p,a,b):
    return (a[0]==b[0]==p[0] and min(a[1],b[1])<=p[1]<=max(a[1],b[1])) or (a[1]==b[1]==p[1] and min(a[0],b[0])<=p[0]<=max(a[0],b[0]))

def check(plan,geometry,manifest):
    # Official symbol rotations can leave <1e-12 coordinate tails. Normalize
    # below one ten-millionth of a drawing unit, not to the electrical grid.
    def point(x,y):return (round(float(x),7),round(float(y),7))
    nets={r['ref']:{str(p):n for p,n in r['pins'].items()} for r in manifest}
    pins=[]
    for row in geometry['rows']:
        for p in row['pins']:
            n=nets.get(row['ref'],{}).get(str(p['number']))
            pins.append((row['ref'],p['number'],n if not p.get('nc') else None,point(p['x'],p['y'])))
    segments=[];bad_orthogonal=[]
    for i,w in enumerate(plan['wires']):
        pts=[point(x,y) for x,y in zip(w['line'][::2],w['line'][1::2])]
        for a,b in zip(pts,pts[1:]):
            if a==b:continue
            if a[0]!=b[0] and a[1]!=b[1]:bad_orthogonal.append(i)
            segments.append((w['net'],a,b,i))
    missing=[];touch=[]
    for ref,pn,net,p in pins:
        connected=[]
        for n,a,b,i in segments:
            if on_segment(p,a,b):connected.append(n)
        for l in plan['labels']:
            if point(l['x'],l['y'])==p:connected.append(l['net'])
        if net and net not in connected:missing.append({'ref':ref,'pin':pn,'net':net,'point':p})
        if any(n!=net for n in connected):touch.append({'ref':ref,'pin':pn,'net':net,'point':p,'touching':sorted(set(connected))})
    endpoints=[];endpoint_seen=set()
    for n,a,b,i in segments:
        for endpoint in (a,b):
            for nn,c,d,j in segments:
                if nn!=n and on_segment(endpoint,c,d):
                    key=(min(i,j),max(i,j),endpoint)
                    if key not in endpoint_seen:
                        endpoint_seen.add(key);endpoints.append({'wire':i,'net':n,'point':endpoint,'otherWire':j,'otherNet':nn})
    dangling_labels=[]
    for l in plan['labels']:
        pt=point(l['x'],l['y'])
        if not any(n==l['net'] and on_segment(pt,a,b) for n,a,b,i in segments) and not any(n==l['net'] and p==pt for ref,pn,n,p in pins):dangling_labels.append(l)
    return {'counts':{'pins':len(pins),'segments':len(segments),'wires':len(plan['wires']),'labels':len(plan['labels'])},'nonOrthogonal':bad_orthogonal,'missingPins':missing,'foreignNetAtPins':touch,'foreignNetAtWireEndpoints':endpoints,'danglingLabels':dangling_labels}

if __name__=='__main__':
    r=check(json.loads((O/'绘图计划.json').read_text(encoding='utf8')),json.loads((O/'移动后几何.json').read_text(encoding='utf8')),json.loads((P/'当前施工输入.json').read_text(encoding='utf8')))
    (O/'绘图几何校验.json').write_text(json.dumps(r,ensure_ascii=False,indent=2),encoding='utf8')
    print({k:len(v) if isinstance(v,list) else v for k,v in r.items()})
