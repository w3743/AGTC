"""Prepare and apply AFE drawing-only local wires via the official API."""
from pathlib import Path
import json, sys
import cad
E=cad.E
b=cad.read(E/'baseline-snapshot.json')
manifest={r['ref']:r['pins'] for r in cad.read(cad.M/'检查/当前施工输入.json')}
rows={r['ref']:r for r in b['rows']}
p={'move':[],'deletePorts':[],'deleteWires':[],'wires':[],'ports':[],'texts':[],'reason':'VC ladder and differential SRP capacitor local physical connections; electrical manifest unchanged'}
oldpoints=set()
def pt(pin): return (pin['x'],pin['y'])
def remove_at(point,nets=None):
 for q in b['ports']:
  if (q['x'],q['y'])==point and (nets is None or q['net'] in nets):p['deletePorts'].append(q['id'])
 oldpoints.add(point)
def wire(net,*points):
 p['wires'].append({'net':net,'points':[v for q in points for v in q]})
def move(ref,x,y,rotation):
 r=rows[ref]
 p['move'].append({'ref':ref,'id':r['id'],'x':x,'y':y,'rotation':rotation})
 for pin in r['pins']:remove_at(pt(pin))
 # Existing independent compact value text follows the moved component.
 ts=[t for t in b['texts'] if abs(t['x']-(r['x']-18))<2 and abs(t['y']-(r['y']-19))<2]
 if len(ts)!=1:raise ValueError((ref,'value text ambiguity',ts))
 p['texts'].append({'id':ts[0]['primitiveId'],'x':x+22,'y':y-10})

for n in range(9):
 y=1690+65*n
 move(f'C_VC{n}',2200,y-35,270)
 r=rows[f'R_VC{n}']
 remove_at((2070,y),{f'VC{n}_FILT'})
 wire(f'VC{n}_FILT',(2070,y),(2200,y),(2390-10*n,y),(2390-10*n,2115+10*n),(2460,2115+10*n))
 wire(f'VC{n}_FILT',(2200,y),(2200,y-20))
 if n:
  wire(f'VC{n-1}_FILT',(2200,y-50),(2200,y-65))
 else:
  wire('B0',(2200,y-50),(2200,y-70))
  p['ports'].append({'net':'B0','x':2200,'y':y-70})

move('C_VC16',2200,2305,270)
wire('VC16_FILT',(2070,2340),(2200,2340),(2200,2320))
remove_at((2070,2340),{'VC16_FILT'})
p['ports'].append({'net':'VC16_FILT','x':2200,'y':2340})
wire('VC15_FILT',(2200,2290),(2200,2275),(2070,2275))
wire('VC15_FILT',(2200,2275),(2290,2275),(2290,2265),(2460,2265))
remove_at((2070,2275),{'VC15_FILT'})

for pin in rows['U1']['pins']:
 net=manifest['U1'].get(pin['number'])
 if net and net.startswith('VC') and net!='VC16_FILT':remove_at(pt(pin),{net})
for y in range(2195,2256,10):
 wire('VC8_FILT',(2430,y),(2460,y))
wire('VC8_FILT',(2430,2195),(2430,2255))

move('C_SRP',2400,2080,270)
p['texts'][-1].update(x=2410,y=2072)
p['move'][-1].update(labelX=2410,labelY=2087)
p['deleteWires'].append('5f45504cad6547fd')
for y,net in [(2095,'SRP'),(2075,'SRN')]:remove_at((2460,y),{net})
wire('SRP',(2460,2095),(2445,2095),(2445,2100),(2400,2100))
wire('SRN',(2460,2075),(2445,2075),(2445,2060),(2400,2060))

for w in b['wires']:
 l=w['line']
 if l and all(tuple(l[i:i+2])==tuple(l[i+2:i+4]) for i in range(0,len(l),4)) and all(tuple(l[i:i+2]) in oldpoints for i in range(0,len(l),4)):
  p['deleteWires'].append(w['primitiveId'])
p['deletePorts']=sorted(set(p['deletePorts']));p['deleteWires']=sorted(set(p['deleteWires']))
cad.write(E/'afe-plan.json',p)
print({k:len(v) for k,v in p.items() if isinstance(v,list)})
if '--apply' in sys.argv:
 code='const p='+json.dumps(p)+';const out={moves:[],wires:[]};'
 code+='''
 for(const m of p.move){const a=(await eda.sch_PrimitiveComponent.get(m.id)).toAsync();a.setState_X(m.x);a.setState_Y(m.y);a.setState_Rotation(m.rotation);await a.done();
 const attrs=await eda.sch_PrimitiveAttribute.getAll(m.id);for(const q of attrs){if(q.getState_Key()==='Designator'){const z=q.toAsync();z.setState_X(m.labelX??m.x+22);z.setState_Y(m.labelY??m.y+8);z.setState_Rotation(0);await z.done();}}
 out.moves.push({ref:m.ref,pins:(await eda.sch_PrimitiveComponent.getAllPinsByPrimitiveId(m.id)).map(q=>({n:q.getState_PinNumber(),x:q.getState_X(),y:q.getState_Y()}))});}
 for(const t of p.texts){const a=(await eda.sch_PrimitiveText.get(t.id)).toAsync();a.setState_X(t.x);a.setState_Y(t.y);await a.done();}
 if(p.deletePorts.length)await eda.sch_PrimitiveComponent.delete(p.deletePorts);
 if(p.deleteWires.length)await eda.sch_PrimitiveWire.delete(p.deleteWires);
 for(const w of p.wires){const a=await eda.sch_PrimitiveWire.create(w.points,w.net,null,1);if(!a)throw Error('Wire creation failed '+w.net);out.wires.push(a.getState_PrimitiveId());}
 for(const q of p.ports)await eda.sch_PrimitiveComponent.createNetPort('BI',q.net,q.x,q.y,0,false);
 return out;
 '''
 result=cad.call(code);cad.write(E/'afe-applied.json',result);print(result)
