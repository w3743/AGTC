"""Read-only point/contact check of a proposed local drawing plan."""
import sys
import cad
sys.path.insert(0,str(cad.M/'检查/DRC修复'))
from verify_semantic_geometry import point_on_segment,endpoint_contacts,strict_interior_intersection
b=cad.read(cad.E/'baseline-snapshot.json');p=cad.read(cad.E/(sys.argv[1] if len(sys.argv)>1 else 'afe-plan.json'))
manifest={r['ref']:r['pins'] for r in cad.read(cad.M/'检查/当前施工输入.json')}
points=[];segments=[]
moves={r['id']:r for r in p.get('move',[])}
for r in b['rows']:
 for pin in r['pins']:
  xy=(pin['x'],pin['y'])
  if r['id'] in moves:
   m=moves[r['id']];dx,dy=pin['x']-r['x'],pin['y']-r['y']
   if m['rotation']==270:xy=(m['x']+dy,m['y']-dx)
   else:xy=(m['x']+dx,m['y']+dy)
  points.append((xy,manifest[r['ref']].get(pin['number'],'NC'),r['ref']+'.'+pin['number']))
for q in b['ports']:
 if q['id'] not in p.get('deletePorts',[]):points.append(((q['x'],q['y']),q['net'],'port '+q['id']))
for w in b['wires']:
 if w['primitiveId'] in p.get('deleteWires',[]):continue
 for i in range(0,len(w['line']),4):segments.append((tuple(w['line'][i:i+2]),tuple(w['line'][i+2:i+4]),w['net'],w['primitiveId']))
new=[]
for w in p['wires']:
 a=w.get('points',w.get('line'))
 for i in range(0,len(a)-2,2):new.append((tuple(a[i:i+2]),tuple(a[i+2:i+4]),w['net'],'new'))
bad=[];cross=[]
for i,(a,z,n,_) in enumerate(new):
 for v,m,s in points:
  if m!=n and point_on_segment(v,a,z):bad.append(('pin/port',n,a,z,m,v,s))
 for c,d,m,s in segments+new[:i]:
  if m!=n:
   hit=endpoint_contacts(a,z,c,d)
   if hit:bad.append(('endpoint',n,a,z,m,list(hit),s))
   h=strict_interior_intersection(a,z,c,d)
   if h:cross.append((n,m,h,s))
out={'conflicts':bad,'interiorCrossings':cross}
cad.write(cad.E/'plan-contact-check.json',out)
print(out)
