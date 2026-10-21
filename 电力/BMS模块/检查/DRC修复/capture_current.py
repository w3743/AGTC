import sys,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tmp'));import drc_fix as d
s=json.loads((d.E/'session.json').read_text())['sessionId']
code=r"""
const all=await eda.sch_PrimitiveComponent.getAll();const rows=[],ports=[];
for(const c of all){const type=c.getState_ComponentType();if(type!=='part'&&type!=='netport')continue;
const pins=(await eda.sch_PrimitiveComponent.getAllPinsByPrimitiveId(c.getState_PrimitiveId())||[]).map(p=>({number:p.getState_PinNumber(),name:p.getState_PinName(),x:p.getState_X(),y:p.getState_Y(),rotation:p.getState_Rotation(),nc:p.getState_NoConnected()}));
const r={ref:c.getState_Designator(),id:c.getState_PrimitiveId(),x:c.getState_X(),y:c.getState_Y(),rotation:c.getState_Rotation(),mirror:c.getState_Mirror(),name:c.getState_Name(),mpn:c.getState_ManufacturerId(),manufacturer:c.getState_Manufacturer(),supplier:c.getState_Supplier(),supplierId:c.getState_SupplierId(),other:c.getState_OtherProperty(),component:c.getState_Component(),symbol:c.getState_Symbol(),footprint:c.getState_Footprint(),bom:c.getState_AddIntoBom(),pcb:c.getState_AddIntoPcb(),net:c.getState_Net(),pins};
(type==='part'?rows:ports).push(r);}
const wires=(await eda.sch_PrimitiveWire.getAll()).map(w=>({primitiveId:w.getState_PrimitiveId(),net:w.getState_Net(),line:w.getState_Line()}));
const texts=(await eda.sch_PrimitiveText.getAll()).map(t=>({primitiveId:t.getState_PrimitiveId(),x:t.getState_X(),y:t.getState_Y(),content:t.getState_Content()}));
return {rows,ports,wires,texts,drc:await eda.sch_Drc.check(true,false,true)};
"""
r=d.call(s,code,sys.argv[1] if len(sys.argv)>1 else 'current-snapshot')
print(len(r['rows']),sum(len(p['pins']) for p in r['rows']),len(r['ports']),r['drc'])
