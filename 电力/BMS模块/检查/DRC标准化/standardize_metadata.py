"""Official API metadata standardization; preserve identity, pins and CAD carriers."""
import argparse,json,sys,re
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tmp'))
import drc_fix as d
E=Path(__file__).resolve().parent
ap=argparse.ArgumentParser();ap.add_argument('session_file');ap.add_argument('output');ap.add_argument('--nine',action='store_true');args=ap.parse_args()
sid=json.loads((E/args.session_file).read_text(encoding='utf-8'))['sessionId']
warning_refs=re.findall(r'([A-Za-z0-9_]+)\(\$', '\n'.join(json.loads((E/'input-audit.json').read_text(encoding='utf-8'))['warnings']))
assert len(warning_refs)==len(set(warning_refs))==188
code=r"""
await eda.dmt_EditorControl.openDocument('350a4904c4ec87b3');
const cs=(await eda.sch_PrimitiveComponent.getAll()).filter(c=>c.getState_ComponentType()==='part');
const ids=[...new Set(cs.map(c=>c.getState_SupplierId()).filter(x=>/^C\d+$/.test(x)))];
const ds=await eda.lib_Device.getByLcscIds(ids);const byId=Object.fromEntries(ds.filter(Boolean).map(x=>[x.supplierId,x]));
const reserved=new Set(['Name','Designator','Manufacturer','Manufacturer Part','Supplier','Supplier Part','Symbol','Footprint','3D Model','3D Model Title','3D Model Transform','Add into BOM','Convert to PCB','SymbolName','FootprintName']);
const pins=async c=>(await eda.sch_PrimitiveComponent.getAllPinsByPrimitiveId(c.getState_PrimitiveId())).map(p=>({number:p.getState_PinNumber(),x:p.getState_X(),y:p.getState_Y(),rotation:p.getState_Rotation(),nc:p.getState_NoConnected()}));
const before=[],changed=[];
const targetRefs=new Set(__TARGETS__);
for(const c of cs){const ref=c.getState_Designator();const catalog=byId[c.getState_SupplierId()];if(!targetRefs.has(ref)||!catalog||(__NINE__&&!/^C_BAL_CELL[1-9]$/.test(ref)))continue;
const name=catalog.otherProperty.Name;if(typeof name!=='string'||!name)throw Error('Missing catalog name '+ref);
const row={ref,id:c.getState_PrimitiveId(),name:c.getState_Name(),mpn:c.getState_ManufacturerId(),manufacturer:c.getState_Manufacturer(),supplier:c.getState_Supplier(),supplierId:c.getState_SupplierId(),other:c.getState_OtherProperty(),component:c.getState_Component(),symbol:c.getState_Symbol(),footprint:c.getState_Footprint(),x:c.getState_X(),y:c.getState_Y(),rotation:c.getState_Rotation(),mirror:c.getState_Mirror(),bom:c.getState_AddIntoBom(),pcb:c.getState_AddIntoPcb(),uniqueId:c.getState_UniqueId(),pins:await pins(c)};
const other={...row.other};for(const [key,value] of Object.entries(catalog.otherProperty)){if(!reserved.has(key))other[key]=value;}
// Preserve the former descriptive display in a separate engineering field.
other['Engineering Description']=row.other['Engineering Description']||row.name;
await eda.sch_PrimitiveComponent.modify(c,{x:row.x,y:row.y,rotation:row.rotation,mirror:row.mirror,addIntoBom:row.bom,addIntoPcb:row.pcb,designator:ref,name,uniqueId:row.uniqueId,manufacturer:row.manufacturer,manufacturerId:row.mpn,supplier:row.supplier,supplierId:row.supplierId,otherProperty:other});before.push(row);changed.push({ref,oldName:row.name,name});}
const now=await eda.sch_PrimitiveComponent.getAll();const byRef=Object.fromEntries(now.map(c=>[c.getState_Designator(),c]));
for(const r of before){const c=byRef[r.ref];if(!c||c.getState_PrimitiveId()!==r.id||c.getState_ManufacturerId()!==r.mpn||c.getState_SupplierId()!==r.supplierId||JSON.stringify(await pins(c))!==JSON.stringify(r.pins)||JSON.stringify(c.getState_Symbol())!==JSON.stringify(r.symbol)||JSON.stringify(c.getState_Footprint())!==JSON.stringify(r.footprint))throw Error('Identity/pin/carrier changed '+r.ref);}
await eda.sch_Document.save();return {before,changed,drc:await eda.sch_Drc.check(true,false,true)};
""".replace('__NINE__','true' if args.nine else 'false').replace('__TARGETS__',json.dumps(warning_refs))
v=d.h.invoke(code,session=sid);(E/(args.output+'.json')).write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');print(json.dumps({'changed':len(v['changed']),'drc':v['drc']},ensure_ascii=False))
