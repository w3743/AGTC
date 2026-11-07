from eda import *
invoke("await eda.sch_Document.save();await eda.dmt_EditorControl.openDocument('bafe8a48e4625018');await eda.sch_Document.save();return true;")
save('原生DRC.json',invoke('return await eda.sch_Drc.check(true,false,true);'))
r=invoke("const f=await eda.sch_ManufactureData.getNetlistFile('水泵驱动.enet','EasyEDA');return {name:f.name,text:await f.text()};")
out=ROOT/'eda工程文件/导出/当前交付';out.mkdir(parents=True,exist_ok=True)
(out/'水泵驱动.enet').write_text(r['text'],encoding='utf8')
save('原生器件绑定.json',invoke("const cs=await eda.sch_PrimitiveComponent.getAll();const out=[];for(const c of cs.filter(v=>v.getState_ComponentType()==='part'))out.push({ref:c.getState_Designator(),id:c.getState_PrimitiveId(),mpn:c.getState_ManufacturerId(),manufacturer:c.getState_Manufacturer(),component:c.getState_Component(),symbol:c.getState_Symbol(),footprint:c.getState_Footprint(),otherProperty:c.getState_OtherProperty(),pins:await eda.sch_PrimitiveComponent.getAllPinsByPrimitiveId(c.getState_PrimitiveId())});return out;"))
print('DRC', (ROOT/'检查/原生DRC.json').read_text(encoding='utf8')[:3000]);print('netlist bytes',len(r['text']))
