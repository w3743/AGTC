"""Reproduce the observer-header deletion on a copy of the pre-change P1."""
import base64
import importlib.util
import json
import shutil
import sys
import tempfile
from collections import Counter
from pathlib import Path

tool_dir = Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location("reflow",tool_dir/"Reflow-SavedFoc.py")
r=importlib.util.module_from_spec(spec)
spec.loader.exec_module(r)
eda=r.eda
name="J_PWR_SENSE"
stage=Path(sys.argv[1])
stage.mkdir(parents=True,exist_ok=True)
contract_path=eda.ROOT/"原理图准备/当前原理图连接.json"
contract=json.loads(contract_path.read_text(encoding="utf-8-sig"))
expected={c["ref"]:c["pins"] for c in contract["components"]}
expected_removed={"1":"5V_BUCK","2":"MCU_3V3","3":"GND_SYS","4":"SAFE_ADC","5":"BUS_ADC","6":"BREAK_N","7":"DRV_ENABLE","8":"BREAK_N"}
if expected.get(name) != expected_removed:
    raise SystemExit("This replay requires the pre-change FOC pin contract and native project.")
before=eda.hashes(eda.NATIVE)
readback=(tool_dir/"Read-SingleSheetActualGeometry.js").read_text(encoding="utf-8-sig")
pdfjs=(tool_dir/"Export-CurrentPdf.js").read_text(encoding="utf-8-sig")
old_net=json.loads((eda.DELIVERY/"FOC_G431_P1.enet").read_text(encoding="utf-8-sig"))
old_bom=r.bom_rows((eda.DELIVERY/"FOC_G431_BOM.csv").read_text(encoding="utf-8-sig"))
r.TARGETS={name:(0,0)}
with tempfile.TemporaryDirectory(prefix="foc_sense_") as folder:
    src=Path(folder)/eda.NATIVE.name
    shutil.copytree(eda.NATIVE,src)
    assert before==eda.hashes(src)
    path=src/(src.name+".eprj3")
    with eda.headless(path) as (invoke,_):
        geometry=invoke(readback)
        drc_before=invoke("return await eda.sch_Drc.check(true,false,true);")
        plan=r.get_plan(geometry,expected,False)
        part=next(x for x in geometry["components"] if x["ref"]==name)
        assert part["manufacturerId"]=="XY-XH2.54-8A11"
        js=(tool_dir/"Reflow-DenseP1.js").read_text(encoding="utf-8").split("for (const [ref, id, x, y, expected, passive]")[0]
        code="const plan="+json.dumps(plan,ensure_ascii=False)+";let stage;try{\n"+js+"""
const prior = (await eda.sch_PrimitiveComponent.getAll(undefined,false)).find(c => c.getState_Designator() === "J_PWR_SENSE");
if (!prior) throw new Error("No target component");
const removed = await eda.sch_PrimitiveComponent.delete([prior.getState_PrimitiveId()]);
const remains = (await eda.sch_PrimitiveComponent.getAll(undefined,false)).some(c => c.getState_Designator() === "J_PWR_SENSE");
if (remains) throw new Error("J_PWR_SENSE deletion returned "+removed);
for (const [y, label] of [
  [1900,'观测及故障注入复用核心板排针'],
  [1875,'J_MCU_L.5 BREAK_N：高阻观测或开漏拉低。'],
  [1850,'J_MCU_R.6/.9/.12：EN、SAFE_ADC、BUS_ADC。'],
  [1825,'电源和GND使用排针原针位，高阻测量。'],
]) {
  if (!await eda.sch_PrimitiveText.create(1810, y, label, 0, null, null, 12, false)) throw new Error('note');
}
if (!await eda.sch_Document.save()) throw new Error('save');
return {removed:true};
} catch(error) {return {stage,error:String(error)};}
"""
        result=invoke(code)
        if result!={"removed":True}: raise RuntimeError(result)
    with eda.headless(path) as (invoke,_):
        drc,net_text,bom=eda.exports(invoke)
        pdf=invoke(pdfjs)
        reopened=invoke(readback)
    after=eda.hashes(src)
    new_net=json.loads(net_text.lstrip("\ufeff"))
    baseline_electrical=r.electrical_netlist(old_net)
    old_id=next(key for key,c in old_net["components"].items() if c["props"]["Designator"]==name)
    baseline_electrical["components"].pop(old_id)
    old_pins=eda.pins(old_net)
    old_pins.pop(name)
    target_bom=[row for row in old_bom if row[old_bom[0].index("Designator")]!=name]
    new_bom=r.bom_rows(bom)
    target_bom=[[str(i),*row[1:]] for i,row in enumerate(target_bom[1:],1)]
    drc_expected=Counter({key:n for key,n in r.warning_primitives(drc_before).items() if key[-1]!=part["id"]})
    report={"candidate":name,"removedIdentity":part["manufacturerId"],"removedSupplier":part["supplierId"],"removedPins":expected_removed,"beforeSha256":before,"reopenedSha256":after,"prunedWires":len(plan["keepWires"]),"removedPorts":len(plan["removePorts"]),"removedWires":len(plan["removeWires"]),"pinDifferences":eda.compare(old_pins,eda.pins(new_net)),"electricalNetlistMatchesExpected":baseline_electrical==r.electrical_netlist(new_net),"bomMatchesExpected":target_bom==new_bom[1:],"drcCounts":dict(Counter(row["type"] for row in drc)),"drcPrimitivesMatchExpected":drc_expected==r.warning_primitives(drc),"ncMarkers":r.nc_markers(reopened),"ncMatches":r.nc_markers(reopened)==r.nc_markers(geometry),"drc":drc}
    (stage/"evidence.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print({k:v for k,v in report.items() if k not in ("drc","beforeSha256","reopenedSha256")})
    assert before==eda.hashes(eda.NATIVE) and before.keys()==after.keys()
    assert report["electricalNetlistMatchesExpected"] and report["bomMatchesExpected"] and report["ncMatches"] and report["drcPrimitivesMatchExpected"]
    assert not report["pinDifferences"] and not report["drcCounts"].get("error")
    for name,sha in after.items():
        if before[name]!=sha:
            assert name in r.NATIVE_ALLOWED
            target=stage/"native"/name
            target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(src/name,target)
    (stage/"FOC_G431_P1.enet").write_bytes(net_text.encode("utf-8"))
    (stage/"FOC_G431_BOM.csv").write_bytes(bom.encode("utf-8"))
    (stage/"FOC_G431_单页原理图.pdf").write_bytes(base64.b64decode(pdf["base64"],validate=True))
