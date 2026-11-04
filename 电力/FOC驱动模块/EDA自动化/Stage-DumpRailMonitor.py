"""Stage the DUMP_5V undervoltage interlock in a separate EasyEDA project."""
import argparse
import base64
import importlib.util
import json
import shutil
from collections import Counter
from pathlib import Path


HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("reflow", HERE / "Reflow-SavedFoc.py")
reflow = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(reflow)
eda = reflow.eda
PIN_TARGET = {
    "U8": {"1": "", "2": "", "3": "GND_SYS", "4": "MCU_3V3", "5": "DUMP_UV_SENSE", "6": "BREAK_N"},
    "R_DUMP_UV_TOP": {"1": "DUMP_5V", "2": "DUMP_UV_SENSE"},
    "R_DUMP_UV_BOT": {"1": "DUMP_UV_SENSE", "2": "GND_SYS"},
    "C_DUMP_UV_VDD": {"1": "MCU_3V3", "2": "GND_SYS"},
}
PART_TARGET = {
    "U8": "TPS3899DL01DSER", "R_DUMP_UV_TOP": "RC0603FR-07180KL",
    "R_DUMP_UV_BOT": "RC0603FR-0726K7L", "C_DUMP_UV_VDD": "C1608X7R1A104K080AE",
}
EDIT = r"""
let step = 'bind';
try {
  const page = await eda.dmt_Schematic.getCurrentSchematicPageInfo();
  if (page?.uuid !== '90aa6c46493624cf') throw new Error('wrong schematic page');
  const parts = new Map((await eda.sch_PrimitiveComponent.getAll(undefined,false))
    .filter(c=>c.getState_Designator()).map(c=>[c.getState_Designator(),c]));
  if (parts.size !== 152 || !parts.has('U7') || parts.has('U8')) throw new Error('wrong source version');
  const exact = async (mpn, supplierId) => {
    const rows = await eda.lib_Device.search(mpn,undefined,undefined,undefined,10,1);
    const matches = rows.filter(item=>item.manufacturerId === mpn && item.supplierId === supplierId);
    if (matches.length !== 1) throw new Error('cannot bind exact device '+mpn);
    return matches[0];
  };
  const capCarrier = await eda.lib_Device.get('4e48d71315ce4d4c9f2ca294a3b1380b','0819f05c4eef4c71ace90d822a990e87');
  const capBase = parts.get('C_U7');
  if (capCarrier?.name !== 'CAP_0603' || capBase.getState_ManufacturerId() !== 'C1608X7R1A104K080AE') {
    throw new Error('bypass identity mismatch');
  }
  const items = [
    ['U8', await exact('TPS3899DL01DSER','C2864821'), 725, 2120, 'TPS3899DL01DSER', null],
    ['R_DUMP_UV_TOP', await exact('RC0603FR-07180KL','C123419'), 295, 2155, null, null],
    ['R_DUMP_UV_BOT', await exact('RC0603FR-0726K7L','C137756'), 295, 2090, null, null],
    ['C_DUMP_UV_VDD', capCarrier, 1160, 2180, '100nF/10V', capBase],
  ];
  const added = [];
  for (const [ref, device, x, y, label, reuse] of items) {
    step = 'create ' + ref;
    const c = await eda.sch_PrimitiveComponent.create(device,x,y,undefined,0,false,true,true);
    if (!c) throw new Error(step);
    const overrides = reuse ? {manufacturer:reuse.getState_Manufacturer(), manufacturerId:reuse.getState_ManufacturerId(),
      supplier:reuse.getState_Supplier(), supplierId:reuse.getState_SupplierId(), value:'100nF/10V'} : {};
    const applied = await eda.sch_PrimitiveComponent.modify(c,{designator:ref,...(label ? {name:label} : {}),...overrides});
    if (!applied) throw new Error('modify '+ref);
    const id = c.getState_PrimitiveId();
    const pins = await eda.sch_PrimitiveComponent.getAllPinsByPrimitiveId(id);
    const expected = target[ref];
    if (pins.length !== Object.keys(expected).length) throw new Error('pin count '+ref);
    for (const p of pins) {
      const number = String(p.getState_PinNumber());
      step = 'route '+ref+'.'+number;
      if (!(number in expected)) throw new Error(step);
      const net = expected[number];
      if (!net) {
        await p.setState_NoConnected(true).done();
        continue;
      }
      const px = Math.round(p.getState_X()*1000000)/1000000;
      const py = Math.round(p.getState_Y()*1000000)/1000000;
      const angle = p.getState_Rotation();
      const deltas = {0:[150,0],90:[0,45],180:[-150,0],270:[0,-45]};
      if (!deltas[angle]) throw new Error('unsupported pin direction');
      const [dx,dy] = deltas[angle];
      const endpoint = [px+dx,py+dy];
      if (!await eda.sch_PrimitiveWire.create([[px,py,...endpoint]],net) ||
          !await eda.sch_PrimitiveComponent.createNetPort('BI',net,...endpoint,angle,false)) throw new Error(step);
    }
    added.push({ref,id});
  }
  for (const [x,y,label,size,bold] of [
      [120,2240,'隔离候选  /  DUMP_5V 欠压请求 Break',15,true],
      [120,2215,'独立由 MCU_3V3 供电。CTR/CTS 开路；U7 母线过压支路仍独立。',12,false],
  ]) {
    step = 'note';
    if (!await eda.sch_PrimitiveText.create(x,y,label,0,null,null,size,bold)) throw new Error(step);
  }
  step = 'save';
  if (!await eda.sch_Document.save()) throw new Error(step);
  return {added};
} catch (error) { return {error:String(error),step}; }
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", type=Path)
    args = parser.parse_args()
    stage = args.stage.resolve()
    if stage.is_relative_to(eda.ROOT):
        parser.error("stage must be outside the live FOC module")
    stage.mkdir(parents=True, exist_ok=True)
    source = eda.hashes(eda.NATIVE)
    copy = stage / eda.NATIVE.name
    shutil.copytree(eda.NATIVE, copy)
    assert source == eda.hashes(copy)
    project = copy / (copy.name + ".eprj3")
    with eda.headless(project) as (invoke, _):
        result = invoke("const target = " + json.dumps(PIN_TARGET) + ";\n" + EDIT)
        if not result or result.get("error"):
            raise RuntimeError(result)
    with eda.headless(project) as (invoke, _):
        pdf = invoke((HERE / "Export-CurrentPdf.js").read_text(encoding="utf-8-sig"))
        drc, net_text, bom = eda.exports(invoke)
        geometry = invoke((HERE / "Read-SingleSheetActualGeometry.js").read_text(encoding="utf-8-sig"))
    before = json.loads((eda.DELIVERY / "FOC_G431_P1.enet").read_text(encoding="utf-8-sig"))
    after = json.loads(net_text.lstrip("\ufeff"))
    pins = eda.pins(after)
    old_drc = json.loads((eda.ROOT / "检查/当前保存源_独立官方复跑.json").read_text(encoding="utf-8-sig"))
    actual = reflow.electrical_netlist(after)
    sense_rule = actual["designRule"]["netRule"].pop("DUMP_UV_SENSE")
    added = {key: part for key, part in actual["components"].items() if part["props"]["Designator"] in PIN_TARGET}
    for key in added:
        del actual["components"][key]
    new_bom = eda.parts(bom)
    old_bom = eda.parts((eda.DELIVERY / "FOC_G431_BOM.csv").read_text(encoding="utf-8-sig"))
    bom_diff = eda.compare(old_bom, new_bom)
    raw_base_bom = reflow.bom_rows((eda.DELIVERY / "FOC_G431_BOM.csv").read_text(encoding="utf-8-sig"))
    raw_new_bom = reflow.bom_rows(bom)
    report = {
        "status": "ISOLATED_CANDIDATE_ONLY", "edit": result,
        "nativeBeforeSha256": source, "candidateNativeSha256": eda.hashes(copy),
        "liveSourceIntact": source == eda.hashes(eda.NATIVE),
        "existingElectricalNetlistMatches": reflow.electrical_netlist(before) == actual,
        "addedNetRule": sense_rule,
        "pinDifferences": eda.compare(eda.pins(before) | PIN_TARGET, pins),
        "candidateProperties": {part["props"]["Designator"]: part["props"] for part in added.values()},
        "candidatePinsFromOfficialGeometry": {c["ref"]: c["pins"] for c in geometry["components"] if c["ref"] in PIN_TARGET},
        "ncMarkers": reflow.nc_markers(geometry), "bomDiff": bom_diff,
        "bomRowCountsBeforeAfter": [len(raw_base_bom)-1, len(raw_new_bom)-1],
        "drcCounts": dict(Counter(row["type"] for row in drc)), "drc": drc,
        "addedDRCPrimitives": [list(row) for row in
                               reflow.warning_primitives(drc) - reflow.warning_primitives(old_drc["drc"])],
    }
    (stage / "official-difference.json").write_text(json.dumps(report, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    (stage / "candidate.enet").write_bytes(net_text.encode("utf-8"))
    (stage / "candidate-bom.csv").write_bytes(bom.encode("utf-8"))
    (stage / "candidate.pdf").write_bytes(base64.b64decode(pdf["base64"], validate=True))
    print({key:report[key] for key in ("liveSourceIntact", "existingElectricalNetlistMatches", "pinDifferences", "bomDiff", "drcCounts")})
    assert report["liveSourceIntact"] and report["existingElectricalNetlistMatches"]
    assert {p for p,h in eda.hashes(copy).items() if source.get(p)!=h} == reflow.NATIVE_ALLOWED
    assert not report["pinDifferences"] and not report["drcCounts"].get("error")
    assert set(bom_diff) == set(PART_TARGET)
    assert all(bom_diff[ref][1]["mpn"] == mpn for ref,mpn in PART_TARGET.items())
    assert report["ncMarkers"] == sorted(old_drc["ncMarkers"] + ["U8.1", "U8.2"])


if __name__ == "__main__":
    main()
