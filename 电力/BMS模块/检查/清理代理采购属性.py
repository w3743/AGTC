"""通过官方 API 写入中性 BOM 清单标记并修正源 DeviceName。"""
import json
import pathlib

from 原生更新 import invoke


root = pathlib.Path(__file__).resolve().parent
review = json.loads((root / '代理采购身份清理清单.json').read_text(encoding='utf-8'))
entries = json.loads((root / '当前施工输入.json').read_text(encoding='utf-8'))
pages = json.loads((root / '原理图页面映射.json').read_text(encoding='utf-8'))

# Keep the 18 Q_P/Q_S source-order-code review items untouched.
actions = {
    row['ref']: {'clearFields': row['clearFields'], 'action': row['action']}
    for row in review['clearOrCorrect']
    if row['action'] != 'manual_source_order_code_review_before_clear'
}
transformer_refs = {f'T_BAL{index}' for index in range(1, 10)}
target_refs = set(actions) | transformer_refs
neutral_supplier_refs = {
    ref for ref, action in actions.items()
    if 'Supplier' in action['clearFields']
} | transformer_refs

manual_refs = set(review.get('reviewBeforeClearing', {}).get('refs', []))
carrier_refs = {
    row['ref'] for row in review.get('carrierOnlyKeep', []) if row.get('ref')
}
if len(actions) != 48 or len(manual_refs) != 18:
    raise ValueError(f'清理清单范围异常；automatic={len(actions)}, manualReview={len(manual_refs)}')
if len(target_refs) != 56 or target_refs & manual_refs or target_refs & carrier_refs:
    raise ValueError(
        '清理目标范围异常；'
        f'unique={len(target_refs)}, manualOverlap={sorted(target_refs & manual_refs)}, '
        f'carrierOnlyKeepOverlap={sorted(target_refs & carrier_refs)}'
    )
if len(neutral_supplier_refs) != 44 or len(target_refs - neutral_supplier_refs) != 12:
    raise ValueError(
        '中性采购字段范围异常；'
        f'neutralSupplier={len(neutral_supplier_refs)}, deviceNameOnly={len(target_refs - neutral_supplier_refs)}'
    )

# The construction input uses the `mpn` key. Multi-unit records under one
# ref must agree before any official API call is made.
mpn_values_by_ref = {ref: set() for ref in target_refs}
for entry in entries:
    ref = entry.get('ref')
    if ref in target_refs:
        mpn_values_by_ref[ref].add(str(entry.get('mpn') or '').strip())
missing_refs = sorted(ref for ref, values in mpn_values_by_ref.items() if not values or values == {''})
conflicting_mpns = {ref: sorted(values) for ref, values in mpn_values_by_ref.items() if len(values) != 1}
if missing_refs or conflicting_mpns:
    raise ValueError(f'源 MPN 缺失或重复单元不一致；missing={missing_refs}, conflicts={conflicting_mpns}')
source_mpns = {ref: next(iter(values)) for ref, values in mpn_values_by_ref.items()}

results = []
for group, page_id in pages.items():
    refs = list(dict.fromkeys(
        entry['ref'] for entry in entries
        if entry.get('pageGroup') == group and entry.get('ref') in target_refs
    ))
    if not refs:
        continue

    code = (
        'const sourceMpns=' + json.dumps(source_mpns, ensure_ascii=False, separators=(',', ':')) + ';'
        + 'const neutralSupplierRefs=' + json.dumps(sorted(neutral_supplier_refs)) + ';'
        + 'const refs=' + json.dumps(refs, ensure_ascii=False) + ';'
    )
    code += 'await eda.dmt_EditorControl.openDocument(' + json.dumps(page_id) + ');'
    code += r'''
const result=[];
for(const c of (await eda.sch_PrimitiveComponent.getAll()).filter(Boolean)){
 const ref=c.getState_Designator();if(!refs.includes(ref))continue;
 const sourceMpn=sourceMpns[ref];
 const prop={otherProperty:{...c.getState_OtherProperty(),DeviceName:sourceMpn}};
 // Nonempty neutral markers prevent proxy-library procurement fields from reappearing in ENET.
 if(neutralSupplierRefs.includes(ref)){prop.supplier='BOM_ONLY';prop.supplierId='SEE_BOM_CSV';}
 // Retain the source manufacturer for Coilcraft transformer entries.
 if(ref.startsWith('T_BAL'))prop.manufacturer='Coilcraft';
 await eda.sch_PrimitiveComponent.modify(c.getState_PrimitiveId(),prop);
 result.push({ref,sourceMpn,manufacturer:ref.startsWith('T_BAL')?'Coilcraft':undefined,supplier:neutralSupplierRefs.includes(ref)?'BOM_ONLY':undefined,supplierId:neutralSupplierRefs.includes(ref)?'SEE_BOM_CSV':undefined});
}
return {result,saved:await eda.sch_Document.save()};
'''
    out = invoke(code)
    results.append({'group': group, **out})
    print(group, len(out['result']), flush=True)

(root / '代理采购属性修复.json').write_text(
    json.dumps(results, ensure_ascii=False, indent=2), encoding='utf-8'
)
