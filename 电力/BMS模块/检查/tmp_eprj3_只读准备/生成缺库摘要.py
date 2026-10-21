import json, re
from pathlib import Path

root = Path(r'E:\Pioneer\先进通用履带底盘工程\电力')
tmp = root / 'BMS模块' / '检查' / 'tmp_eprj3_只读准备'
report = json.loads((tmp / '单页eprj3施工清单.json').read_text(encoding='utf-8-sig'))
missing = [x for x in report['components'] if x.get('bindingSource') == 'missing']
by_mpn = {}
for x in missing:
    by_mpn.setdefault(x['mpn'], []).append(x)

# Search only saved project JSON snapshots, recording exact identity fields plus UUIDs.
identity_fields = {'mpn','part','partnumber','manufacturerpartnumber','manufacturerpart','orderingcode','partno','devicepartnumber'}
results = {m: [] for m in by_mpn}
for p in (root / 'BMS模块' / '检查').rglob('*.json'):
    if tmp in p.parents or p.name == '单页eprj3施工清单.json':
        continue
    try:
        obj = json.loads(p.read_text(encoding='utf-8-sig'))
    except Exception:
        continue
    def visit(node, path='$'):
        if isinstance(node, dict):
            vals = {str(k).lower(): v for k, v in node.items()}
            for mpn in by_mpn:
                if any(isinstance(vals.get(k), str) and vals[k].strip().casefold() == mpn.casefold() for k in identity_fields):
                    ids = {k: v for k, v in node.items() if re.search(r'(uuid|supplierpart|footprintname|symbolname|libtype|status)', str(k), re.I)}
                    if ids:
                        results[mpn].append((str(p.relative_to(root)), path, ids))
            for k, v in node.items():
                visit(v, f'{path}.{k}')
        elif isinstance(node, list):
            for i, v in enumerate(node):
                visit(v, f'{path}[{i}]')
    visit(obj)

lines = []
lines.append('# 单页eprj3离线binding准备清单（tmp）')
lines.append('')
lines.append(f"输入源SHA256：`{report['sourceInputSha256']}`；源记录{report['sourceRowCount']}、唯一ref {report['sourceRefCount']}、电气符号{report['electricalSymbolRefCount']}、无电气pin装配注记{report['nonElectricalItemCount']}。严格核对后缺binding为{len(missing)}个ref、{len(by_mpn)}个不同MPN。")
lines.append('')
lines.append('旧映射ref继承仅在source MPN与旧map MPN完全相同（或双方MPN均空）时允许；因此像Q_PRE1/2的NCE01P30K、U_TIMER的TI TPL5111DDCR不继承旧同ref UUID。')
lines.append('')
lines.append('下列`pin→net`只代表现行连接源的电气pin合同。因该型号未绑定库，不能据此确认symbol pin name到footprint pad的映射或实物极性；若厂商MPN有多封装/多厂版本，需为选定精确ordering code单独确认。')
lines.append('')
for mpn, rows in sorted(by_mpn.items(), key=lambda kv: kv[0].casefold()):
    lines.append(f'## {mpn} — {len(rows)} ref')
    per_map = {}
    for row in rows:
        pinmap = row.get('pins') or {}
        sig = ', '.join(f'{pin}→{net}' for pin,net in sorted(pinmap.items(), key=lambda kv: (int(kv[0]) if str(kv[0]).isdigit() else 999, str(kv[0])))) or '(无pin)'
        per_map.setdefault(sig, []).append(row['ref'])
    for sig, refs in per_map.items():
        lines.append(f'- `{", ".join(sorted(refs))}`：{sig}')
    hits = results.get(mpn, [])
    if hits:
        lines.append(f'- 本地缓存中发现{len(hits)}条具有同名字段和UUID的候选记录（只能视为搜库痕迹，需读回确认具体设备库与完整pin/pad映射）：')
        for filepath, path, ids in hits[:8]:
            lines.append(f'  - `{filepath}` {path}: `{json.dumps(ids, ensure_ascii=False, sort_keys=True)}`')
    else:
        lines.append('- 本地检查JSON未找到带同MPN身份字段及UUID的记录。')
    lines.append('')

out = tmp / '单页eprj3真实缺库型号与源pin合同.md'
out.write_text('\n'.join(lines) + '\n', encoding='utf-8')
print(f'written={out}; missing_refs={len(missing)}; distinct_mpn={len(by_mpn)}')
print('Local cache exact-ID hits:')
for mpn,hits in sorted(results.items()):
    if hits:
        print(mpn, len(hits), hits[0])

