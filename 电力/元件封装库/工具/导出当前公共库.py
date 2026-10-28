#!/usr/bin/env python3
"""导出当前共用型号与原生库资产；不导出板级位号和原理图。"""
import argparse
import hashlib
import json
from pathlib import Path

LIB = Path(__file__).resolve().parents[1]
ROOT = LIB.parent


def read(p):
    return json.loads(p.read_text(encoding='utf-8-sig'))


def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'tmp/vehicle-power-hardware-common-publish')
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    out = args.output.resolve()
    index_path = out / '库/当前型号与原生资产.json'
    if args.check:
        index = read(index_path)
        errors = [a['path'] for a in index['assets']
                  if not (out / a['path']).is_file() or digest(out / a['path']) != a['sha256']]
        if index['input_index_sha256'] != digest(LIB / '索引/元件总表.json'):
            errors.append('原生索引已变化，需要重导出')
        if index.get('procurement_index_sha256') != digest(LIB / '索引/采购补充.json'):
            errors.append('采购索引已变化，需要重导出')
        known = {a['path'] for a in index['assets']}
        actual = {p.relative_to(out).as_posix() for p in (out / '库/原生片段').rglob('*') if p.is_file()}
        errors.extend('未被当前资产索引引用:' + p for p in sorted(actual - known))
        for part in index['parts']:
            for binding in part['bindings']:
                for role in ('device_asset', 'symbol_asset', 'footprint_asset'):
                    if binding[role]['path'] not in known:
                        errors.append(part['key'] + ':' + role)
        print(json.dumps({'pass': not errors, 'parts': len(index['parts']),
                          'assets': len(index['assets']), 'errors': errors}, ensure_ascii=False))
        return bool(errors)
    total_path = LIB / '索引/元件总表.json'
    total = read(total_path)
    purchase_path = LIB / '索引/采购补充.json'
    purchase = read(purchase_path)
    by_part = {}
    for inst in total['instances']:
        by_part.setdefault(inst['part_key'], []).append(inst)
    assets = {}
    parts = []
    for p in total['parts']:
        bindings = {}
        for inst in by_part[p['key']]:
            binding = {'carrier_manufacturer': inst['carrier_manufacturer'],
                       'carrier_mpn': inst['carrier_mpn'], 'target_differs_from_carrier': inst['mpn_diff']}
            for role in ('device_asset', 'symbol_asset', 'footprint_asset'):
                a = inst[role]
                rel = '库/原生片段/' + a['type'] + '/' + a['sha256'] + '.jsonl'
                binding[role] = {k: a.get(k) for k in ('type', 'title', 'source_uuid', 'library_uuid', 'sha256', 'status')}
                binding[role]['path'] = rel
                src = ROOT / a['path']
                if digest(src) != a['sha256']:
                    raise ValueError('原生资产SHA变化: ' + str(src))
                dest = out / rel
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(src.read_bytes())
                assets[rel] = {'type': a['type'], 'path': rel, 'sha256': a['sha256']}
            key = json.dumps(binding, sort_keys=True, ensure_ascii=False)
            bindings[key] = binding
        parts.append({k: p[k] for k in ('key', 'manufacturer', 'mpn', 'supplier_parts')} |
                     {'bindings': list(bindings.values()), 'status': '当前采用身份；适用性与载体差异仍须按具体用途核对'})
    procurement = {}
    for r in purchase['records']:
        if r['native_keys'] or not (r['mpn_source'] or '').strip():
            continue
        key = (r['manufacturer_source'], r['mpn_source'])
        part = procurement.setdefault(key, {'manufacturer_source': key[0], 'mpn_source': key[1], 'source_statuses': []})
        if r['status_source'] not in part['source_statuses']:
            part['source_statuses'].append(r['status_source'])
    index = {'schema_version': 1,
             'boundary': '仅当前原生BOM的器件身份及所引用原生资产。JSONL不是独立可导入库包；无板级位号或原理图。型号覆盖载体时不得直接把载体当作目标器件。',
             'input_index_sha256': digest(total_path), 'procurement_index_sha256': digest(purchase_path),
             'parts': parts, 'procurement_parts': list(procurement.values()),
             'procurement_boundary': '原生BOM之外的现行采购记录，保留原始型号与候选状态；不宣称有原生符号封装绑定。',
             'assets': list(assets.values())}
    index_path.parent.mkdir(parents=True, exist_ok=True)
    # Only retire files previously owned by this generated manifest. Never scan
    # or delete unrelated source/library files.
    if index_path.exists():
        previous = read(index_path)
        for old in previous.get('assets', []):
            rel = old['path']
            if rel in assets:
                continue
            candidate = (out / rel).resolve()
            owned = (out / '库/原生片段').resolve()
            if not candidate.is_relative_to(owned):
                raise ValueError('旧资产路径越界: ' + rel)
            if candidate.is_file():
                if digest(candidate) != old['sha256']:
                    raise ValueError('待清理旧导出资产已被修改: ' + rel)
                candidate.unlink()
    index_path.write_text(json.dumps(index, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'parts': len(parts), 'assets': len(assets)}, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
