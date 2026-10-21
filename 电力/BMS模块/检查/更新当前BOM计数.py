"""Refresh the current purchasing count evidence from the current BOM.

This is an inventory report, not an electrical or implementation test.
"""
from pathlib import Path
from collections import defaultdict
import csv
import datetime
import hashlib
import json
import re
import zipfile

MODULE = Path(__file__).resolve().parents[1]
BOM = MODULE / '原理图准备' / 'BOM.csv'
INTERFACE = MODULE / '原理图准备' / '接口元件连接.json'
OUT = MODULE / '检查' / '当前BOM计数证据.json'
PREFIX = re.compile(r'^(?:ADI|Anderson|Eaton|Infineon|KOA Speer|Littelfuse|MDD|Murata|onsemi|Songle|TE Connectivity|TE|TI|WeAct Normal|Yageo)\s+', re.I)

def normalize(s):
    s = PREFIX.sub('', s.strip())
    s = re.sub(r'\s*\(LCSC\s+[^)]*\)', '', s, flags=re.I)
    m = re.search(r'Phoenix\s+MSTBA.*\((\d+)\)', s, re.I)
    if m:
        s = m.group(1)
    if re.fullmatch(r'WSF[ ._-]*60[ ._-]*N[ ._-]*06', s, re.I):
        s = 'WSF60N06'
    return s.lower()

def qty(row):
    return int(float(row['每套用量']))

def main():
    rows = list(csv.DictReader(BOM.open(encoding='utf-8-sig', newline='')))
    old = json.loads(OUT.read_text(encoding='utf-8-sig')) if OUT.exists() else {}
    ntc_rows = [row for row in rows if row['位号'].startswith('NTC_')]
    ntc_refs = [row['位号'] for row in ntc_rows]
    accessory_rows = [row for row in rows if row['类别'] == '板外/装配配套']
    accessory_refs = [row['位号'] for row in accessory_rows]
    board_mounted_accessory_refs = [ref for ref in accessory_refs if ref == 'FH_FAN']
    external_accessory_refs = [ref for ref in accessory_refs if ref not in board_mounted_accessory_refs]
    interface_parts = json.loads(INTERFACE.read_text(encoding='utf-8-sig'))
    interface_notes = {
        part.get('ref', ''): part.get('note', '')
        for part in interface_parts
        if isinstance(part, dict)
    }
    if OUT.exists():
        before = OUT.read_bytes()
        key = '计数刷新前_' + datetime.datetime.now().strftime('%Y%m%d_%H%M%S') + '/检查/当前BOM计数证据.json'
        with zipfile.ZipFile(MODULE / '归档' / '历史与调试.zip', 'a', compression=zipfile.ZIP_DEFLATED) as z:
            z.writestr(key, before)
            if hashlib.sha256(z.read(key)).digest() != hashlib.sha256(before).digest():
                raise RuntimeError('计数证据归档读回不一致')
    grouped = defaultdict(lambda: {'refs': 0, 'qty': 0})
    aliases = defaultdict(set)
    model_qty = defaultdict(lambda: {'refs': 0, 'qty': 0})
    for row in rows:
        grouped[row['类别']]['refs'] += 1
        grouped[row['类别']]['qty'] += qty(row)
        if row['MPN'].strip():
            model = normalize(row['MPN'])
            aliases[model].add(row['MPN'])
            model_qty[model]['refs'] += 1
            model_qty[model]['qty'] += qty(row)
    empty = [row['位号'] for row in rows if not row['MPN'].strip()]
    models_without_core = {normalize(row['MPN']) for row in rows if row['MPN'].strip() and row['位号'] != 'CORE_BOARD'}
    old.update({
        'snapshotDate': datetime.date.today().isoformat(),
        'updatedAt': datetime.datetime.now().astimezone().isoformat(),
        'sourceBOM': 'BMS模块/原理图准备/BOM.csv',
        'bomSha256': hashlib.sha256(BOM.read_bytes()).hexdigest(),
        'counts': {
            'uniqueRefRows': len(rows),
            'uniqueRefs': len({row['位号'] for row in rows}),
            'perSetTotalQty': sum(qty(row) for row in rows),
            'rawNonemptyMpnTextVariants': len({row['MPN'] for row in rows if row['MPN'].strip()}),
            'normalizedMpnTypesIncludingCoreBoard': len(aliases),
            'normalizedSelectedMpnTypesExcludingCoreBoard': len(models_without_core),
            'mpnEmptyRows': len(empty),
            'mpnEmptyRefs': empty,
            'nonComponentAnchorRefsExcluded': ['J_BAL_SERVICE', 'J_BAT_HV', 'J_BUS_OUTPUT', 'J_CELL_SERVICE', 'J_PACK_HV'],
            'dnpPopulatedRows': 0,
            'dnpRefs': []
        },
        'byCategory': dict(grouped),
        'top10ByQuantity': [dict(mpn=k, **v) for k,v in sorted(model_qty.items(), key=lambda item: (-item[1]['qty'], item[0]))[:10]]
    })
    old.setdefault('normalization', {})['aliasGroups'] = [dict(normalizedMpn=k, rawForms=sorted(v)) for k,v in aliases.items() if len(v)>1]
    old['normalization']['rawAliasGroupsCollapsed'] = len(old['normalization']['aliasGroups'])
    for key in ('categories', 'categoryBreakdown', 'sourceBreakdown'):
        if key in old:
            old[key] = dict(grouped)
    if 'separateItems' in old and 'maintenanceHarness' in old['separateItems']:
        harness = old['separateItems']['maintenanceHarness']
        harness['refs'] = grouped.get('维护线束', {}).get('refs', 0)
        harness['qty'] = grouped.get('维护线束', {}).get('qty', 0)
    if 'separateItems' in old:
        accessories = old['separateItems'].setdefault('assemblyAccessories', {})
        accessories['refs'] = accessory_refs
        accessories['qty'] = sum(qty(row) for row in accessory_rows)
        accessories['boardMountedRefs'] = board_mounted_accessory_refs
        accessories['externalOrAssemblyRefs'] = external_accessory_refs
        accessories['notesByRef'] = {
            row['位号']: row.get('备注', '').strip()
            for row in accessory_rows
        }
        accessories['notes'] = 'refs、用量和逐项说明由当前BOM“板外/装配配套”类别生成；板上风扇保险座FH_FAN已退出，当前这些项均为外部/装配配套。该类别总数不等于PCB贴装数。'
        sensors = old['separateItems'].setdefault('temperatureSensors', {})
        sensors['refs'] = ntc_refs
        sensors['qty'] = sum(qty(row) for row in ntc_rows)
        sensors['cellProbesClearlyExternal'] = [
            ref for ref in ntc_refs if re.fullmatch(r'NTC_CELL\d+', ref)
        ]
        sensors.pop('otherLocationAmbiguous', None)
        sensors['placementByRef'] = {
            ref: {
                'location': location,
                'interfaceNote': interface_notes.get(ref, ''),
                'source': 'BMS模块/原理图准备/接口元件连接.json',
            }
            for ref, location in (
                ('NTC_BOARD', '板上相邻均衡功率区热点（BQ76952 TS2采样）'),
                ('NTC_MOS', 'MOS功率区热点'),
            )
            if ref in ntc_refs
        }
        cell_zones = {}
        for ref in ntc_refs:
            if not re.fullmatch(r'NTC_CELL\d+', ref):
                continue
            note = interface_notes.get(ref, '')
            match = re.search(r'representative cell-pack thermal zone\s+([A-Z])', note, re.I)
            if match:
                cell_zones[ref] = match.group(1).upper()
        sensors['representativeCellThermalZones'] = cell_zones
        sensors['cellCoverageNote'] = '每个NTC_CELL探头代表其标注热区，不表示逐电芯组测温。'
        old['separateItems']['coreBoardInternalAdcTemperatureSensor'] = {
            'ref': 'CORE_BOARD内STM32G431内部温度传感器',
            'qty': 0,
            'isBOMItem': False,
            'note': '仅作核心板MCU芯片温度参考，不是独立采购NTC或BOM位号。',
        }
    electrical_qty = sum(
        qty(row) for row in rows
        if row['类别'] not in {'维护线束', '外部控制板', '板外/装配配套'}
        or row['位号'] in board_mounted_accessory_refs
    )
    clearly_external_refs = {'F1', 'R_PRE', 'SW_RUN_STORE', 'NTC_CELL1', 'NTC_CELL3', 'NTC_CELL5'}
    board_rows = [
        row for row in rows
        if row['类别'] not in {'维护线束', '外部控制板'}
        and (row['类别'] != '板外/装配配套' or row['位号'] in board_mounted_accessory_refs)
        and row['位号'] not in clearly_external_refs
    ]
    board_models = {normalize(row['MPN']) for row in board_rows if row['MPN'].strip()}
    old['boardMountBoundary'] = {
        'electricalCircuitQty': electrical_qty,
        'mainPcbMountedQty': sum(qty(row) for row in board_rows),
        'mainPcbUniqueRefs': len({row['位号'] for row in board_rows}),
        'mainPcbNormalizedMpnTypes': len(board_models),
        'mainPcbMpnEmptyRefs': [row['位号'] for row in board_rows if not row['MPN'].strip()],
        'mainPcbRefs': [row['位号'] for row in board_rows],
        'knownBoardMountedAccessoryRefs': board_mounted_accessory_refs,
        'knownExternalOrAssemblyAccessoryRefs': external_accessory_refs,
        'externalCircuitRefsExcluded': sorted(clearly_external_refs),
        'note': '用户要求仅设计BMS主板。主板统计排除全部维护线束、外置核心板、板外配套、外置主保险F1、外置预充电阻R_PRE、面板开关SW_RUN_STORE及三只电芯探头；保留板装NTC_BOARD（板上相邻均衡功率区热点）/NTC_MOS、排母及其他板内器件。纯焊盘/连接边界不计采购器件。统计为现行设计拟装配数量，不是layout或实物验收。'
    }
    OUT.write_text(json.dumps(old, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(old['counts'], ensure_ascii=False))

if __name__ == '__main__':
    main()
