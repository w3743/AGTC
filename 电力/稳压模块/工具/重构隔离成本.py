"""Price the official component delta using one-unit China DigiKey snapshots."""
import csv
import json
from decimal import Decimal
from pathlib import Path

run = Path(__file__).resolve().parents[1] / '检查' / '候选_预充与超容隔离_R1'
source = 'https://www.digikey.cn/zh/products/detail'
supplier = {
    'TI': 'texas-instruments', 'TDK': 'tdk', 'ADI': 'analog-devices-inc',
    'DIODES': 'diodes-incorporated', 'YAGEO': 'yageo',
}
prices = {
    'TPSI3052QDWZRQ1': ('55.41', 'TI', '17748455'),
    'TPS386000RGPR': ('30.60', 'TI', '2232784'),
    'TLV76033DBZR': ('1.90', 'TI', '7402851'),
    'SN74LVC1G74DCTR': ('6.45', 'TI', '2195375'),
    'SN74LVC1G11DCKR': ('1.82', 'TI', '770206'),
    'SN74LVC1G08DCKR': ('0.83', 'TI', '385719'),
    'SN74LVC1G17DCKR': ('2.15', 'TI', '389053'),
    'SN74LVC1G07DCKR': ('0.83', 'TI', '377456'),
    'DMP3098L-7': ('4.63', 'DIODES', '1964699'),
    'CGA6P3X7R1E226M250AB': ('15.05', 'TDK', '9555472'),
    '1N5819HW-7-F': ('2.56', 'DIODES', '814970'),
    'RC0603FR-0710RL': ('0.91', 'YAGEO', '726879'),
    'RT0603BRD0710KL': ('0.83', 'YAGEO', '4340589'),
    'RC0603FR-07100KL': ('0.83', 'YAGEO', '726889'),
    'RC0603FR-0747KL': ('0.83', 'YAGEO', '727253'),
    'RC0603FR-072K2L': ('0.83', 'YAGEO', '727016'),
    'RC0603FR-071KL': ('0.91', 'YAGEO', '726843'),
    'RC0603FR-074K7L': ('0.83', 'YAGEO', '727212'),
    'RC0603FR-071ML': ('0.83', 'YAGEO', '729791'),
    'RC0603FR-0722KL': ('0.83', 'YAGEO', '727056'),
    'LTC7001IMSE#PBF': ('77.57', 'ADI', '7363804'),
    'C3225X7R2A105K200AA': ('5.54', 'TDK', '789688'),
}
history = {'RS-03K1002FT': ('0.0125', 'C115324', 100), 'MMBT5551': ('0.087494', 'C408396', 50),
           '2541WV-2X01P': ('0.04066', 'C41430884', 100)}
subtotals = {'+': Decimal(0), '-': Decimal(0)}
legacy = Decimal(0)
unpriced = []
with (run / '器件增减.csv').open(encoding='utf-8-sig', newline='') as fp:
    rows = list(csv.DictReader(fp))
with (run / '净BOM差额.csv').open('w', encoding='utf-8-sig', newline='') as fp:
    writer = csv.writer(fp)
    writer.writerow(['变更', '位号', '制造商', '料号', '得捷中国单件含税元', '得捷报价链接',
                     '项目历史摊销单价元（不同口径）', '项目历史立创SKU', '项目历史MOQ'])
    for row in rows:
        sign, ref, maker, part = (row[n] for n in ['变更','位号','制造商','料号'])
        if part in prices:
            unit, seller, link_id = prices[part]
            url = f'{source}/{supplier[seller]}/{part.replace("#", "-")}/{link_id}'
            subtotals[sign] += Decimal(unit)
        else:
            unit, url = '', ''
            unpriced.append({'ref': ref, 'part': part, 'change': sign})
        price, sku, moq = history.get(part, ('', '', ''))
        if price:
            legacy += Decimal(price) * (1 if sign == '+' else -1)
        writer.writerow([sign, ref, maker, part, unit, url, price, sku, moq])
summary = {
    'currency': 'CNY',
    'basis': 'DigiKey China, 1 piece, 13% VAT inclusive, per installed vehicle quantity',
    'priced_additions': str(subtotals['+']),
    'priced_removals': str(subtotals['-']),
    'priced_net': str(subtotals['+']-subtotals['-']),
    'other_historical_basis_net': str(legacy),
    'unpriced_board_items': unpriced,
    'not_in_board_delta': ['BMS receive/latch implementation', 'service fuse and disconnect',
                           'cable and mating housings', 'assembly and test'],
    'actual_procurement_savings': '0',
}
(run / '净BOM汇总.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
print(json.dumps({k:v for k,v in summary.items() if k != 'unpriced_board_items'}, ensure_ascii=False))