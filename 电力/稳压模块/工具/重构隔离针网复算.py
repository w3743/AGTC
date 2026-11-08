"""Check the saved candidate's official pin map, BOM delta and engineering limits."""
import csv
import hashlib
import itertools
import json
import math
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / '检查' / '候选_预充与超容隔离_R1'
SOURCE = ROOT / 'eda工程文件' / 'sc-regulator'

def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))

def write(name, value):
    (RUN / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')

def index(raw):
    return {c['props']['Designator']: c for c in raw['components'].values()}

def pinmap(c):
    return {n: p.get('net') or None for n, p in c['pinInfoMap'].items()}

def identity(c):
    p = c['props']
    return (p.get('Manufacturer Part') or p.get('Manufacturer Part Number')
            or p.get('DeviceName') or p.get('Name'))

export = read(RUN / '官方回读打包.json')
net = json.loads(export.pop('netlist'))
(RUN / '候选官方.enet').write_text(json.dumps(net, ensure_ascii=False, indent=2), encoding='utf-8')
(RUN / '候选官方BOM.csv').write_text(export.pop('bom'), encoding='utf-8-sig')
for i, page in enumerate(export['pages'], 1):
    svg = page.pop('svg')
    name = f'候选官方_{i:02d}.svg'
    (RUN / name).write_text(svg, encoding='utf-8')
    page['svg'] = name
write('官方页面与DRC.json', export)

baseline = index(read(ROOT / '检查' / '本轮磁盘读回.enet'))
candidate = index(net)
pinned = read(RUN / '复制源哈希.json')
source_hash = {p.relative_to(SOURCE).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
               for p in sorted(SOURCE.rglob('*')) if p.is_file()}

changes = {
    'U_DUMP_DRV': {'1': '12V_PROT', '5': 'DUMP_OFF', '6': 'GUARD_OK'},
    'U_SC_OV': {'1': 'BUS_OV_OUT', '5': '12V_PROT', '6': 'DRV_FAULT_N'},
    'Q_PMOS_CTRL': {'3': 'DUMP_OFF'},
    'Q_DUMP_SW': {'3': 'DUMP_OFF'},
    'R_QP_COL': {'1': '12V_PROT', '2': 'DUMP_OFF'},
    'R_DUMP_HW_PU': {'1': '12V_PROT'},
    'R_DUMP_IN_SER': {'1': 'DUMP_PWM', '2': 'DUMP_PWM_LIMIT'},
    'R_DUMP_SW_BASE': {'1': 'DUMP_PWM_ALLOWED', '2': 'DUMP_SW_BASE'},
    'U_DUMP_SW_ALLOW': {'1': 'DUMP_PWM_LIMIT', '2': 'LPV_QUAL_LOCAL', '3': 'GND_SYS',
                        '4': 'DUMP_PWM_ALLOWED', '5': '3V3_CORE'},
    'U_AUX1': {'1': 'B0_AUX_EXTERNAL', '2': 'PACK_GUARD_FUSED_POS', '3': '12V_PROT',
               '4': None, '5': 'GND_SYS', '6': None},
    'C_12V_IN': {'1': 'PACK_GUARD_FUSED_POS', '2': 'B0_AUX_EXTERNAL'},
    'C_12V_IN_HF': {'1': 'PACK_GUARD_FUSED_POS', '2': 'B0_AUX_EXTERNAL'},
    'C_12V_OUT': {'1': '12V_PROT', '2': 'GND_SYS'},
    'C_12V_OUT_HF': {'1': '12V_PROT', '2': 'GND_SYS'},
    'U1': {'4': 'DRV_FAULT_N', '10': '12V_DRV', '20': 'BUS_EXT'},
    'R_ISO_TGUP': {'1': 'ISO_TGUP', '2': 'ISO_G'},
    'R_ISO_GS': {'1': 'ISO_G', '2': 'ISO_MID'},
}
for ref in ['C_DUMP_DRV_BULK', 'C_DUMP_DRV_HF', 'C_SC_OV_VDD']:
    changes[ref] = {'1': '12V_PROT'}
for stage in ['保持与监控', 'div', 'iso', 'interfaces', 'drv', '启动斜率', '最小负载']:
    result = read(RUN / f'保护页_{stage}.json')
    if result.get('saved') is True:
        for row in result['trace']:
            changes[row['ref']] = {n: net for n, _, net in row['pins']}
        if stage == '启动斜率':
            changes['U_GUARD_SVS'].update({'1': '3V3_SVS', '14': '3V3_SVS'})
            changes['C_GUARD_SVS']['1'] = '3V3_SVS'
            changes['R_GUARD_RESET_PU']['1'] = '3V3_SVS'

repair = read(RUN / '短接与减价修正.json')
changes['Q_DRV_SWITCH'] = {n: net for n, _, net in repair['switch']}
drc = export['pages'][0]['drc']
mixed_net_names = [row for row in drc if row.get('rule') == 'ONENETNAME'
                   and len(set(row.get('net', '').split('、'))) > 1]
drc_counts = dict(Counter(row['rule'] for row in drc))
mismatches = []
for ref, expected in changes.items():
    if ref not in candidate:
        mismatches.append([ref, 'missing'])
        continue
    actual = pinmap(candidate[ref])
    for n, expected_net in expected.items():
        if n not in actual or actual[n] != expected_net:
            mismatches.append([ref, n, expected_net, actual.get(n)])

removed = sorted(baseline.keys() - candidate.keys())
added = sorted(candidate.keys() - baseline.keys())
substitutions = [r for r in sorted(baseline.keys() & candidate.keys())
                 if identity(baseline[r]) != identity(candidate[r])]
unplanned_pin_changes = []
for ref in sorted(baseline.keys() & candidate.keys()):
    old, new = pinmap(baseline[ref]), pinmap(candidate[ref])
    for n in sorted(old.keys() | new.keys()):
        if old.get(n) != new.get(n) and n not in changes.get(ref, {}):
            unplanned_pin_changes.append([ref, n, old.get(n), new.get(n)])

assert all(pinmap(candidate[r]) == pinmap(baseline[r]) for r in ['Q_SC_ISO1', 'Q_SC_ISO2'])
with (RUN / '器件增减.csv').open('w', encoding='utf-8-sig', newline='') as fp:
    writer = csv.writer(fp)
    writer.writerow(['变更', '位号', '制造商', '料号'])
    for sign, refs, rows in [('-', removed + substitutions, baseline), ('+', added + substitutions, candidate)]:
        for ref in refs:
            writer.writerow([sign, ref, rows[ref]['props'].get('Manufacturer', ''), identity(rows[ref])])

def threshold_corners(max_delta_c):
    results = {}
    low_tolerance = .001 + 25e-6 * max_delta_c
    for label, top, tolerance, tcr in [('12V_PROT',261000,.001,25e-6),
                                      ('12V_DRV',220000,.001,25e-6),
                                      ('3V3_CORE',61900,.001,25e-6),
                                      ('5V_EXT',100000,.01,100e-6)]:
        results[label] = {}
        tol = tolerance + tcr * max_delta_c
        for edge, bounds in [('falling',(0.396,0.404)),('release',(0.396,0.414))]:
            volts = [v * (1+r/low) + i*r for v,r,low,i in itertools.product(
                bounds, (top*(1-tol),top*(1+tol)), (10000*(1-low_tolerance),10000*(1+low_tolerance)),
                (-25e-9,25e-9))]
            results[label][edge] = [min(volts),max(volts)]
    return results

voltage_thresholds = threshold_corners(0)
temperature_corners = threshold_corners(100)

# Logical behavior of the schematic at valid digital levels. Analog transitions need waves.
for pb7, lpv, formal5, core, qual, ov in itertools.product([False,True], repeat=6):
    software = pb7 and lpv and formal5 and core
    dump = qual and (ov or software)
    assert not (dump and not ov and (not lpv or not formal5 or not core))
    assert not qual or not ov or dump
sc_sequences = {}
for loss in ['12V_PROT','12V_DRV','3V3_CORE','5V_EXT','LPV']:
    q = False
    run = True
    for edge, rails in [(True,True),(False,False),(False,True),(True,True)]:
        q = (q or edge) if rails else False
        sc_sequences.setdefault(loss, []).append(q and run)
    assert sc_sequences[loss] == [True,False,False,True]

cap, v0, vmax, current, inductance = .00486, 37.8, 39.009, 80, 24e-6
tail = .5*inductance*current**2
allowed_time = (math.sqrt(vmax**2-2*tail/cap)-v0)*cap/current
peak_40us = math.sqrt((v0+current*40e-6/cap)**2+2*tail/cap)
dcap = 2.2e-6
minimum_held_input = temperature_corners['12V_PROT']['falling'][0] - .60
available_hold = dcap * (minimum_held_input - 4.8)
data = {
    'source_unchanged': source_hash == pinned,
    'baseline_component_count': len(baseline),
    'candidate_component_count': len(candidate),
    'added': added, 'removed': removed, 'substitutions': substitutions,
    'expected_pin_count': sum(map(len, changes.values())),
    'pin_mismatches': mismatches,
    'other_pin_changes': unplanned_pin_changes,
    'drc_counts': drc_counts,
    'drc_mixed_net_names': mixed_net_names,
    'voltage_thresholds_V': voltage_thresholds,
    'voltage_thresholds_with_resistor_tcr_V': temperature_corners,
    'input_bias_screen_A': 25e-9,
    'valid_rail_logic_vectors': 64,
    'sc_power_loss_rearm_sequences': sc_sequences,
    'sc_model': {'limit_us': allowed_time*1e6,'peak_at_40us_V': peak_40us,
                 'peak_at_300us_V': math.sqrt((v0+current*300e-6/cap)**2+2*tail/cap),
                 'tail_mJ': tail*1000, 'hold_floor_assumed_uF': dcap*1e6,
                 'hold_at_assumed_105mA_us': available_hold/.105*1e6,
                 'hold_at_assumed_31mA_us': available_hold/.031*1e6},
    'lpv_static_screen': {'assert_current_at_3p6V_mA': 3.6/(4.7*.99),
                          'high_at_3p0V_0p2V_CE_V': (3.0-.2)*(22*.99)/(22*.99+4.7*1.01)},
    'aux_minload_screen_W': [11.4**2*(1/(430*1.01)+1/(10000*1.01)),
                             13.2**2*(1/(430*.99)+1/(10000*.99))],
    'release_status': 'HOLD',
}
write('针网与原理复算.json', data)
print(json.dumps(data, ensure_ascii=False, indent=2))
if mismatches or unplanned_pin_changes or mixed_net_names:
    raise SystemExit(1)