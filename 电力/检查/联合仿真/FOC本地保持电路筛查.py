"""Compare the two shared-LPV receivers with and without local Buck hold."""

import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from 双拒充泄放电路复跑 import OUT, POWER, run_deck


ENET = POWER / 'FOC驱动模块/eda工程文件/导出/当前交付/FOC_G431_P1.enet'
HOLD_END_S = 0.003
PHASES = {
    'cold_precharge': (0.0005, 0.0009),
    'normal_permission': (0.0013, 0.0015),
    'permission_handoff': (0.00155, 0.0018),
    'shared_permission_lost': (0.0018, 0.0021),
    'local_reset': (0.0026, 0.0029),
}
BOARD = '''
R_CLAMP_{board} BUS clamp_{board} 99k
R_VALID_{board} LPV base_{board} 4.747k
R_VALID_PD_{board} base_{board} 0 99k
Q_RELEASE_{board} clamp_{board} base_{board} 0 BC847_PROXY
Q_CLAMP_{board} rt_{board} clamp_{board} 0 BC847_PROXY
R_RT_{board} BUS rt_{board} 277.2k
B_RT_{board} rt_{board} 0 I={{max(0,(V(rt_{board})-2.2)/100)}}
'''
HOLD = '''
V_GPIO_{board} gpio_{board} 0 {waveform}
R_HOLD_{board} gpio_{board} hold_{board} 4.747k
R_HOLD_PD_{board} hold_{board} 0 99k
Q_HOLD_{board} clamp_{board} hold_{board} 0 BC847_PROXY
'''
NORMAL_GPIO = 'PWL(0 0 1.3m 0 1.301m 2.4 2.3m 2.4 2.301m 0)'


def verify_baseline():
    source = json.loads(ENET.read_text(encoding='utf-8-sig'))
    by_ref = {item['props']['Designator']: item for item in source['components'].values()}
    for ref, expected in {
        'Q_BUCK_CLAMP': ['BUCK_CLAMP_BASE', 'GND_SYS', 'BUCK_RT_SD'],
        'Q_BUCK_RELEASE': ['BUCK_RELEASE_BASE', 'GND_SYS', 'BUCK_CLAMP_BASE'],
        'R_BUCK_CLAMP_BUS': ['BUS', 'BUCK_CLAMP_BASE'],
        'R_BUCK_VALID_BASE': ['LOAD_POWER_VALID', 'BUCK_RELEASE_BASE'],
    }.items():
        pins = by_ref[ref]['pinInfoMap']
        actual = [pins[str(n)]['net'] for n in range(1, len(expected) + 1)]
        if (actual != expected if ref.startswith('Q_') else set(actual) != set(expected)):
            raise ValueError(f'Baseline net drift at {ref}')


def run_case(work, bus_v, temperature_c, variant):
    holds = {'baseline': {}, 'hold_A': {'a': NORMAL_GPIO},
             'hold_both': dict.fromkeys('ab', NORMAL_GPIO),
             'gpio_A_stuck_high': {'a': 'DC 2.4'}}[variant]
    circuit = f'''AGTC two-board Buck permission with independent hold candidate
.temp {temperature_c}
V_BUS BUS 0 {bus_v}
V_SAFE SAFE 0 4.75
R_SHARED SAFE OPT_IN 1k
V_LPV_STATE EN 0 PWL(0 0 1m 0 1.001m 5 1.6m 5 1.601m 0)
S_OPTO OPT_IN LPV EN 0 OPTO_PROXY
.model OPTO_PROXY SW(Ron=300 Roff=1G Vt=2 Vh=0)
.model BC847_PROXY NPN(IS=1e-14 BF=100 VAF=80 CJE=8p CJC=4p TF=1n TR=100n)
'''
    circuit += ''.join(BOARD.format(board=board) for board in 'ab')
    circuit += ''.join(HOLD.format(board=board, waveform=waveform) for board, waveform in holds.items())
    circuit += f'''
.control
set noaskquit
set wr_singlescale
tran 1u {HOLD_END_S}
wrdata wave.txt v(rt_a) v(rt_b) v(lpv)
quit
.endc
.end
'''
    wave = run_deck(work, circuit, HOLD_END_S)
    phases = {}
    for phase, (start, end) in PHASES.items():
        rows = [sample[1:] for sample in wave if start <= sample[0] <= end]
        phases[phase] = {name: {'min_v': min(column), 'max_v': max(column)}
                         for name, column in zip(('RT_A', 'RT_B', 'LPV'), zip(*rows))}
    return {'bus_v': bus_v, 'temperature_c': temperature_c, 'variant': variant, 'phases': phases}


def main():
    verify_baseline()
    with TemporaryDirectory(prefix='agtc_foc_hold_') as tmp:
        cases = [run_case(Path(tmp), voltage, temp, variant)
                 for voltage in (18, 22.5, 39.009) for temp in (0, 60)
                 for variant in ('baseline', 'hold_A', 'hold_both', 'gpio_A_stuck_high')]
    for case in cases:
        handoff = case['phases']['permission_handoff']
        for node, holds in (('RT_A', case['variant'] != 'baseline'),
                            ('RT_B', case['variant'] == 'hold_both')):
            if holds and handoff[node]['min_v'] <= 1.05:
                raise ValueError(f'Candidate hold dipped during permission handoff: {case}')
        for phase, (a, b) in {
            'cold_precharge': (case['variant'] == 'gpio_A_stuck_high', False),
            'normal_permission': (True, True),
            'shared_permission_lost': (case['variant'] != 'baseline', case['variant'] == 'hold_both'),
            'local_reset': (case['variant'] == 'gpio_A_stuck_high', False),
        }.items():
            voltages = case['phases'][phase]
            for node, enabled in (('RT_A', a), ('RT_B', b)):
                expected = voltages[node]['min_v'] > 1.05 if enabled else voltages[node]['max_v'] < 0.30
                if not expected:
                    raise ValueError(f'Unexpected RT/SD result at {phase}: {case}')
            if phase != 'normal_permission' and voltages['LPV']['max_v'] >= 0.3:
                raise ValueError(f'Unexpected LPV backfeed at {phase}: {case}')
    result = {
        'qualification': 'HOLD',
        'scope': '两路现行RT/SD夹位加独立开集电极保持的针网比较。模型含通用NPN、等效光耦、1.05 V释放筛查和0.30 V夹低目标。RT有简化偏置，未模拟Buck开关、CPU掉电波形、主路许可物理证明或电机扭矩。',
        'why': 'LPV撤销时原网两路均被夹低。本地支路允许A单独保持，B的LPV和夹位仍维持撤销态。',
        'blocking': 'GPIO卡高反例在预充时会放开A路RT/SD。若要求覆盖GPIO单故障，须另证冷启动/主路确认门控。GPIO复位默认、引脚选择、真实器件全温和功率保持未证明。',
        'sources': {
            'baseline_enet_sha256': hashlib.sha256(ENET.read_bytes()).hexdigest(),
            'TI_LM5008A': 'https://www.ti.com/lit/ds/symlink/lm5008a.pdf',
            'Nexperia_BC847C': 'https://assets.nexperia.com/documents/data-sheet/BC847X_SER.pdf',
        },
        'cases': cases,
    }
    (OUT / 'FOC本地保持电路筛查.json').write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n',
                                                  encoding='utf-8')
    print(f'FOC Buck hold: {len(cases)} cases, expected states verified, HOLD')


if __name__ == '__main__':
    main()
