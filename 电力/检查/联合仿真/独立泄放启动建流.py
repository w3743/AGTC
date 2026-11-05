"""Run the selected TI macro-models and the existing ST diode in ngspice."""

from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory, gettempdir
from urllib.request import urlopen
from zipfile import ZipFile

from 双拒充泄放电路复跑 import DIODE, EXE, OUT, run_deck


MODELS = {
    'ucc.lib': ('slvmcj6', 'eb78c0ce0d9cf2dd5bbc5f937bbfc6cc38c95215e035feaeb3950672e8c05c6d'),
    'ref.lib': ('slim161a', '97baf1b53f3d07255c485c6f12ac40b4d6c62dacfc727a9abb7ba43f6bb0d057'),
    'cmp.lib': ('snom762a', '8a273cb9af327982356c4cd0071dd9a5183cd84967ed16a1f7c2c9b8db1124e8'),
}
THRESHOLD_V = 38.7382808
BUS_LIMIT_V = 39.009
REGEN_W = 450.03


def prepare_models(work):
    cache = Path(gettempdir()) / 'agtc_dump_approved_ti_macros'
    cache.mkdir(exist_ok=True)
    for file, (archive, expected) in MODELS.items():
        url = f'https://www.ti.com/lit/zip/{archive}'
        cached = cache / file
        raw = cached.read_bytes() if cached.is_file() else b''
        if sha256(raw).hexdigest() != expected:
            with urlopen(url, timeout=20) as response, ZipFile(BytesIO(response.read())) as package:
                lib = next(name for name in package.namelist() if name.lower().endswith('.lib'))
                raw = package.read(lib)
            if sha256(raw).hexdigest() != expected:
                raise ValueError(f'Model source has changed: {url}')
            cached.write_bytes(raw)
        (work / file).write_bytes(raw)
    shutil.copyfile(DIODE, work / 'st.lib')
    (work / '.spiceinit').write_text('set ngbehavior=ps\n', encoding='ascii')


DECK = '''AGTC startup and active brake dump, vendor macros
.include ucc.lib
.include ref.lib
.include cmp.lib
.include st.lib
.options method=gear reltol=1e-4
.temp {temp}
V_HW HW 0 {supply}
V_OLD OLD 0 {old}
V_REGEN REGEN 0 {injection}
V_START SRC 0 {bus}
V_CHG CHG 0 {charger}
S_CHARGE SRC BUS CHG 0 PRECHARGE
.model PRECHARGE SW(Ron=1m Roff=1T Vt=0.5 Vh=0)
C_BUS BUS 0 52.86m
B_INJECT 0 BUS I={{V(REGEN)/max(1,V(BUS))}}
{reference}
R_HI BUS SENSE {r_high}
R_LO SENSE 0 10k
R_HYS CMP_OUT SENSE 6.8Meg
C_SENSE SENSE 0 20p
R_REF_INPUT REF REF_SENSE 10k
C_REF_INPUT REF_SENSE 0 20p
X_COMP SENSE REF_SENSE HW 0 CMP_OUT TLV1811
R_BASE CMP_OUT BASE 22k
R_BASEPD BASE 0 100k
Q_REQ DUMP_OFF_HW BASE 0 BC847_PROXY
.model BC847_PROXY NPN(IS=1e-14 BF=100 VAF=80 CJE=8p CJC=4p TF=1n TR=100n)
R_OFF HW DUMP_OFF_HW 47k
R_READY HW READY {ready_pullup}
R_READY_PD READY 0 {ready_pulldown}
C_READY READY 0 {ready_cap}
{supervisor}
X_UCC DUMP_OFF_HW READY HW 0 GATE_H GATE_L UCC27511A
R_ON GATE_H GATE 10
R_OFF_GATE GATE_L GATE 10
R_GATE_PD GATE 0 10k
C_GATE GATE 0 25n
R_DUMP BUS IND {resistor}
L_DUMP IND SW {inductor}
C_SWITCH SW 0 2n
X_FLYBACK SW BUS SW STPS30100ST
S_DUMP SW 0 GATE 0 POWER_PROXY
.model POWER_PROXY SW(Ron=8m Roff=1G Vt=8 Vh=0.25)
.control
set noaskquit
set wr_singlescale
tran 1u {end}
wrdata wave.txt v(bus) v(hw) v(old) v(ref) v(sense) v(ready) v(dump_off_hw) v(gate) i(l_dump)
quit
.endc
.end
'''

SUPERVISOR = '''B_CTL READY 0 I={{V(READY)*0.005*(3-tanh((V(HW)-11.2)/0.02)-tanh((V(REF)-{minimum_ref})/0.002)-tanh((time-{blank_s})/1u))}}'''
VENDOR_REFERENCE = '''X_REF HW TEMPO 0 TRIM REF_MACRO REF5025
R_REF_ESR REF_MACRO REF_CAP 0.2
C_REF REF_CAP 0 10u
R_TRIM TRIM 0 1G
R_TEMPO TEMPO 0 1G
E_BUFFER REF 0 REF_MACRO 0 1'''


def run_case(work, name, supply, *, old='0', regen=False, supervisor='window', temp=25,
             resistor=1.0, inductor=100e-6, blank_s=0.0, r_high=143_600, end=0.024, ref_profile=None):
    component_supervisor = '' if supervisor == 'unqualified' else SUPERVISOR.format(
        minimum_ref=2.44 if supervisor == 'two_uv_proxy' else 2.49, blank_s=blank_s)
    pullup, pulldown, cap = ('10k', '10k', '1u') if supervisor == 'rc_proxy' else ('47k', '1Meg', '100n')
    deck = DECK.format(supply=supply, old=old,
                       injection='PWL(0 0 8m 0 8.001m 450.03 18m 450.03 18.001m 0)' if regen else '0',
                       charger='PWL(0 1 7.9m 1 7.901m 0)' if regen else '1',
                       bus=37.8, r_high=r_high, temp=temp, supervisor=component_supervisor,
                       reference=VENDOR_REFERENCE if ref_profile is None else f'V_REF REF 0 {ref_profile}',
                       ready_pullup=pullup, ready_pulldown=pulldown, ready_cap=cap,
                       resistor=resistor, inductor=inductor, end=end)
    wave = run_deck(work, deck, end)
    # wrdata columns: time, BUS, HW, old 12 V, REF, SENSE, READY, IN-, VGS, branch current.
    before_regen = [row for row in wave if row[0] < 0.008] if regen else wave
    trigger = next((row[0] for row in wave if regen and row[1] >= THRESHOLD_V), None)
    sense_cross = next((row[0] for row in wave if regen and row[0] >= 0.008 and row[5] >= row[4]), None)
    match = next((row[0] for row in wave if regen and 0.008 <= row[0] < 0.018
                  and row[8] > 8.25 and row[9] >= REGEN_W / row[1]), None)
    first_gate = next((row[0] for row in wave if row[8] > 8.25), None)
    max_bus = max(row[1] for row in wave)
    build_delay = None if match is None or sense_cross is None else match - sense_cross
    return {
        'case': name, 'simulation': 'completed', 'supervisor': supervisor, 'temperature_c': temp, 'wire_inductance_h': inductor,
        'branch_resistance_ohm': resistor, 'rhigh_ohm': r_high, 'supply_profile': supply,
        'reference_profile': ref_profile or 'TI REF5025 macro',
        'old_supply_profile': old, 'max_bus_v': max_bus, 'peak_gate_before_regen_v': max(row[8] for row in before_regen),
        'peak_branch_before_regen_a': max(row[9] for row in before_regen),
        'peak_branch_a': max(row[9] for row in wave), 'reference_final_v': wave[-1][4],
        'first_gate_s': first_gate, 'comparison_threshold_crossing_s': trigger, 'sense_crossing_s': sense_cross,
        'first_power_balance_s': match, 'sense_to_power_balance_s': build_delay,
        'sense_to_power_balance_0p5ms_comparison_met': build_delay is not None and 0 <= build_delay <= 0.0005,
        'voltage_comparison_met': max_bus < BUS_LIMIT_V,
    }


def attempt_case(work, name, supply, **options):
    try:
        return run_case(work, name, supply, **options)
    except RuntimeError as exc:
        if 'Timestep too small' not in str(exc):
            raise
        line = next(line for line in str(exc).splitlines() if 'Timestep too small' in line)
        return {'case': name, 'simulation': 'not_converged', 'supply_profile': supply,
                'options': options, 'reason': line.strip()}


def main():
    with TemporaryDirectory(prefix='agtc_dump_macro_') as tmp:
        work = Path(tmp)
        prepare_models(work)
        fast_reference = 'PWL(0 0 0.1m 0 0.5m 2.445 1.5m 2.445 2m 2.5)'
        cases = [
            attempt_case(work, 'cold_fast_unqualified', 'PWL(0 0 0.1m 0 0.101m 12)', supervisor='unqualified',
                     ref_profile=fast_reference, end=0.004),
            attempt_case(work, 'cold_fast_uv_proxy', 'PWL(0 0 0.1m 0 0.101m 12)', supervisor='two_uv_proxy',
                     blank_s=0.000251, ref_profile=fast_reference, end=0.004),
            attempt_case(work, 'cold_fast_rc_proxy', 'PWL(0 0 0.1m 0 0.101m 12)', supervisor='rc_proxy',
                     blank_s=0.000251, ref_profile=fast_reference, end=0.004),
            attempt_case(work, 'cold_slow_rc_proxy', 'PWL(0 0 0.1m 0 20m 12)', supervisor='rc_proxy',
                     ref_profile='PWL(0 0 6m 0 12m 2.445 21m 2.445 22m 2.5)'),
            attempt_case(work, 'recovery_rc_proxy', 'PWL(0 12 1m 12 1.001m 5 1.2m 5 1.201m 12)',
                     supervisor='rc_proxy', ref_profile='PWL(0 2.5 1m 2.5 1.1m 2.445 2m 2.445 2.5m 2.5)', end=0.006),
            attempt_case(work, 'warm_old_12v_shorted_100uH', '12', old='PWL(0 12 8m 12 8.001m 0)',
                     regen=True, temp=60, ref_profile='2.5', end=0.014),
            attempt_case(work, 'warm_threshold_corner_100uH', '12', old='PWL(0 12 8m 12 8.001m 0)',
                     regen=True, temp=60, ref_profile='2.5', end=0.014, r_high=144_800),
            attempt_case(work, 'warm_old_12v_shorted_40uH', '12', old='PWL(0 12 8m 12 8.001m 0)',
                     regen=True, temp=0, inductor=40e-6, ref_profile='2.5', end=0.014),
            attempt_case(work, 'warm_ref5025_connected_probe', '12', regen=True, temp=60, end=0.014),
            attempt_case(work, 'warm_no_independent_supply', '0', regen=True, supervisor='unqualified',
                     ref_profile='0', end=0.014),
        ]
    result = {
        'qualification': 'HOLD',
        'scope': 'TI UCC27511A、TLV1811 与 STPS30100ST 厂商宏模型。REF5025 联接宏模型未收敛，冷启动采用给定参考波形，已就绪算例采用 2.5 V 参考；已就绪算例仅覆盖首个接管窗口，延长仿真未收敛。独立电源是注入波形，监督是行为近似，MOS、门极负载、请求管为代理。比较器 20 pF 和参考输入 10 kΩ/20 pF 为数值筛查参数，未进入候选采购。旧12V短接只作为域输入，不代表短接真实母线或故障阻抗。',
        'models': {f'https://www.ti.com/lit/zip/{model}': sha for model, sha in MODELS.values()},
        'local_input_sha256': {str(file.relative_to(OUT.parents[2])): sha256(file.read_bytes()).hexdigest()
                              for file in (DIODE, EXE)},
        'cases': cases,
    }
    positive = next(case for case in cases if case['case'] == 'warm_old_12v_shorted_100uH')
    negative = next(case for case in cases if case['case'] == 'warm_no_independent_supply')
    if not positive.get('sense_to_power_balance_0p5ms_comparison_met') or negative.get('voltage_comparison_met') is not False:
        raise ValueError('The conditional takeover and no-power control must both be reproduced.')
    OUT.joinpath('独立泄放启动建流.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n',
                                                encoding='utf-8')
    print(json.dumps({'qualification': 'HOLD', 'case_count': len(cases)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
