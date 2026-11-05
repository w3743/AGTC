"""在理想栅极时序下复跑早泄放的头几周期，保留真实续流方向。"""

import hashlib
import json
from pathlib import Path
import shutil
import subprocess
from tempfile import TemporaryDirectory


OUT = Path(__file__).resolve().parent
POWER = OUT.parents[1]
EXE = POWER / 'FOC驱动模块/仿真/tools/ngspice-47-portable/Spice64/bin/ngspice_con.exe'
DIODE = POWER / '稳压模块/仿真/输入/厂家模型/ST_Schottky_PSpice_v6/ST_POWER_SCHOTTKY_V10.LIB'
DECK = '''AGTC dual-reject dump first cycles
.include st.lib
.options method=gear reltol=1e-5
.ic V(SW)={start_v} V(IND)={start_v}
C_BUS BUS 0 52.86m IC={start_v}
B_REGEN 0 BUS I={{{power_w}/max(V(BUS),1)}}
R_DUMP BUS IND {dump_r}
L_WIRE IND SW {wire_l} IC=0
C_SWITCH SW 0 2n IC={start_v}
X_FLYBACK SW BUS SW STPS30100ST
{gate_control}
S_DUMP SW 0 G 0 DUMP
.model DUMP SW(Ron=1m Roff=1G Vt=5 Vh=0)
.control
set noaskquit
set wr_singlescale
tran 50n {end_s} uic
wrdata wave.txt v(bus) i(l_wire)
quit
.endc
.end
'''


def run_deck(work, netlist, end_s):
    (work / 'dump.cir').write_text(netlist, encoding='ascii')
    wave = work / 'wave.txt'
    wave.unlink(missing_ok=True)
    proc = subprocess.run([str(EXE), '-b', 'dump.cir'], cwd=work, capture_output=True,
                          text=True, errors='replace', timeout=60,
                          creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
    if proc.returncode or 'Error:' in proc.stdout + proc.stderr or not wave.is_file():
        raise RuntimeError(proc.stdout + proc.stderr)
    samples = [tuple(map(float, line.split())) for line in wave.read_text(encoding='ascii').splitlines()]
    if not samples or samples[-1][0] < end_s - 1e-6:
        raise RuntimeError(f'泄放接管波形未跑完\n{proc.stdout[-1000:]}\n{proc.stderr[-1000:]}')
    return samples


def run_case(work, start_v, dump_r, wire_l, power_w=390, gate_delay_s=160e-6, end_s=1e-3,
             release_v=None):
    gate_control = f'V_GATE G 0 PULSE(0 10 {gate_delay_s} 2n 2n 30u 50u)'
    if release_v is not None:
        gate_control = f'''V_GUARD GUARD 0 10
S_CMD GUARD RAW BUS 0 CMP
R_RAW RAW 0 1k
.model CMP SW(Ron=1 Roff=1G Vt={(start_v + release_v) / 2} Vh={(start_v - release_v) / 2})
B_GATE G 0 V={{V(RAW)*u(time-{gate_delay_s})}}'''
    netlist = DECK.format(start_v=start_v, dump_r=dump_r, wire_l=wire_l,
                          power_w=power_w, gate_control=gate_control, end_s=end_s)
    samples = run_deck(work, netlist, end_s)
    first_gate_end = gate_delay_s + 30e-6 if release_v is None else end_s
    first_source_current_match = next(
        (time for time, voltage, current in samples
         if gate_delay_s <= time < first_gate_end and current >= power_w / voltage),
        None,
    )
    return {'trigger_v': start_v, 'dump_r_ohm': dump_r, 'wire_l_h': wire_l,
            'net_injection_w': power_w, 'gate_delay_s': gate_delay_s, 'end_s': end_s,
            'hysteresis_release_v': release_v,
            'first_source_current_match_s': first_source_current_match,
            'bus_peak_v': max(row[1] for row in samples), 'bus_at_end_v': samples[-1][1],
            **({'bus_at_1ms_v': samples[-1][1]} if end_s == 1e-3 else {}),
            'wire_current_peak_a': max(row[2] for row in samples),
            'bus_voltage_comparison': 'PASS' if max(row[1] for row in samples) <= 39.009 else 'FAIL'}


def run_release_case(work, gate_off_s):
    release_v, resistance, end_s = 37.88849391597611, 0.779, 0.0003
    initial_current = release_v / (resistance + 0.001)
    initial_drain_v = initial_current * 0.001
    gate = f'V_GATE G 0 PWL(0 10 {gate_off_s} 10 {gate_off_s + 2e-9} 0)'
    netlist = DECK.replace('.ic V(SW)={start_v} V(IND)={start_v}',
                           '.ic V(SW)={drain_v} V(IND)={drain_v}').replace(
                           'C_SWITCH SW 0 2n IC={start_v}', 'C_SWITCH SW 0 2n IC={drain_v}').replace(
                           'L_WIRE IND SW {wire_l} IC=0', 'L_WIRE IND SW {wire_l} IC={current}')
    samples = run_deck(work, netlist.format(start_v=release_v, drain_v=initial_drain_v,
                       dump_r=resistance, wire_l=100e-6, current=initial_current, power_w=0,
                       gate_control=gate, end_s=end_s), end_s)
    lowest = min(voltage for _, voltage, _ in samples)
    return {'release_start_v': release_v, 'initial_branch_a': initial_current,
            'gate_off_delay_s': gate_off_s, 'bus_min_v': lowest,
            'over_37p8_v_comparison': lowest > 37.8}


def main():
    with TemporaryDirectory(prefix='agtc_dump_') as temp:
        work = Path(temp)
        shutil.copyfile(DIODE, work / 'st.lib')
        cases = [run_case(work, threshold, resistance, inductance)
                 for threshold, resistance, inductance in ((38.50, 1.0, 40e-6), (38.50, 1.0, 100e-6),
                                                          (38.50, 0.779, 40e-6), (38.738, 1.0, 40e-6))]
        fault_window_cases = [
            run_case(work, 38.7382808, 1.0, inductance, power_w=450.03,
                     gate_delay_s=delay, end_s=2e-3)
            for inductance in (40e-6, 100e-6)
            for delay in (450e-6, 500e-6, 1300e-6)
        ]
        hysteresis_cases = [
            run_case(work, 38.7382808, 1.0, inductance, power_w=450.03,
                     gate_delay_s=delay, end_s=2e-3, release_v=38.504)
            for inductance in (40e-6, 100e-6)
            for delay in (450e-6, 500e-6, 1300e-6)
        ]
        release_cases = [run_release_case(work, delay) for delay in (40e-6, 80e-6, 100e-6)]
    result = {'qualification': 'HOLD',
              'scope': '从真实触发起的1ms开关级功率支路比较。390W恒定净回灌、52.86mF、从160µs起每50µs有30µs有效导通。采用厂家Schottky宏模型，2nF为收敛假设。仅比较BUS包络和线束电流，理想栅极开关不证明控制、电源、实装MOS尖峰、长时恒压或热。',
              'input_sha256': {str(path.relative_to(POWER.parent)): hashlib.sha256(path.read_bytes()).hexdigest()
                               for path in (EXE, DIODE)},
              'cases': cases,
              'fault_window_scope': '450.03 W 与最迟硬门槛的头 2 ms 对照。模拟旧 12 V 消失时独立驱动仍按给定时序出脉冲。只测功率支路在条件成立时的波形，独立电源、真实逻辑和器件延时未计。',
              'fault_window_cases': fault_window_cases,
              'hysteresis_scope': '450.03 W 同一应力的迟滞对照。首轮注入给定延迟，随后理想门限立即开关，释放角38.504 V为静态筛查上角。未绑定实际门槛与其同一单位的温漂角，也未证明迟滞周期的真实时序、尖峰与热。',
              'hysteresis_cases': hysteresis_cases,
              'release_scope': '退出对照以最低释放角、0.779 Ω、100 µH、接近支路静态峰流及零回灌起算，母线不接电池。若在关断前跌破37.8 V，实车满电电池仍允许放电时可能由电池给制动支路送电。理想关栅与无分布阻抗只用于筛查。',
              'release_cases': release_cases}
    Path(__file__).with_suffix('.json').write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
