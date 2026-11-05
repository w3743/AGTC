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
B_REGEN 0 BUS I={{390/max(V(BUS),1)}}
R_DUMP BUS IND {dump_r}
L_WIRE IND SW {wire_l} IC=0
C_SWITCH SW 0 2n IC={start_v}
X_FLYBACK SW BUS SW STPS30100ST
V_GATE G 0 PULSE(0 10 160u 2n 2n 30u 50u)
S_DUMP SW 0 G 0 DUMP
.model DUMP SW(Ron=1m Roff=1G Vt=5 Vh=0)
.control
set noaskquit
set wr_singlescale
tran 50n 1m uic
wrdata wave.txt v(bus) i(l_wire)
quit
.endc
.end
'''


def run_case(work, start_v, dump_r, wire_l):
    netlist = DECK.format(start_v=start_v, dump_r=dump_r, wire_l=wire_l)
    (work / 'dump.cir').write_text(netlist, encoding='ascii')
    wave = work / 'wave.txt'
    wave.unlink(missing_ok=True)
    proc = subprocess.run([str(EXE), '-b', 'dump.cir'], cwd=work, capture_output=True,
                          text=True, errors='replace', timeout=60,
                          creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
    if proc.returncode or 'Error:' in proc.stdout + proc.stderr or not wave.is_file():
        raise RuntimeError(proc.stdout + proc.stderr)
    samples = [tuple(map(float, line.split())) for line in wave.read_text(encoding='ascii').splitlines()]
    if not samples or samples[-1][0] < 0.000999:
        raise RuntimeError(f'泄放接管波形未跑完\n{proc.stdout[-1000:]}\n{proc.stderr[-1000:]}')
    return {'trigger_v': start_v, 'dump_r_ohm': dump_r, 'wire_l_h': wire_l,
            'bus_peak_v': max(row[1] for row in samples), 'bus_at_1ms_v': samples[-1][1],
            'wire_current_peak_a': max(row[2] for row in samples),
            'bus_voltage_comparison': 'PASS' if max(row[1] for row in samples) <= 39.009 else 'FAIL'}


def main():
    with TemporaryDirectory(prefix='agtc_dump_') as temp:
        work = Path(temp)
        shutil.copyfile(DIODE, work / 'st.lib')
        cases = [run_case(work, threshold, resistance, inductance)
                 for threshold, resistance, inductance in ((38.50, 1.0, 40e-6), (38.50, 1.0, 100e-6),
                                                          (38.50, 0.779, 40e-6), (38.738, 1.0, 40e-6))]
    result = {'qualification': 'HOLD',
              'scope': '从真实触发起的1ms开关级功率支路比较。390W恒定净回灌、52.86mF、从160µs起每50µs有30µs有效导通。采用厂家Schottky宏模型，2nF为收敛假设。仅比较BUS包络和线束电流，理想栅极开关不证明控制、电源、实装MOS尖峰、长时恒压或热。',
              'input_sha256': {str(path.relative_to(POWER.parent)): hashlib.sha256(path.read_bytes()).hexdigest()
                               for path in (EXE, DIODE)},
              'cases': cases}
    Path(__file__).with_suffix('.json').write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
