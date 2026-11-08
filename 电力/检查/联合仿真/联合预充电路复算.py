"""用 ngspice 交叉核对四类板的公共母线预充比较点。"""

import hashlib
import json
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory


OUT = Path(__file__).resolve().parent
POWER = OUT.parents[1]
EXE = POWER / 'FOC驱动模块/仿真/tools/ngspice-47-portable/Spice64/bin/ngspice_con.exe'
NETLIST = '''AGTC precharge cross-check
VPACK pack 0 37.8
BPRE pack bus I={{u(time-{startup_s})*min({limit_a},max(0,V(pack,bus)/{path_ohm}))}}
BLOAD bus 0 I={{u(time-{startup_s})*{load_a}*min(1,max(0,V(bus)/0.001))}}
CFOC_A bus 0 36.07068m IC=0
CFOC_B bus 0 36.07068m IC=0
CREG bus 0 7.03872m IC=0
CPUMP bus 0 603.744u IC=0
.control
set noaskquit
set wr_singlescale
tran 100u {end_s} uic
wrdata bus.txt v(bus)
quit
.endc
.end
'''


def run(work, *, startup_s, limit_a, path_ohm, load_a, end_s, gap_v):
    netlist = NETLIST.format(startup_s=startup_s, limit_a=limit_a, path_ohm=path_ohm,
                             load_a=load_a, end_s=end_s)
    (work / 'pre.cir').write_text(netlist, encoding='ascii')
    output = work / 'bus.txt'
    output.unlink(missing_ok=True)
    proc = subprocess.run([str(EXE), '-b', 'pre.cir'], cwd=work, capture_output=True,
                          text=True, errors='replace', timeout=60,
                          creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
    if proc.returncode or 'Error:' in proc.stdout + proc.stderr or not output.is_file():
        raise RuntimeError(proc.stdout + proc.stderr)
    wave = [tuple(map(float, line.split())) for line in output.read_text(encoding='ascii').splitlines()]
    if not wave or wave[-1][0] < end_s * 0.999:
        raise RuntimeError('预充波形不完整')
    crossing_s = next((t for t, bus in wave if bus >= 37.8 - gap_v), None)
    return {'gap_comparison_v': gap_v, 'load_a': load_a, 'end_s': end_s,
            'gap_at_end_v': 37.8 - wave[-1][1], 'gap_crossing_s': crossing_s}


def main():
    with TemporaryDirectory(prefix='agtc_pre_') as temp:
        work = Path(temp)
        legacy = run(work, startup_s=0, limit_a=1000, path_ohm=8.726,
                     load_a=0, end_s=2.738, gap_v=0.5)
        candidates = [run(work, startup_s=0.075, limit_a=0.791696, path_ohm=0.30,
                          load_a=load, end_s=5.593, gap_v=0.076) for load in (0.099, 0.100)]
    for case in candidates:
        case['total_with_proof_budget_s'] = case['gap_crossing_s'] + 0.191 + 0.010 + 0.060
        case['margin_to_earliest_hard_window_s'] = 5.593 - case['total_with_proof_budget_s']
        case['model_screen'] = 'PASS' if case['margin_to_earliest_hard_window_s'] >= 0 else 'FAIL'
    legacy['model_screen'] = 'PASS' if legacy['gap_crossing_s'] is not None else 'FAIL'
    sources = (EXE, POWER / 'BMS模块/设计细节.md', POWER / 'BMS模块/仿真/结果/跨板母线预充.json')
    result = {
        'qualification': 'HOLD',
        'source': '电力/BMS模块/设计细节.md，四板共用的首版上电配置复核表',
        'input_sha256': {str(path.relative_to(POWER.parent)): hashlib.sha256(path.read_bytes()).hexdigest()
                         for path in sources},
        'scope': '仅电压／时序交叉核对。各板容差高角共79.783824mF，泵仍保守计入预充，外部源与负载禁能、候选恒流与启动延时均作为模型输入。未验证实际预充吸流包络、MOS SOA、硬计时或低阻证明。',
        'legacy_screen': legacy,
        'candidate_screen': candidates,
    }
    output = OUT / '联合预充电路复算.json'
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    print(output.name)


if __name__ == '__main__':
    main()
