"""复用稳压关断后续流网表，复算掉电压力及双 FOC 供电下界。"""

import json
import math
from pathlib import Path
import runpy
import shutil
import subprocess
from tempfile import TemporaryDirectory


OUT = Path(__file__).resolve().parent
POWER = OUT.parents[1]
ROOT = POWER.parent
SOURCE = POWER / '稳压模块/仿真/源码/隔离关断后电感续流.py'
R1 = POWER / '稳压模块/检查/候选_预充与超容隔离_R1/候选官方.enet'
CAP_F = 4.86e-3
BUS_START_V = 37.8
BUS_LIMIT_V = 39.009
INJECT_A = 80.0
L_H = 24e-6
TAIL_J = 0.5 * L_H * INJECT_A**2


def compare_isolation(prior):
    cases = []
    with TemporaryDirectory(prefix='agtc_ngspice_') as temp:
        working = Path(temp)
        model = prior['MODEL']
        shutil.copyfile(model, working / model.name)
        for delay_us in (20, 30, 40, 300):
            delay_s = delay_us * 1e-6
            bus_at_open = BUS_START_V + INJECT_A * delay_s / CAP_F
            energy_bound = math.sqrt(bus_at_open**2 + 2 * TAIL_J / CAP_F)
            for current, sc_connected_v in (
                (-INJECT_A, None), (INJECT_A, None),
                (-INJECT_A, 17.6), (-INJECT_A, 21.6),
            ):
                sc_initial_v = sc_connected_v if sc_connected_v is not None else 17.6
                netlist = prior['NETLIST_TEMPLATE'].format(
                    case_id='bounded_R1_tail', l_value=L_H, i_value=current,
                    sw0_value=-1.1187 if current > 0 else bus_at_open + 1.1187,
                    wave_name='tail.txt',
                )
                netlist = netlist.replace('37.8', str(bus_at_open)).replace('5.4m', str(CAP_F))
                netlist = netlist.replace('17.6', str(sc_initial_v))
                if sc_connected_v is not None:
                    netlist = netlist.replace('.control', f'V_SC_STACK SC_FUSED 0 {sc_connected_v}\n.control', 1)
                (working / 'tail.cir').write_text(netlist, encoding='ascii')
                wave = working / 'tail.txt'
                wave.unlink(missing_ok=True)
                run = subprocess.run(
                    [str(prior['EXE']), '-b', 'tail.cir'], cwd=working,
                    capture_output=True, text=True, errors='replace', timeout=60,
                    creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0),
                )
                if run.returncode or 'Error:' in run.stdout + run.stderr or not wave.exists():
                    raise RuntimeError(run.stdout + run.stderr)
                rows = prior['parse_wave'](wave)
                if not rows or rows[-1][0] < prior['TSTOP'] * 0.999:
                    raise RuntimeError('关断后续流波形不完整')
                peak = max(row[1] for row in rows)
                if sc_connected_v is None and peak > energy_bound + 0.001:
                    raise RuntimeError('网表峰值超过本算例的尾能比较上界')
                cases.append({
                    'pre_open_injection_us': delay_us, 'inductor_initial_a': current,
                    'sc_stack_still_connected_v': sc_connected_v,
                    'bus_at_open_v': bus_at_open, 'ngspice_peak_v': peak,
                    'all_inductor_energy_peak_v': energy_bound if sc_connected_v is None else None,
                    'tail_current_at_end_a': rows[-1][4],
                    'margin_to_comparison_v': BUS_LIMIT_V - peak,
                    'voltage_screen': 'PASS' if peak <= BUS_LIMIT_V else 'FAIL',
                    'physical_qualification': 'HOLD' if peak <= BUS_LIMIT_V else 'FAIL_IN_COMPARISON',
                })
    return cases


def compare_peak_supply():
    dual_foc_w = 2 * 1800
    candidate_battery_a = 130
    return [{
        'bus_v': voltage,
        'ideal_dual_foc_total_a': dual_foc_w / voltage,
        'min_sc_bus_output_w_at_candidate_battery_current': max(0, dual_foc_w - voltage * candidate_battery_a),
    } for voltage in (22.5, 27.0, dual_foc_w / candidate_battery_a, 32.4, 37.8)]


def main():
    prior = runpy.run_path(str(SOURCE))
    paths = (SOURCE, prior['MODEL'], prior['EXE'], R1, POWER / 'BMS模块/设计需求.md')
    max_delay = CAP_F * (math.sqrt(BUS_LIMIT_V**2 - 2 * TAIL_J / CAP_F) - BUS_START_V) / INJECT_A
    result = {
        'qualification': 'HOLD',
        'input_sha256': {str(path.relative_to(ROOT)): prior['sha256'](path) for path in paths},
        'isolation_scope': '80A 恒定注入解析段 + 主桥已关断的原厂 Schottky 续流网表。另比较主桥可独立先关而隔离开关仍导通的分支，超容以固定电压源表示。未建模门驱响应、线束尖峰、热、MOS Coss/Qrr 或其他负载源。',
        'ti_timeout_datasheet': 'https://www.ti.com/lit/ds/symlink/tpsi3052-q1.pdf',
        'timing_scope': '20/30µs 为敏感性比较，40µs 为原前段预算，300µs 为 TPSI3052 主电源失电典型比较点，无最大值。',
        'local_bus_f': CAP_F,
        'initial_bus_v': BUS_START_V,
        'comparison_limit_v': BUS_LIMIT_V,
        'isolated_bus_injection_a': INJECT_A,
        'tail_energy_j': TAIL_J,
        'max_injection_duration_us_with_full_tail_energy': max_delay * 1e6,
        'isolation_cases': compare_isolation(prior),
        'dual_foc_scope': '理想功率下界，130A 未获验证。忽略主路及转换损失、其他负载和包压降，持续时间待定。',
        'dual_foc_cases': compare_peak_supply(),
    }
    output = OUT / '超容掉电与双峰值复核.json'
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    print(output.name)


if __name__ == '__main__':
    main()
